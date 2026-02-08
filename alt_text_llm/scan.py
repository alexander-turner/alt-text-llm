"""
Scan markdown files for assets without meaningful alt text.

This script produces a JSON work-queue.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import unquote

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from markdown_it.token import Token

from alt_text_llm import utils


@dataclass(slots=True)
class QueueItem:
    """Represents a single asset lacking adequate alt text."""

    markdown_file: str
    asset_path: str
    line_number: int  # 1-based, must be positive
    context_snippet: str

    def __post_init__(self) -> None:
        if self.line_number <= 0:
            raise ValueError("line_number must be positive")

    def to_json(self) -> dict[str, str | int]:  # pylint: disable=C0116
        return asdict(self)


def _create_queue_item(
    md_path: Path,
    asset_path: str,
    line_number: int,
    lines: Sequence[str],
) -> QueueItem:
    return QueueItem(
        markdown_file=str(md_path),
        asset_path=asset_path,
        line_number=line_number,
        context_snippet=utils.paragraph_context(lines, line_number - 1),
    )


_PLACEHOLDER_ALTS: set[str] = {
    "img",
    "image",
    "photo",
    "placeholder",
    "screenshot",
    "picture",
}

_PLACEHOLDER_VIDEO_LABELS: set[str] = {
    "video",
    "movie",
    "clip",
    "media",
    "content",
    "placeholder",
}

# Common image and video extensions for wikilink detection
# Using tuple for deterministic ordering in parameterized tests
WIKILINK_ASSET_EXTENSIONS: tuple[str, ...] = (
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".mov",
    ".png",
    ".svg",
    ".webm",
    ".webp",
)


def _is_alt_meaningful(alt: str | None) -> bool:
    if alt is None:
        return False
    alt_stripped = alt.strip().lower()
    return bool(alt_stripped) and alt_stripped not in _PLACEHOLDER_ALTS


def _iter_media_tokens(tokens: Sequence[Token]) -> Iterable[Token]:
    """Yield all tokens (including nested children) that correspond to
    images or videos."""

    stack: list[Token] = list(tokens)
    while stack:
        token = stack.pop()

        if token.type == "image":
            yield token
            continue

        if token.type in {"html_inline", "html_block"}:
            content_lower = token.content.lower()
            if "<img" in content_lower or "<video" in content_lower:
                yield token
                continue

        if token.type == "inline":
            content_lower = token.content.lower()
            # Yield inline tokens for wikilink images
            if "![[" in content_lower:
                yield token
            # Also yield inline tokens containing <video> so the full
            # HTML (e.g. <video><source src="..."></video>) is available.
            # markdown-it splits inline HTML into fragments, so child
            # html_inline tokens may not have the complete tag.
            if "<video" in content_lower:
                yield token

        if token.children:
            stack.extend(token.children)


def _is_video_label_meaningful(label: str | None) -> bool:
    if label is None:
        return False
    label_stripped = label.strip().lower()
    return bool(label_stripped) and label_stripped not in _PLACEHOLDER_VIDEO_LABELS


def _extract_html_img_info(token: Token) -> list[tuple[str, str | None]]:
    """Return list of (src, alt) pairs for each <img> within the token."""
    soup = BeautifulSoup(token.content, "html.parser")
    infos: list[tuple[str, str | None]] = []

    for img in soup.find_all("img"):
        src = img.get("src") or None
        alt_raw = img.get("alt")
        alt = str(alt_raw) if alt_raw is not None else None
        if src:
            infos.append((str(src), alt))

    return infos


def _extract_html_video_info(token: Token) -> list[tuple[str, dict[str, str | None]]]:
    """Return list of (src, accessibility_attrs) for each <video>.
    Extracts src from video tag or first <source> child.
    Returns dict with aria-label, title, aria-describedby values.
    """
    soup = BeautifulSoup(token.content, "html.parser")
    infos: list[tuple[str, dict[str, str | None]]] = []

    for video in soup.find_all("video"):
        src_raw = video.get("src")
        if not src_raw:
            source = video.find("source")
            src_raw = source.get("src") if source else None

        if src_raw:
            accessibility_attrs: dict[str, str | None] = {
                "aria_label": video.get("aria-label") or None,
                "title": video.get("title") or None,
                "aria_describedby": video.get("aria-describedby") or None,
            }
            infos.append((str(src_raw), accessibility_attrs))

    return infos


def _get_line_number(token: Token, lines: Sequence[str], search_snippet: str) -> int:
    if token.map:
        return token.map[0] + 1

    # Build candidate snippets: exact, URL-decoded, without parens, decoded without parens.
    # markdown-it URL-encodes Unicode chars in src attrs, so we try decoded variants.
    candidates = [search_snippet, unquote(search_snippet)]
    if search_snippet.startswith("(") and search_snippet.endswith(")"):
        inner = search_snippet[1:-1]
        candidates += [inner, unquote(inner)]

    for candidate in dict.fromkeys(candidates):
        for idx, ln in enumerate(lines):
            if candidate in ln:
                return idx + 1

    raise ValueError(f"Could not find asset '{search_snippet}' in markdown file")


def _handle_md_asset(
    token: Token, md_path: Path, lines: Sequence[str]
) -> list[QueueItem]:
    """
    Process a markdown ``image`` token.

    Args:
        token: The ``markdown_it`` token representing the asset.
        md_path: Current markdown file path.
        lines: Contents of *md_path* split by lines.

    Returns:
        Zero or one-element list containing a ``QueueItem`` for assets with
        missing or placeholder alt text.
    """

    src_attr = token.attrGet("src") or None
    alt_text = token.content or None
    if not src_attr or _is_alt_meaningful(alt_text):
        return []

    # markdown-it URL-encodes Unicode chars in src; decode for the queue item
    decoded_src = unquote(src_attr)
    line_no = _get_line_number(token, lines, f"({src_attr})")
    return [_create_queue_item(md_path, decoded_src, line_no, lines)]


def _handle_html_asset(
    token: Token, md_path: Path, lines: Sequence[str]
) -> list[QueueItem]:
    """
    Process an ``html_inline`` or ``html_block`` token containing ``<img>``.

    Args:
        token: Token potentially containing one or more ``<img>`` tags.
        md_path: Current markdown file path.
        lines: Contents of *md_path* split by lines.

    Returns:
        List of ``QueueItem`` instances—one for each offending ``<img>``.
    """

    items: list[QueueItem] = []
    for src_attr, alt_text in _extract_html_img_info(token):
        # In HTML, alt="" explicitly marks an image as decorative
        if alt_text is not None and alt_text.strip() == "":
            continue
        if _is_alt_meaningful(alt_text):
            continue

        line_no = _get_line_number(token, lines, src_attr)
        items.append(_create_queue_item(md_path, src_attr, line_no, lines))

    return items


def _handle_html_video(
    token: Token, md_path: Path, lines: Sequence[str]
) -> list[QueueItem]:
    """
    Process an ``html_inline`` or ``html_block`` token containing ``<video>``.

    Args:
        token: Token potentially containing one or more ``<video>`` tags.
        md_path: Current markdown file path.
        lines: Contents of *md_path* split by lines.

    Returns:
        List of ``QueueItem`` instances—one for each ``<video>`` lacking accessibility.
    """
    items: list[QueueItem] = []

    for src_attr, accessibility_attrs in _extract_html_video_info(token):
        # Check if any accessibility attribute is present and meaningful
        has_accessibility = any(
            _is_video_label_meaningful(attr) for attr in accessibility_attrs.values()
        )
        if has_accessibility:
            continue

        line_no = _get_line_number(token, lines, src_attr)
        items.append(_create_queue_item(md_path, src_attr, line_no, lines))

    return items


def _iter_wikilink_images(content: str) -> Iterable[tuple[str, str | None]]:
    """Yield (src, alt) pairs for each Obsidian-style wikilink image.

    Supports:
      - ![[path.ext]]
      - ![[path.ext|alt]]
    """

    idx = 0
    while True:
        start = content.find("![[", idx)
        if start == -1:
            return

        end = content.find("]]", start + 3)
        if end == -1:
            return

        inner = content[start + 3 : end]
        if not inner:
            idx = end + 2
            continue

        if "|" in inner:
            src, alt = inner.split("|", 1)
        else:
            src, alt = inner, None

        src = src.strip()
        alt = (alt.strip() or None) if alt else None

        if src:
            yield src, alt

        idx = end + 2


def _handle_wikilink_asset(
    token: Token, md_path: Path, lines: Sequence[str]
) -> list[QueueItem]:
    """Process a token containing wikilink-style images: ![[path.ext]] or ![[path.ext|alt]]."""

    items: list[QueueItem] = []
    for src_attr, alt_text in _iter_wikilink_images(token.content):
        src_lower = src_attr.lower()
        has_asset_ext = any(
            src_lower.endswith(ext) for ext in WIKILINK_ASSET_EXTENSIONS
        )
        if not has_asset_ext or _is_alt_meaningful(alt_text):
            continue

        search_snippet = f"![[{src_attr}"
        line_no = _get_line_number(token, lines, search_snippet)
        items.append(_create_queue_item(md_path, src_attr, line_no, lines))

    return items


def _process_file(md_path: Path) -> list[QueueItem]:
    md = MarkdownIt("commonmark")
    source_text = md_path.read_text(encoding="utf-8")
    lines = source_text.splitlines()

    items: list[QueueItem] = []
    tokens = md.parse(source_text)
    # Track video asset paths already found via inline tokens to avoid
    # double-counting when html_inline children are also yielded.
    processed_video_assets: set[str] = set()

    for token in _iter_media_tokens(tokens):
        if token.type == "image":
            token_items = _handle_md_asset(token, md_path, lines)
        elif token.type in {"html_inline", "html_block"}:
            soup = BeautifulSoup(token.content, "html.parser")

            if soup.find("img"):
                token_items = _handle_html_asset(token, md_path, lines)
            elif soup.find("video"):
                token_items = _handle_html_video(token, md_path, lines)
                # Filter out videos already found via parent inline token
                token_items = [
                    it for it in token_items
                    if it.asset_path not in processed_video_assets
                ]
            else:
                token_items = []
        elif token.type == "inline":
            token_items = []
            soup = BeautifulSoup(token.content, "html.parser")
            if soup.find("video"):
                video_items = _handle_html_video(token, md_path, lines)
                for video_item in video_items:
                    processed_video_assets.add(video_item.asset_path)
                token_items.extend(video_items)
            # Always check for wikilinks in inline tokens
            token_items.extend(_handle_wikilink_asset(token, md_path, lines))
        else:
            token_items = _handle_wikilink_asset(token, md_path, lines)
        items.extend(token_items)
    return items


def build_queue(root: Path) -> list[QueueItem]:
    """Return a queue of assets lacking alt text beneath *root*."""

    md_files = utils.get_files(root, filetypes_to_match=(".md",), use_git_ignore=True)
    queue: list[QueueItem] = []
    for md_file in md_files:
        queue.extend(_process_file(md_file))

    return queue
