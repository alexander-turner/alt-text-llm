"""
Scan markdown files for assets without meaningful alt text.

This script produces a JSON work-queue.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

        if token.type == "inline" and "![[" in token.content:
            yield token

        if token.children:
            stack.extend(token.children)


_WIKILINK_IMG_RE = re.compile(r"!\[\[(?P<src>[^\]|]+)(?:\|(?P<alt>[^\]]*))?\]\]")


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
        src = img.get("src")
        alt = img.get("alt")
        if src:
            infos.append((src, alt))

    return infos


def _extract_html_video_info(token: Token) -> list[tuple[str, dict[str, str | None]]]:
    """Return list of (src, accessibility_attrs) for each <video>.
    
    Extracts src from video tag or first <source> child.
    Returns dict with aria-label, title, aria-describedby values.
    """
    soup = BeautifulSoup(token.content, "html.parser")
    infos: list[tuple[str, dict[str, str | None]]] = []

    for video in soup.find_all("video"):
        src = video.get("src")
        if not src:
            source = video.find("source")
            src = source.get("src") if source else None

        if src:
            accessibility_attrs = {
                "aria_label": video.get("aria-label"),
                "title": video.get("title"),
                "aria_describedby": video.get("aria-describedby"),
            }
            infos.append((src, accessibility_attrs))

    return infos


def _get_line_number(token: Token, lines: Sequence[str], search_snippet: str) -> int:
    if token.map:
        return token.map[0] + 1

    # Try exact match first
    for idx, ln in enumerate(lines):
        if search_snippet in ln:
            return idx + 1

    # If exact match fails, try with whitespace variations
    # Remove parentheses and search for just the asset path with flexible whitespace
    if search_snippet.startswith("(") and search_snippet.endswith(")"):
        asset_path = search_snippet[1:-1]  # Remove parentheses
        for idx, ln in enumerate(lines):
            if asset_path in ln:
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

    src_raw = token.attrGet("src")
    src_attr: str | None = str(src_raw) if src_raw is not None else None

    alt_text: str | None = token.content  # alt stored here
    if not src_attr or _is_alt_meaningful(alt_text):
        return []

    line_no = _get_line_number(token, lines, f"({src_attr})")
    return [_create_queue_item(md_path, src_attr, line_no, lines)]


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


def _handle_wikilink_asset(
    token: Token, md_path: Path, lines: Sequence[str]
) -> list[QueueItem]:
    """
    Process a token containing wikilink-style images: ![[path.ext]] or ![[path.ext|alt]].

    Args:
        token: Token potentially containing one or more wikilink images.
        md_path: Current markdown file path.
        lines: Contents of *md_path* split by lines.

    Returns:
        List of ``QueueItem`` instances—one for each wikilink image lacking alt text.
    """
    items: list[QueueItem] = []
    for match in _WIKILINK_IMG_RE.finditer(token.content):
        src_attr = match.group("src")
        alt_text = match.group("alt")

        # Check if it ends with a valid image extension
        src_lower = src_attr.lower()
        has_image_ext = any(
            src_lower.endswith(ext) for ext in WIKILINK_ASSET_EXTENSIONS
        )
        if not has_image_ext or _is_alt_meaningful(alt_text):
            continue

        # Search for the wikilink pattern in the file
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
    processed_inline_videos = set()
    
    for token in _iter_media_tokens(tokens):
        if token.type == "image":
            token_items = _handle_md_asset(token, md_path, lines)
        elif token.type in {"html_inline", "html_block"}:
            soup = BeautifulSoup(token.content, "html.parser")

            if soup.find("img"):
                token_items = _handle_html_asset(token, md_path, lines)
            elif soup.find("video"):
                token_items = _handle_html_video(token, md_path, lines)
            else:
                token_items = []
        elif token.type == "inline":
            soup = BeautifulSoup(token.content, "html.parser")
            if soup.find("video") and id(token) not in processed_inline_videos:
                processed_inline_videos.add(id(token))
                token_items = _handle_html_video(token, md_path, lines)
            else:
                token_items = _handle_wikilink_asset(token, md_path, lines)
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
