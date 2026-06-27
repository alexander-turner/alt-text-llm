"""Apply labeled alt text to markdown files."""

import json
import re
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, exceptions as bs4_exceptions
from rich.console import Console
from rich.text import Text

from alt_text_llm import utils


def _escape_markdown_alt_text(alt_text: str) -> str:
    """
    Escape special characters in alt text for markdown.

    Args:
        alt_text: The alt text to escape

    Returns:
        Escaped alt text safe for markdown
    """
    # Escape backslashes first to avoid double-escaping
    alt_text = alt_text.replace("\\", "\\\\")
    # Escape dollar signs to prevent LaTeX interpretation
    alt_text = alt_text.replace("$", "\\$")
    # Escape square brackets so they don't break the markdown image syntax
    alt_text = alt_text.replace("[", "\\[")
    alt_text = alt_text.replace("]", "\\]")
    return alt_text


def _apply_markdown_image_alt(
    line: str, asset_path: str, new_alt: str
) -> tuple[str, str | None]:
    """
    Apply alt text to a markdown image syntax.

    Args:
        line: The line containing the image
        asset_path: The asset path to match
        new_alt: The new alt text to apply

    Returns:
        Tuple of (modified line, old alt text or None)
    """
    # Match markdown image syntax: ![alt](path)
    # Need to escape special regex chars in asset_path
    escaped_path = re.escape(asset_path)
    # Optionally allow a markdown title after the path: ![](img.png "Title").
    # Group 2 captures the title (including its leading whitespace) so it can be
    # preserved on rewrite.
    pattern = rf"!\[([^\]]*)\]\({escaped_path}(\s+\"[^\"]*\")?\s*\)"

    match = re.search(pattern, line)
    if not match:
        return line, None

    old_alt = match.group(1) or None
    # Escape special characters in alt text
    escaped_alt = _escape_markdown_alt_text(new_alt)

    # Replace the alt text for every matching image on the line (count=0),
    # preserving any title present in each individual match.
    def _replace(m: re.Match[str]) -> str:
        title = m.group(2) or ""
        return f"![{escaped_alt}]({asset_path}{title})"

    new_line = re.sub(pattern, _replace, line, count=0)
    return new_line, old_alt


def _extract_media_src(tag_name: str, element: object) -> str | None:
    """Best-effort extraction of the asset URL/path from an HTML media tag."""
    # BeautifulSoup Tag is duck-typed: has .get and .find
    src = element.get("src")  # type: ignore[attr-defined]
    if src:
        return src

    if tag_name == "video":
        source = element.find("source")  # type: ignore[attr-defined]
        return source.get("src") if source else None

    return None


def _escape_html_attr(value: str) -> str:
    """Escape a string for use inside a double-quoted HTML attribute value."""
    value = value.replace("&", "&amp;")
    value = value.replace("<", "&lt;")
    value = value.replace(">", "&gt;")
    value = value.replace('"', "&quot;")
    return value


def _sourcepos_to_index(text: str, sourceline: int | None, sourcepos: int | None) -> int | None:
    """Convert a (1-based line, 0-based column) BeautifulSoup position to an index."""
    if sourceline is None or sourcepos is None:
        return None
    lines = text.split("\n")
    if not 1 <= sourceline <= len(lines):
        return None
    # Account for the "\n" separators removed by split().
    offset = sum(len(lines[i]) + 1 for i in range(sourceline - 1))
    return offset + sourcepos


def _find_opening_tag_end(text: str, start: int) -> int | None:
    """Return the index of the '>' that closes the opening tag starting at ``start``.

    Skips over '>' characters that appear inside quoted attribute values.
    """
    quote: str | None = None
    for i in range(start, len(text)):
        char = text[i]
        if quote is not None:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == ">":
            return i
    return None


def _set_attribute_in_opening_tag(
    opening_tag: str, write_attr: str, new_value: str
) -> str:
    """Replace ``write_attr`` in ``opening_tag`` if present, otherwise insert it.

    Only the opening tag's text is touched; the value is HTML-escaped and always
    emitted with double quotes.
    """
    escaped = _escape_html_attr(new_value)
    replacement = f'{write_attr}="{escaped}"'

    # Match an existing attribute with optional value (quoted, unquoted, or bare).
    # The trailing lookahead pins the end of the attribute *name* so we don't
    # match a prefix of a longer attribute (e.g. ``aria-label`` inside
    # ``aria-labelledby``), which would corrupt the markup on rewrite.
    attr_pattern = re.compile(
        rf"""(?<=[\s<])({re.escape(write_attr)})(?=[\s=/>])(\s*=\s*("[^"]*"|'[^']*'|[^\s>/]*))?""",
        re.IGNORECASE,
    )
    new_tag, count = attr_pattern.subn(lambda _m: replacement, opening_tag, count=1)
    if count:
        return new_tag

    # Attribute absent: insert before the closing '>' (or '/>').
    insertion_point = len(opening_tag) - (2 if opening_tag.endswith("/>") else 1)
    prefix = opening_tag[:insertion_point].rstrip()
    suffix = opening_tag[insertion_point:]
    return f"{prefix} {replacement}{suffix}"


def _apply_html_tag_attribute(
    *,
    text: str,
    tag_name: str,
    asset_path: str,
    new_value: str,
    read_old_from: tuple[str, ...],
    write_attr: str,
) -> tuple[str, str | None]:
    """Apply an attribute update to every matching HTML tag in ``text``.

    ``text`` may be a single line or a whole file; the rewrite is a surgical,
    in-place splice of each matched opening tag, so markup outside those opening
    tags is preserved verbatim. This is what lets us handle elements that span
    multiple lines (e.g. a ``<video>`` whose ``<source>`` children sit on later
    lines, or an opening tag whose attributes wrap across lines) — cases the
    parser sees correctly but a line-at-a-time rewrite cannot.

    Notes:
        - BeautifulSoup is used only to *locate* matching tags and read their old
          values. We never reserialize via ``str(soup)``: on a whole file that
          would rewrite the entire document as parsed HTML, and even on a
          fragment it "repairs" unclosed elements and corrupts the markup.
        - Matching is exact equality against the resolved src (for videos,
          prefers @src and otherwise the first <source src=...> child).
        - Multiple matches are spliced from last to first so earlier source
          offsets stay valid as the text is mutated.
        - BeautifulSoup can raise ParserRejectedMarkup on text containing
          non-HTML markup that confuses the parser (e.g. regex-like fragments in
          JS/TS code blocks). In that case we leave the text unchanged.
    """
    try:
        soup = BeautifulSoup(text, "html.parser")
    except bs4_exceptions.ParserRejectedMarkup:
        print(f"{text=} created a bs4 parsing error! Ignoring.")
        return text, None

    # (start, end_exclusive, replacement) splices plus the first match's old value.
    edits: list[tuple[int, int, str]] = []
    first_old_value: str | None = None
    for el in soup.find_all(tag_name):
        if _extract_media_src(tag_name, el) != asset_path:
            continue

        start = _sourcepos_to_index(text, el.sourceline, el.sourcepos)  # type: ignore[attr-defined]
        end = _find_opening_tag_end(text, start) if start is not None else None
        if start is None or end is None:
            # Cannot locate the tag precisely; skip rather than risk corrupting
            # the document. (Modern bs4 always reports source positions.)
            continue

        if not edits:
            old_value_raw = next((el.get(a) for a in read_old_from if el.get(a)), None)
            first_old_value = str(old_value_raw) if old_value_raw is not None else None

        new_opening = _set_attribute_in_opening_tag(
            text[start : end + 1], write_attr, new_value
        )
        edits.append((start, end + 1, new_opening))

    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]

    return text, first_old_value


def _apply_html_image_alt(
    text: str, asset_path: str, new_alt: str
) -> tuple[str, str | None]:
    """Apply alt text to HTML img tags in ``text``."""
    return _apply_html_tag_attribute(
        text=text,
        tag_name="img",
        asset_path=asset_path,
        new_value=new_alt,
        read_old_from=("alt",),
        write_attr="alt",
    )


def _apply_html_video_label(
    text: str, asset_path: str, new_label: str
) -> tuple[str, str | None]:
    """Apply accessibility label to HTML video tags in ``text``."""
    return _apply_html_tag_attribute(
        text=text,
        tag_name="video",
        asset_path=asset_path,
        new_value=new_label,
        read_old_from=("aria-label", "title", "aria-describedby"),
        write_attr="aria-label",
    )


def _apply_wikilink_image_alt(
    line: str, asset_path: str, new_alt: str
) -> tuple[str, str | None]:
    """
    Apply alt text to a wikilink-style image syntax (e.g. Obsidian).

    Args:
        line: The line containing the image
        asset_path: The asset path to match
        new_alt: The new alt text to apply

    Returns:
        (modified line, old alt text or None)
    """
    # Match wikilink image syntax: ![[path]] or ![[path|alt]]
    # Need to escape special regex chars in asset_path
    escaped_path = re.escape(asset_path)
    pattern = rf"!\[\[{escaped_path}(?:\|([^\]]*))?\]\]"

    match = re.search(pattern, line)
    if not match:
        return line, None

    old_alt = match.group(1) or None
    # Escape special characters in alt text (wikilinks are still markdown)
    escaped_alt = _escape_markdown_alt_text(new_alt)
    # Replace with new alt text - use lambda to avoid backslash interpretation
    new_line = re.sub(
        pattern, lambda m: f"![[{asset_path}|{escaped_alt}]]", line, count=0
    )
    return new_line, old_alt


def _display_unused_entries(
    unused_entries: set[tuple[str, str]], console: Console
) -> None:
    if not unused_entries:
        return

    console.print(
        f"[yellow]Note: {len(unused_entries)} {'entry' if len(unused_entries) == 1 else 'entries'} without 'final_alt' will be skipped:[/yellow]"
    )
    for markdown_file, asset_basename in sorted(unused_entries):
        console.print(f"[dim]  {markdown_file}: {asset_basename}[/dim]")


def _normalize_alt(new_alt: str) -> str:
    """Collapse runs of line breaks in alt text into a single ellipsis."""
    return re.sub(r"(\r\n|\r|\n)+", " ... ", new_alt)


# Line-oriented markdown syntaxes, applied one line at a time.
_LINE_FORMAT_APPLIERS = (
    _apply_markdown_image_alt,
    _apply_wikilink_image_alt,
)
# Whole-text HTML appliers, run over the entire file so that elements spanning
# multiple lines are matched correctly.
_HTML_FORMAT_APPLIERS = (
    _apply_html_image_alt,
    _apply_html_video_label,
)
_FORMAT_APPLIERS = _LINE_FORMAT_APPLIERS + _HTML_FORMAT_APPLIERS


def _try_all_image_formats(
    line: str, asset_path: str, new_alt: str
) -> tuple[str, str | None]:
    """
    Try each supported format on a single line until one matches.

    Returns:
        Tuple of (modified line, old alt text or None)
    """
    normalized_alt = _normalize_alt(new_alt)

    for apply_format in _FORMAT_APPLIERS:
        modified_line, old_alt = apply_format(line, asset_path, normalized_alt)
        if modified_line != line:
            return modified_line, old_alt

    return line, None


def _apply_caption_to_lines(
    lines: list[str],
    caption_item: utils.AltGenerationResult,
    md_path: Path,
    console: Console,
) -> tuple[str | None, str] | None:
    """
    Apply a caption to all instances of an asset across in-memory lines.

    Mutates ``lines`` in place. Does not perform any file I/O so that callers
    can batch multiple captions into a single read/write cycle.

    Markdown and wikilink images are single-line syntaxes and are handled one
    line at a time. HTML ``<img>``/``<video>`` elements are handled by parsing
    the whole file, so tags whose attributes or ``<source>`` children span
    multiple lines are still matched. HTML attribute splices never add or remove
    newlines, so the line count is preserved.

    Args:
        lines: The file's lines (mutated in place when a match is found)
        caption_item: The AltGenerationResult with final_alt to apply
        md_path: Path to the markdown file (used only for the warning message)
        console: Rich console for output

    Returns:
        Tuple of (old_alt, new_alt) if successful, None otherwise
    """
    assert caption_item.final_alt is not None, "final_alt must be set"

    asset_path = caption_item.asset_path
    normalized_alt = _normalize_alt(caption_item.final_alt)

    changed = False
    old_alt: str | None = None

    # 1) Per-line markdown + wikilink images.
    for line_idx, line in enumerate(lines):
        for apply_format in _LINE_FORMAT_APPLIERS:
            modified_line, line_old_alt = apply_format(line, asset_path, normalized_alt)
            if modified_line != line:
                lines[line_idx] = modified_line
                line = modified_line
                changed = True
                if old_alt is None:
                    old_alt = line_old_alt

    # 2) Whole-text HTML <img>/<video>.
    original_text = "\n".join(lines)
    text = original_text
    for html_format in _HTML_FORMAT_APPLIERS:
        modified_text, html_old_alt = html_format(text, asset_path, normalized_alt)
        if modified_text != text:
            text = modified_text
            changed = True
            if old_alt is None:
                old_alt = html_old_alt
    if text != original_text:
        # Splices never change the number of lines (see docstring).
        lines[:] = text.split("\n")

    if not changed:
        console.print(
            f"[yellow]Warning: Could not find asset '{asset_path}' in {md_path}[/yellow]"
        )
        return None

    return (old_alt, caption_item.final_alt)


def _apply_caption_to_file(
    md_path: Path,
    caption_item: utils.AltGenerationResult,
    console: Console,
    dry_run: bool = False,
) -> tuple[str | None, str] | None:
    """
    Apply a caption to all instances of an asset in a markdown file.

    Args:
        md_path: Path to the markdown file
        caption_item: The AltGenerationResult with final_alt to apply
        console: Rich console for output
        dry_run: If True, don't actually modify files

    Returns:
        Tuple of (old_alt, new_alt) if successful, None otherwise
    """
    source_text = md_path.read_text(encoding="utf-8")
    lines = source_text.splitlines()

    result = _apply_caption_to_lines(lines, caption_item, md_path, console)
    if result is None:
        return None

    if not dry_run:
        new_content = "\n".join(lines)
        # Preserve trailing newline if original had one
        if source_text.endswith("\n"):
            new_content += "\n"
        md_path.write_text(new_content, encoding="utf-8")

    return result


def _load_and_parse_captions(
    captions_path: Path,
) -> tuple[list[utils.AltGenerationResult], set[tuple[str, str]]]:
    """
    Load captions from JSON and parse into AltGenerationResult objects.

    Args:
        captions_path: Path to the captions JSON file

    Returns:
        Tuple of (captions to apply, unused entries)
    """
    with open(captions_path, encoding="utf-8") as f:
        captions_data = json.load(f)

    captions_to_apply: list[utils.AltGenerationResult] = []
    unused_entries: set[tuple[str, str]] = set()

    for item in captions_data:
        if (item.get("final_alt") or "").strip():
            captions_to_apply.append(utils.AltGenerationResult.from_json(item))
        else:
            unused_entries.add(
                (
                    item["markdown_file"],
                    Path(item["asset_path"]).name,
                )
            )

    return captions_to_apply, unused_entries


def _group_captions_by_file(
    captions: list[utils.AltGenerationResult],
) -> dict[str, list[utils.AltGenerationResult]]:
    """
    Group captions by their markdown file.

    Args:
        captions: List of captions to group

    Returns:
        Dictionary mapping file paths to lists of captions
    """
    by_file: dict[str, list[utils.AltGenerationResult]] = defaultdict(list)
    for item in captions:
        by_file[item.markdown_file].append(item)
    return by_file


def _display_caption_result(
    result: tuple[str | None, str],
    item: utils.AltGenerationResult,
    console: Console,
    dry_run: bool,
) -> None:
    """
    Display the result of applying a caption.

    Args:
        result: Tuple of (old_alt, new_alt)
        item: The caption item that was applied
        console: Rich console for output
        dry_run: Whether this is a dry run
    """
    old_alt, new_alt = result
    status = "Would apply" if dry_run else "Applied"
    old_text = f'"{old_alt}"' if old_alt else "(no alt)"

    # Build message with Text to avoid markup parsing issues
    message = Text("  ")
    message.append(f"{status}:", style="green")
    message.append(f' {old_text} → "{new_alt}"')
    console.print(message)


def _process_file_captions(
    md_path: Path,
    items: list[utils.AltGenerationResult],
    console: Console,
    dry_run: bool,
) -> int:
    """
    Process all captions for a single file.

    Args:
        md_path: Path to the markdown file
        items: List of captions to apply to this file
        console: Rich console for output
        dry_run: If True, don't actually modify files

    Returns:
        Number of successfully applied captions
    """
    if not md_path.exists():
        console.print(f"[yellow]Warning: File not found: {md_path}[/yellow]")
        return 0

    console.print(f"\n[dim]Processing {md_path} ({len(items)} captions)[/dim]")

    # Read the file once, apply every caption in memory, then write once.
    source_text = md_path.read_text(encoding="utf-8")
    lines = source_text.splitlines()

    applied_count = 0
    for item in items:
        result = _apply_caption_to_lines(
            lines=lines,
            caption_item=item,
            md_path=md_path,
            console=console,
        )

        if result:
            applied_count += 1
            _display_caption_result(result, item, console, dry_run)

    if applied_count > 0 and not dry_run:
        new_content = "\n".join(lines)
        # Preserve trailing newline if original had one
        if source_text.endswith("\n"):
            new_content += "\n"
        md_path.write_text(new_content, encoding="utf-8")

    return applied_count


def apply_captions(
    captions_path: Path,
    console: Console,
    dry_run: bool = False,
) -> int:
    """
    Apply captions from a JSON file to markdown files.

    Args:
        captions_path: Path to the captions JSON file
        console: Rich console for output
        dry_run: If True, show what would be done without modifying files

    Returns:
        Number of successfully applied captions
    """
    captions_to_apply, unused_entries = _load_and_parse_captions(captions_path)

    _display_unused_entries(unused_entries, console)

    if not captions_to_apply:
        console.print(
            f"[yellow]No captions with 'final_alt' found in {captions_path}[/yellow]"
        )
        return 0

    console.print(
        f"[blue]Found {len(captions_to_apply)} captions to apply{' (dry run)' if dry_run else ''}[/blue]"
    )

    by_file = _group_captions_by_file(captions_to_apply)

    applied_count = 0
    for md_file, items in by_file.items():
        md_path = Path(md_file)
        applied_count += _process_file_captions(md_path, items, console, dry_run)

    return applied_count


def apply_from_captions_file(captions_file: Path, dry_run: bool = False) -> None:
    """
    Load captions from file and apply them to markdown files.

    Args:
        captions_file: Path to the captions JSON file
        dry_run: If True, show what would be done without modifying files
    """
    console = Console()

    if not captions_file.exists():
        console.print(f"[red]Error: Captions file not found: {captions_file}[/red]")
        return

    applied_count = apply_captions(captions_file, console, dry_run=dry_run)

    # Summary
    if dry_run:
        console.print(
            f"\n[blue]Dry run complete: {applied_count} captions would be applied[/blue]"
        )
    else:
        console.print(f"\n[green]Successfully applied {applied_count} captions[/green]")
