"""Apply labeled alt text to markdown files."""

import json
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.text import Text

from alt_text_llm import utils


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
    pattern = rf"!\[([^\]]*)\]\({escaped_path}\s*\)"

    match = re.search(pattern, line)
    if not match:
        return line, None

    old_alt = match.group(1) if match.group(1) else None
    # Replace the alt text - use lambda to avoid backslash interpretation in replacement
    new_line = re.sub(
        pattern, lambda m: f"![{new_alt}]({asset_path})", line, count=1
    )
    return new_line, old_alt


def _apply_html_image_alt(
    line: str, asset_path: str, new_alt: str
) -> tuple[str, str | None]:
    """
    Apply alt text to an HTML img tag.

    Args:
        line: The line containing the img tag
        asset_path: The asset path to match
        new_alt: The new alt text to apply

    Returns:
        Tuple of (modified line, old alt text or None)
    """
    # Escape special regex chars in asset_path
    escaped_path = re.escape(asset_path)

    # Match img tag with this src (handles both > and /> endings)
    # Capture group 1: attributes, Group 2: whitespace before closing, Group 3: closing slash
    img_pattern = rf'<img\s+([^>]*src="{escaped_path}"[^/>]*?)(\s*)(/?)>'

    match = re.search(img_pattern, line, re.IGNORECASE | re.DOTALL)
    if not match:
        return line, None

    img_attrs = match.group(1).rstrip()  # Remove trailing whitespace
    old_alt: str | None = None
    whitespace_before_close = match.group(2)  # Whitespace before closing
    closing_slash = match.group(3)  # Either "/" or ""

    # Check if alt attribute exists
    alt_pattern = r'alt="([^"]*)"'
    alt_match = re.search(alt_pattern, img_attrs, re.IGNORECASE)

    if alt_match:
        old_alt = alt_match.group(1)
        # Replace existing alt - use lambda to avoid backslash interpretation
        new_attrs = re.sub(
            alt_pattern,
            lambda m: f'alt="{new_alt}"',
            img_attrs,
            count=1,
            flags=re.IGNORECASE,
        )
    else:
        # Add alt attribute (insert before src or at the end)
        # Use lambda to avoid backslash interpretation in replacement
        new_attrs = re.sub(
            rf'(src="{escaped_path}")',
            lambda m: f'alt="{new_alt}" {m.group(1)}',
            img_attrs,
            count=1,
            flags=re.IGNORECASE,
        )

    # Reconstruct the img tag with proper closing, preserving original whitespace
    old_tag = f"<img {img_attrs}{whitespace_before_close}{closing_slash}>"
    new_tag = f"<img {new_attrs}{whitespace_before_close}{closing_slash}>"
    new_line = line.replace(old_tag, new_tag)
    return new_line, old_alt


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

    old_alt = match.group(1) if match.group(1) else None
    # Replace with new alt text - use lambda to avoid backslash interpretation
    new_line = re.sub(
        pattern, lambda m: f"![[{asset_path}|{new_alt}]]", line, count=1
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


def _apply_caption_to_file(
    md_path: Path,
    caption_item: utils.AltGenerationResult,
    console: Console,
    dry_run: bool = False,
) -> tuple[str | None, str] | None:
    """
    Apply a caption to a specific asset in a markdown file.

    Args:
        md_path: Path to the markdown file
        caption_item: The AltGenerationResult with final_alt to apply
        console: Rich console for output
        dry_run: If True, don't actually modify files

    Returns:
        Tuple of (old_alt, new_alt) if successful, None otherwise
    """
    assert caption_item.final_alt is not None, "final_alt must be set"

    source_text = md_path.read_text(encoding="utf-8")
    lines = source_text.splitlines()

    target_line = caption_item.line_number

    # Validate line number
    if target_line < 1 or target_line > len(lines):
        console.print(
            f"[yellow]Warning: Line {target_line} out of range for {md_path}[/yellow]"
        )
        return None

    # Get the target line (convert to 0-based index)
    line_idx = target_line - 1
    original_line = lines[line_idx]

    # Try markdown image first
    modified_line, old_alt = _apply_markdown_image_alt(
        original_line, caption_item.asset_path, caption_item.final_alt
    )

    # If no change, try wikilink image
    if modified_line == original_line:
        modified_line, old_alt = _apply_wikilink_image_alt(
            original_line, caption_item.asset_path, caption_item.final_alt
        )

    # If no change, try HTML image
    if modified_line == original_line:
        modified_line, old_alt = _apply_html_image_alt(
            original_line, caption_item.asset_path, caption_item.final_alt
        )

    # Check if anything changed
    if modified_line == original_line:
        console.print(
            f"[yellow]Warning: Could not find asset '{caption_item.asset_path}' on line {target_line} in {md_path}[/yellow]"
        )
        return None

    # Apply the change
    lines[line_idx] = modified_line

    if not dry_run:
        # Write back to file
        new_content = "\n".join(lines)
        # Preserve trailing newline if original had one
        if source_text.endswith("\n"):
            new_content += "\n"
        md_path.write_text(new_content, encoding="utf-8")

    return (old_alt, caption_item.final_alt)


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
    # Load captions as AltGenerationResult objects
    with open(captions_path, encoding="utf-8") as f:
        captions_data = json.load(f)

    # Convert to AltGenerationResult objects and filter for final_alt
    captions_to_apply: list[utils.AltGenerationResult] = []
    unused_entries: set[tuple[str, str]] = set()

    for item in captions_data:
        if item.get("final_alt") and item.get("final_alt").strip():
            captions_to_apply.append(
                utils.AltGenerationResult(
                    markdown_file=item["markdown_file"],
                    asset_path=item["asset_path"],
                    suggested_alt=item["suggested_alt"],
                    model=item["model"],
                    context_snippet=item["context_snippet"],
                    line_number=int(item["line_number"]),
                    final_alt=item["final_alt"],
                )
            )
        else:
            unused_entries.add(
                (
                    item["markdown_file"],
                    Path(item["asset_path"]).name,
                )
            )

    _display_unused_entries(unused_entries, console)

    if not captions_to_apply:
        console.print(
            f"[yellow]No captions with 'final_alt' found in {captions_path}[/yellow]"
        )
        return 0

    console.print(
        f"[blue]Found {len(captions_to_apply)} captions to apply{' (dry run)' if dry_run else ''}[/blue]"
    )

    # Group by file for better organization
    by_file: dict[str, list[utils.AltGenerationResult]] = defaultdict(list)
    for item in captions_to_apply:
        by_file[item.markdown_file].append(item)

    applied_count = 0

    # Process each file
    for md_file, items in by_file.items():
        md_path = Path(md_file)

        if not md_path.exists():
            console.print(
                f"[yellow]Warning: File not found: {md_path}[/yellow]"
            )
            continue

        console.print(
            f"\n[dim]Processing {md_path} ({len(items)} captions)[/dim]"
        )

        # Sort by line number (descending) to avoid line shifts when modifying
        items_sorted = sorted(items, key=lambda x: x.line_number, reverse=True)

        for item in items_sorted:
            result = _apply_caption_to_file(
                md_path=md_path,
                caption_item=item,
                console=console,
                dry_run=dry_run,
            )

            if result:
                old_alt, new_alt = result
                applied_count += 1
                status = "Would apply" if dry_run else "Applied"
                old_text = f'"{old_alt}"' if old_alt else "(no alt)"

                # Build message with Text to avoid markup parsing issues
                message = Text("  ")
                message.append(f"{status}:", style="green")
                message.append(
                    f' {old_text} → "{new_alt}" @ line {item.line_number}'
                )
                console.print(message)

    return applied_count


def apply_from_captions_file(
    captions_file: Path, dry_run: bool = False
) -> None:
    """
    Load captions from file and apply them to markdown files.

    Args:
        captions_file: Path to the captions JSON file
        dry_run: If True, show what would be done without modifying files
    """
    console = Console()

    if not captions_file.exists():
        console.print(
            f"[red]Error: Captions file not found: {captions_file}[/red]"
        )
        return

    applied_count = apply_captions(captions_file, console, dry_run=dry_run)

    # Summary
    if dry_run:
        console.print(
            f"\n[blue]Dry run complete: {applied_count} captions would be applied[/blue]"
        )
    else:
        console.print(
            f"\n[green]Successfully applied {applied_count} captions[/green]"
        )
