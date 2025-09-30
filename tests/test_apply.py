"""Tests for apply module."""

import json
from pathlib import Path

import pytest
from rich.console import Console
from io import StringIO

from alt_text_llm import apply, utils


@pytest.fixture
def console():
    """Create a Rich console for tests."""
    return Console()


@pytest.fixture
def console_with_output():
    """Create a Rich console that captures output to StringIO."""
    output = StringIO()
    return Console(file=output, width=120), output


@pytest.fixture
def markdown_file_with_image(temp_dir: Path) -> Path:
    """Create a test markdown file with an image."""
    md_path = temp_dir / "test.md"
    content = """# Test File

This is a test ![old alt](image.png) image.

Another paragraph.
"""
    md_path.write_text(content)
    return md_path


@pytest.fixture
def html_file_with_image(temp_dir: Path) -> Path:
    """Create a test markdown file with an HTML img tag."""
    md_path = temp_dir / "test.md"
    content = """# Test File

<img alt="old alt" src="image.png">

Another paragraph.
"""
    md_path.write_text(content)
    return md_path


@pytest.fixture
def caption_item(markdown_file_with_image: Path) -> utils.AltGenerationResult:
    """Create a test AltGenerationResult."""
    return utils.AltGenerationResult(
        markdown_file=str(markdown_file_with_image),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=3,
        final_alt="new caption",
    )


def test_apply_markdown_image_alt() -> None:
    """Test applying alt text to markdown image syntax."""
    line = "This is ![old alt](path/to/image.png) in text"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt == "old alt"
    assert new_line == "This is ![new alt text](path/to/image.png) in text"


def test_apply_markdown_image_alt_empty() -> None:
    """Test applying alt text when original is empty."""
    line = "This is ![](path/to/image.png) in text"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt is None
    assert new_line == "This is ![new alt text](path/to/image.png) in text"


def test_apply_markdown_image_alt_whitespace_before_paren() -> None:
    """Test applying alt text when there's whitespace before closing paren."""
    line = "This is ![old alt](path/to/image.png ) in text"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt == "old alt"
    assert new_line == "This is ![new alt text](path/to/image.png) in text"


def test_apply_html_image_alt_existing() -> None:
    """Test applying alt text to HTML img tag with existing alt."""
    line = '<img alt="old alt" src="path/to/image.png">'
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt == "old alt"
    assert new_line == '<img alt="new alt text" src="path/to/image.png">'


def test_apply_html_image_alt_no_alt() -> None:
    """Test applying alt text to HTML img tag without alt."""
    line = '<img src="path/to/image.png">'
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt is None
    assert new_line == '<img alt="new alt text" src="path/to/image.png">'


def test_apply_html_image_alt_self_closing() -> None:
    """Test applying alt text to self-closing HTML img tag."""
    line = '<img alt="old alt" src="path/to/image.png" class="theme-emoji"/>'
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt == "old alt"
    assert (
        new_line
        == '<img alt="new alt text" src="path/to/image.png" class="theme-emoji"/>'
    )


def test_apply_html_image_alt_self_closing_no_alt() -> None:
    """Test adding alt text to self-closing HTML img tag without alt."""
    line = '<img src="path/to/image.png" class="icon"/>'
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt is None
    assert (
        new_line
        == '<img alt="new alt text" src="path/to/image.png" class="icon"/>'
    )


def test_apply_caption_to_file_markdown(
    markdown_file_with_image: Path,
    caption_item: utils.AltGenerationResult,
    console: Console,
) -> None:
    """Test applying caption to a markdown file."""
    result = apply._apply_caption_to_file(
        md_path=markdown_file_with_image,
        caption_item=caption_item,
        console=console,
        dry_run=False,
    )

    assert result is not None
    old_alt, new_alt = result
    assert old_alt == "old alt"
    assert new_alt == "new caption"

    # Verify file was updated
    new_content = markdown_file_with_image.read_text()
    assert "![new caption](image.png)" in new_content
    assert "![old alt](image.png)" not in new_content


def test_apply_caption_to_file_html(
    html_file_with_image: Path, console: Console
) -> None:
    """Test applying caption to HTML img tag in markdown file."""
    caption_item = utils.AltGenerationResult(
        markdown_file=str(html_file_with_image),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=3,
        final_alt="new caption",
    )

    result = apply._apply_caption_to_file(
        md_path=html_file_with_image,
        caption_item=caption_item,
        console=console,
        dry_run=False,
    )

    assert result is not None
    old_alt, new_alt = result
    assert old_alt == "old alt"
    assert new_alt == "new caption"

    # Verify file was updated
    new_content = html_file_with_image.read_text()
    assert 'alt="new caption"' in new_content
    assert 'alt="old alt"' not in new_content


def test_apply_captions_dry_run(
    temp_dir: Path, markdown_file_with_image: Path, console: Console
) -> None:
    """Test dry run mode doesn't modify files."""
    original_content = markdown_file_with_image.read_text()

    # Create captions file
    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": str(markdown_file_with_image),
            "asset_path": "image.png",
            "line_number": 3,
            "suggested_alt": "suggested",
            "final_alt": "new caption",
            "model": "test-model",
            "context_snippet": "context",
        }
    ]
    captions_path.write_text(json.dumps(captions_data))

    applied_count = apply.apply_captions(captions_path, console, dry_run=True)

    assert applied_count == 1
    # Verify file was NOT modified
    assert markdown_file_with_image.read_text() == original_content


def test_apply_captions_multiple_images(
    temp_dir: Path, console: Console
) -> None:
    """Test applying captions to multiple images in same file."""
    # Create a test markdown file
    md_path = temp_dir / "test.md"
    content = """# Test File

First image: ![alt1](image1.png)

Second image: ![alt2](image2.png)
"""
    md_path.write_text(content)

    # Create captions file
    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": str(md_path),
            "asset_path": "image1.png",
            "line_number": 3,
            "suggested_alt": "suggested1",
            "final_alt": "new caption 1",
            "model": "test-model",
            "context_snippet": "context",
        },
        {
            "markdown_file": str(md_path),
            "asset_path": "image2.png",
            "line_number": 5,
            "suggested_alt": "suggested2",
            "final_alt": "new caption 2",
            "model": "test-model",
            "context_snippet": "context",
        },
    ]
    captions_path.write_text(json.dumps(captions_data))

    applied_count = apply.apply_captions(captions_path, console, dry_run=False)

    assert applied_count == 2

    # Verify both were updated
    new_content = md_path.read_text()
    assert "![new caption 1](image1.png)" in new_content
    assert "![new caption 2](image2.png)" in new_content


def test_apply_wikilink_image_alt_with_existing_alt() -> None:
    """Test applying alt text to wikilink image syntax with existing alt."""
    line = "This is ![[path/to/image.png|old alt]] in text"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt == "old alt"
    assert new_line == "This is ![[path/to/image.png|new alt text]] in text"


def test_apply_wikilink_image_alt_no_alt() -> None:
    """Test applying alt text to wikilink image syntax without alt."""
    line = "This is ![[path/to/image.png]] in text"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt is None
    assert new_line == "This is ![[path/to/image.png|new alt text]] in text"


def test_apply_wikilink_image_alt_url() -> None:
    """Test applying alt text to wikilink with full URL."""
    line = "![[https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif]]"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line,
        "https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif",
        "new alt text",
    )

    assert old_alt is None
    assert (
        new_line
        == "![[https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif|new alt text]]"
    )


def test_apply_wikilink_image_alt_no_match() -> None:
    """Test wikilink function returns unchanged line when no match."""
    line = "This is ![markdown](image.png) not wikilink"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, "image.png", "new alt text"
    )

    assert old_alt is None
    assert new_line == line


@pytest.fixture
def wikilink_file_with_image(temp_dir: Path) -> Path:
    """Create a test markdown file with a wikilink image."""
    md_path = temp_dir / "test.md"
    content = """# Test File

This is a test ![[image.png|old alt]] image.

Another paragraph.
"""
    md_path.write_text(content)
    return md_path


def test_apply_caption_to_file_wikilink(
    wikilink_file_with_image: Path, console: Console
) -> None:
    """Test applying caption to wikilink image in markdown file."""
    caption_item = utils.AltGenerationResult(
        markdown_file=str(wikilink_file_with_image),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=3,
        final_alt="new caption",
    )

    result = apply._apply_caption_to_file(
        md_path=wikilink_file_with_image,
        caption_item=caption_item,
        console=console,
        dry_run=False,
    )

    assert result is not None
    old_alt, new_alt = result
    assert old_alt == "old alt"
    assert new_alt == "new caption"

    # Verify file was updated
    new_content = wikilink_file_with_image.read_text()
    assert "![[image.png|new caption]]" in new_content
    assert "![[image.png|old alt]]" not in new_content


def test_apply_wikilink_image_alt_special_chars() -> None:
    """Test wikilink with special regex characters in path."""
    line = "Image: ![[path/to/image (1).png]] here"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, "path/to/image (1).png", "new alt"
    )

    assert old_alt is None
    assert new_line == "Image: ![[path/to/image (1).png|new alt]] here"


def test_markdown_image_alt_with_backslash() -> None:
    """Test applying alt text containing backslash (regex special char)."""
    line = "This is ![old alt](image.png) in text"
    # Alt text with backslash that would cause re.error if not handled properly
    new_alt = r"A diagram\showing states A, B, and C"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "image.png", new_alt
    )

    assert old_alt == "old alt"
    assert new_line == f"This is ![{new_alt}](image.png) in text"


def test_markdown_image_alt_with_dollar_sign() -> None:
    """Test applying alt text containing dollar sign (backreference char)."""
    line = "This is ![](image.png) in text"
    new_alt = "Price is $100 and $200"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "image.png", new_alt
    )

    assert old_alt is None
    assert new_line == f"This is ![{new_alt}](image.png) in text"


def test_html_image_alt_with_backslash() -> None:
    """Test applying HTML alt text containing backslash."""
    line = '<img alt="old" src="image.png">'
    new_alt = r"Diagram\showing process"
    new_line, old_alt = apply._apply_html_image_alt(line, "image.png", new_alt)

    assert old_alt == "old"
    assert new_line == f'<img alt="{new_alt}" src="image.png">'


def test_html_image_alt_no_alt_with_backslash() -> None:
    """Test adding HTML alt text containing backslash when no alt exists."""
    line = '<img src="image.png">'
    new_alt = r"A state transition diagram\showing paths"
    new_line, old_alt = apply._apply_html_image_alt(line, "image.png", new_alt)

    assert old_alt is None
    assert new_line == f'<img alt="{new_alt}" src="image.png">'


def test_wikilink_image_alt_with_backslash() -> None:
    """Test applying wikilink alt text containing backslash."""
    line = "Image: ![[image.png|old alt]] here"
    new_alt = r"Diagram\with backslash"
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, "image.png", new_alt
    )

    assert old_alt == "old alt"
    assert new_line == f"Image: ![[image.png|{new_alt}]] here"


def test_markdown_image_alt_with_multiple_special_chars() -> None:
    """Test alt text with multiple regex special characters."""
    line = "![](test.png)"
    # Complex alt text that would fail without proper escaping
    new_alt = r"A diagram (version 1.0) showing $variable\in set {A, B, C}"
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "test.png", new_alt
    )

    assert old_alt is None
    assert new_line == f"![{new_alt}](test.png)"


def test_display_unused_entries_empty(console_with_output: tuple) -> None:
    """Test displaying unused entries with empty set."""
    console, output = console_with_output
    unused_entries: set[tuple[str, str]] = set()

    apply._display_unused_entries(unused_entries, console)

    # Should produce no output
    assert output.getvalue() == ""


def test_display_unused_entries_single(console_with_output: tuple) -> None:
    """Test displaying a single unused entry."""
    console, output = console_with_output
    unused_entries = {("path/to/file.md", "image.png")}

    apply._display_unused_entries(unused_entries, console)

    result = output.getvalue()
    assert "1 entry without 'final_alt' will be skipped:" in result
    assert "path/to/file.md: image.png" in result


def test_display_unused_entries_multiple(console_with_output: tuple) -> None:
    """Test displaying multiple unused entries."""
    console, output = console_with_output
    unused_entries = {
        ("path/to/file1.md", "image1.png"),
        ("path/to/file2.md", "image2.png"),
        ("path/to/file1.md", "image3.png"),
    }

    apply._display_unused_entries(unused_entries, console)

    result = output.getvalue()
    assert "3 entries without 'final_alt' will be skipped:" in result
    assert "path/to/file1.md: image1.png" in result
    assert "path/to/file1.md: image3.png" in result
    assert "path/to/file2.md: image2.png" in result


def test_display_unused_entries_sorted() -> None:
    """Test that unused entries are displayed in sorted order."""
    from io import StringIO

    output = StringIO()
    console = Console(file=output, width=120)
    unused_entries = {
        ("z_file.md", "z_image.png"),
        ("a_file.md", "a_image.png"),
        ("m_file.md", "m_image.png"),
    }

    apply._display_unused_entries(unused_entries, console)

    result = output.getvalue()
    lines = [
        line.strip()
        for line in result.splitlines()
        if ":" in line and ".md" in line
    ]
    assert len(lines) == 3
    assert "a_file.md: a_image.png" in lines[0]
    assert "m_file.md: m_image.png" in lines[1]
    assert "z_file.md: z_image.png" in lines[2]


def test_apply_captions_with_unused_entries(
    temp_dir: Path, markdown_file_with_image: Path, console_with_output: tuple
) -> None:
    """Test that apply_captions correctly reports unused entries."""
    console, output = console_with_output

    # Create captions file with both used and unused entries
    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": str(markdown_file_with_image),
            "asset_path": "image.png",
            "line_number": 3,
            "suggested_alt": "suggested",
            "final_alt": "new caption",
            "model": "test-model",
            "context_snippet": "context",
        },
        {
            "markdown_file": str(markdown_file_with_image),
            "asset_path": "unused_image.png",
            "line_number": 5,
            "suggested_alt": "suggested for unused",
            "final_alt": "",  # Empty final_alt
            "model": "test-model",
            "context_snippet": "context",
        },
        {
            "markdown_file": "another_file.md",
            "asset_path": "another_image.png",
            "line_number": 1,
            "suggested_alt": "suggested for another",
            "final_alt": None,  # No final_alt
            "model": "test-model",
            "context_snippet": "context",
        },
    ]
    captions_path.write_text(json.dumps(captions_data))

    applied_count = apply.apply_captions(captions_path, console, dry_run=False)

    assert applied_count == 1

    result = output.getvalue()
    assert "2 entries without 'final_alt' will be skipped:" in result
    assert "unused_image.png" in result
    assert "another_image.png" in result


def test_apply_captions_deduplicates_unused_entries(
    temp_dir: Path, markdown_file_with_image: Path, console_with_output: tuple
) -> None:
    """Test that duplicate unused entries are deduplicated."""
    console, output = console_with_output

    # Create captions file with duplicate unused entries
    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": "file.md",
            "asset_path": "path/to/image.png",
            "line_number": 1,
            "suggested_alt": "suggested",
            "final_alt": "",
            "model": "test-model",
            "context_snippet": "context",
        },
        {
            "markdown_file": "file.md",
            "asset_path": "path/to/image.png",
            "line_number": 2,
            "suggested_alt": "suggested again",
            "final_alt": None,
            "model": "test-model",
            "context_snippet": "context",
        },
    ]
    captions_path.write_text(json.dumps(captions_data))

    applied_count = apply.apply_captions(captions_path, console, dry_run=False)

    assert applied_count == 0

    result = output.getvalue()
    # Should only show 1 entry, not 2 (deduplicated)
    assert "1 entry without 'final_alt' will be skipped:" in result
    assert "file.md: image.png" in result
