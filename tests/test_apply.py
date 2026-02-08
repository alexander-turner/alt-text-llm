"""Tests for apply module."""

import json
import textwrap
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from alt_text_llm import apply, scan, utils


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Cost is $100 and $200", r"Cost is \$100 and \$200"),
        (r"Path is C:\Users\test", r"Path is C:\\Users\\test"),
        (r"Formula: x\in set, cost $50", r"Formula: x\\in set, cost \$50"),
        (
            "A simple description with no special chars",
            "A simple description with no special chars",
        ),
    ],
)
def test_escape_markdown_alt_text(input_text: str, expected: str) -> None:
    """Test escaping special characters in markdown alt text."""
    assert apply._escape_markdown_alt_text(input_text) == expected


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("Tom & Jerry", "Tom &amp; Jerry"),
        ("Formula: x < y and y > z", "Formula: x &lt; y and y &gt; z"),
        ('She said "hello"', "She said &quot;hello&quot;"),
        (
            '<tag attr="value"> & more',
            "&lt;tag attr=&quot;value&quot;&gt; &amp; more",
        ),
        ("A simple description", "A simple description"),
    ],
)
def test_escape_html_alt_text(input_text: str, expected: str) -> None:
    """Test escaping special characters in HTML alt text."""
    assert apply._escape_html_alt_text(input_text) == expected


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


@pytest.mark.parametrize(
    "line,expected_old_alt,expected_new_line",
    [
        pytest.param(
            "This is ![old alt](path/to/image.png) in text",
            "old alt",
            "This is ![new alt text](path/to/image.png) in text",
            id="existing_alt",
        ),
        pytest.param(
            "This is ![](path/to/image.png) in text",
            None,
            "This is ![new alt text](path/to/image.png) in text",
            id="empty_alt",
        ),
        pytest.param(
            "This is ![old alt](path/to/image.png ) in text",
            "old alt",
            "This is ![new alt text](path/to/image.png) in text",
            id="whitespace_before_paren",
        ),
    ],
)
def test_apply_markdown_image_alt(
    line: str, expected_old_alt: str | None, expected_new_line: str
) -> None:
    """Test applying alt text to markdown image syntax."""
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "path/to/image.png", "new alt text"
    )
    assert old_alt == expected_old_alt
    assert new_line == expected_new_line


@pytest.mark.parametrize(
    "line,expected_old_alt,expected_new_line",
    [
        pytest.param(
            '<img alt="old alt" src="path/to/image.png">',
            "old alt",
            '<img alt="new alt text" src="path/to/image.png"/>',
            id="existing_alt",
        ),
        pytest.param(
            '<img src="path/to/image.png">',
            None,
            '<img alt="new alt text" src="path/to/image.png"/>',
            id="no_alt",
        ),
        pytest.param(
            '<img alt="old alt" src="path/to/image.png" class="theme-emoji"/>',
            "old alt",
            '<img alt="new alt text" class="theme-emoji" src="path/to/image.png"/>',
            id="self_closing",
        ),
        pytest.param(
            '<img src="path/to/image.png" class="icon"/>',
            None,
            '<img alt="new alt text" class="icon" src="path/to/image.png"/>',
            id="self_closing_no_alt",
        ),
    ],
)
def test_apply_html_image_alt(
    line: str, expected_old_alt: str | None, expected_new_line: str
) -> None:
    """Test applying alt text to HTML img tags."""
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )
    assert old_alt == expected_old_alt
    assert new_line == expected_new_line


def test_apply_html_image_alt_parser_rejected_markup_is_ignored() -> None:
    """BeautifulSoup can reject non-HTML markup; we should ignore those lines."""
    # This is representative of the kind of JS/TS regex literal that triggered
    # bs4.exceptions.ParserRejectedMarkup in real content.
    line = "const notePattern = /^\\s*[*_]*note[*_]*:[*_]* (?<text>.*)(?<![*_])[*_]*/gim;"
    new_line, old_alt = apply._apply_html_image_alt(
        line, "path/to/image.png", "new alt text"
    )

    assert old_alt is None
    assert new_line == line


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


def test_apply_captions_multiple_images(temp_dir: Path, console: Console) -> None:
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


@pytest.mark.parametrize(
    "line,asset_path,expected_old_alt,expected_new_line",
    [
        pytest.param(
            "This is ![[path/to/image.png|old alt]] in text",
            "path/to/image.png",
            "old alt",
            "This is ![[path/to/image.png|new alt text]] in text",
            id="existing_alt",
        ),
        pytest.param(
            "This is ![[path/to/image.png]] in text",
            "path/to/image.png",
            None,
            "This is ![[path/to/image.png|new alt text]] in text",
            id="no_alt",
        ),
        pytest.param(
            "![[https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif]]",
            "https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif",
            None,
            "![[https://assets.turntrout.com/static/images/posts/distillation-robustifies-unlearning-20250612141417.avif|new alt text]]",
            id="full_url",
        ),
        pytest.param(
            "This is ![markdown](image.png) not wikilink",
            "image.png",
            None,
            "This is ![markdown](image.png) not wikilink",
            id="no_match",
        ),
    ],
)
def test_apply_wikilink_image_alt(
    line: str, asset_path: str, expected_old_alt: str | None, expected_new_line: str
) -> None:
    """Test applying alt text to wikilink image syntax."""
    new_line, old_alt = apply._apply_wikilink_image_alt(
        line, asset_path, "new alt text"
    )
    assert old_alt == expected_old_alt
    assert new_line == expected_new_line


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


@pytest.mark.parametrize(
    "line,new_alt,expected_old_alt,expected_escaped",
    [
        (
            "This is ![old alt](image.png) in text",
            r"A diagram\showing states A, B, and C",
            "old alt",
            r"A diagram\\showing states A, B, and C",
        ),
        (
            "This is ![](image.png) in text",
            "Price is $100 and $200",
            None,
            r"Price is \$100 and \$200",
        ),
        (
            "![](test.png)",
            r"A diagram (version 1.0) showing $variable\in set {A, B, C}",
            None,
            r"A diagram (version 1.0) showing \$variable\\in set {A, B, C}",
        ),
    ],
)
def test_markdown_image_alt_with_special_chars(
    line: str,
    new_alt: str,
    expected_old_alt: str | None,
    expected_escaped: str,
) -> None:
    """Test applying markdown alt text with special characters that need escaping."""
    new_line, old_alt = apply._apply_markdown_image_alt(
        line, "image.png" if "image.png" in line else "test.png", new_alt
    )
    assert old_alt == expected_old_alt
    asset_path = "image.png" if "image.png" in line else "test.png"
    expected_line = line.replace(
        f"![{expected_old_alt or ''}]({asset_path})",
        f"![{expected_escaped}]({asset_path})",
    )
    assert new_line == expected_line


@pytest.mark.parametrize(
    "line,new_alt,expected_old_alt,expected_escaped",
    [
        # Backslashes don't need escaping in HTML
        (
            '<img alt="old" src="image.png">',
            r"Diagram\showing process",
            "old",
            r"Diagram\showing process",
        ),
        (
            '<img src="image.png">',
            r"A state transition diagram\showing paths",
            None,
            r"A state transition diagram\showing paths",
        ),
        # HTML special characters should be escaped
        (
            '<img alt="old" src="image.png">',
            'Tom & Jerry <3 "quotes"',
            "old",
            "Tom &amp; Jerry &lt;3 &quot;quotes&quot;",
        ),
        (
            '<img src="image.png">',
            "x < y > z & more",
            None,
            "x &lt; y &gt; z &amp; more",
        ),
    ],
)
def test_html_image_alt_with_special_chars(
    line: str,
    new_alt: str,
    expected_old_alt: str | None,
    expected_escaped: str,
) -> None:
    """Test applying HTML alt text with special characters."""
    new_line, old_alt = apply._apply_html_image_alt(line, "image.png", new_alt)
    assert old_alt == expected_old_alt
    # Check that the alt attribute is present with correct escaping
    assert 'src="image.png"' in new_line
    assert (
        expected_escaped in new_line
        or expected_escaped.replace("&quot;", '"') in new_line
    )
    assert new_line.endswith("/>")


@pytest.mark.parametrize(
    "line,new_alt,expected_old_alt,expected_escaped",
    [
        (
            "Image: ![[image.png|old alt]] here",
            r"Diagram\with backslash",
            "old alt",
            r"Diagram\\with backslash",
        ),
        (
            "Image: ![[image.png]] here",
            "Cost $100",
            None,
            r"Cost \$100",
        ),
    ],
)
def test_wikilink_image_alt_with_special_chars(
    line: str,
    new_alt: str,
    expected_old_alt: str | None,
    expected_escaped: str,
) -> None:
    """Test applying wikilink alt text with special characters (uses markdown escaping)."""
    new_line, old_alt = apply._apply_wikilink_image_alt(line, "image.png", new_alt)
    assert old_alt == expected_old_alt
    expected_line = line.replace(
        f"![[image.png{f'|{expected_old_alt}' if expected_old_alt else ''}]]",
        f"![[image.png|{expected_escaped}]]",
    )
    assert new_line == expected_line


@pytest.mark.parametrize(
    "unused_entries,expected_count_text,expected_entries",
    [
        pytest.param(set(), None, [], id="empty"),
        pytest.param(
            {("path/to/file.md", "image.png")},
            "1 entry without 'final_alt' will be skipped:",
            ["path/to/file.md: image.png"],
            id="single",
        ),
        pytest.param(
            {
                ("path/to/file1.md", "image1.png"),
                ("path/to/file2.md", "image2.png"),
                ("path/to/file1.md", "image3.png"),
            },
            "3 entries without 'final_alt' will be skipped:",
            [
                "path/to/file1.md: image1.png",
                "path/to/file1.md: image3.png",
                "path/to/file2.md: image2.png",
            ],
            id="multiple",
        ),
    ],
)
def test_display_unused_entries(
    unused_entries: set[tuple[str, str]],
    expected_count_text: str | None,
    expected_entries: list[str],
) -> None:
    """Test displaying unused entries with various counts."""
    output = StringIO()
    console = Console(file=output, width=120)
    apply._display_unused_entries(unused_entries, console)
    result = output.getvalue()
    if expected_count_text is None:
        assert result == ""
    else:
        assert expected_count_text in result
        for entry in expected_entries:
            assert entry in result


def test_display_unused_entries_sorted() -> None:
    """Test that unused entries are displayed in sorted order."""
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
        line.strip() for line in result.splitlines() if ":" in line and ".md" in line
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


@pytest.mark.parametrize(
    "line,new_alt,expected_fragment",
    [
        # Markdown: Unix line break
        (
            "![old](image.png)",
            "Line one\nLine two",
            "![Line one ... Line two](image.png)",
        ),
        # Markdown: Multiple consecutive line breaks (should collapse)
        (
            "![old](image.png)",
            "Line one\n\nLine two",
            "![Line one ... Line two](image.png)",
        ),
        # Markdown: Many consecutive line breaks
        (
            "![old](image.png)",
            "Line one\n\n\n\nLine two",
            "![Line one ... Line two](image.png)",
        ),
        # HTML: Unix line break
        (
            '<img src="image.png">',
            "Line one\nLine two",
            'alt="Line one ... Line two"',
        ),
        # HTML: Windows line break
        (
            '<img src="image.png">',
            "Line one\r\nLine two",
            'alt="Line one ... Line two"',
        ),
        # HTML: Multiple consecutive line breaks
        (
            '<img src="image.png">',
            "Line one\n\n\nLine two",
            'alt="Line one ... Line two"',
        ),
        # Wikilink: Unix line break
        (
            "![[image.png]]",
            "Line one\nLine two",
            "![[image.png|Line one ... Line two]]",
        ),
        # Wikilink: Multiple line breaks
        ("![[image.png|old]]", "A\nB\nC", "![[image.png|A ... B ... C]]"),
        # Mixed line break types
        (
            "![old](image.png)",
            "A\nB\r\nC\rD",
            "![A ... B ... C ... D](image.png)",
        ),
        # Mixed consecutive line breaks
        (
            "![old](image.png)",
            "A\n\r\n\nB",
            "![A ... B](image.png)",
        ),
        # Leading newline
        (
            "![old](image.png)",
            "\nLine one",
            "![ ... Line one](image.png)",
        ),
        # Trailing newline
        (
            "![old](image.png)",
            "Line one\n",
            "![Line one ... ](image.png)",
        ),
        # Only newlines
        (
            "![old](image.png)",
            "\n\n\n",
            "![ ... ](image.png)",
        ),
        # Newlines with text that needs escaping
        (
            "![old](image.png)",
            "Cost $100\nAnother $200",
            r"![Cost \$100 ... Another \$200](image.png)",
        ),
    ],
)
def test_try_all_image_formats_normalizes_line_breaks(
    line: str, new_alt: str, expected_fragment: str
) -> None:
    """Test that line breaks in alt text are replaced with ellipses."""
    new_line, _ = apply._try_all_image_formats(line, "image.png", new_alt)
    assert expected_fragment in new_line


def test_apply_caption_with_line_breaks_end_to_end(
    temp_dir: Path, console: Console
) -> None:
    """Test end-to-end that line breaks in captions are replaced with ellipses."""
    md_path = temp_dir / "test.md"
    content = """# Test File

Image: ![old alt](image.png)
"""
    md_path.write_text(content)

    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": str(md_path),
            "asset_path": "image.png",
            "line_number": 3,
            "suggested_alt": "suggested",
            "final_alt": "First line\nSecond line\nThird line",
            "model": "test-model",
            "context_snippet": "context",
        }
    ]
    captions_path.write_text(json.dumps(captions_data))

    applied_count = apply.apply_captions(captions_path, console, dry_run=False)

    assert applied_count == 1
    new_content = md_path.read_text()
    assert "![First line ... Second line ... Third line](image.png)" in new_content
    assert "\n" not in new_content.split("![")[1].split("]")[0]


def test_apply_caption_with_none_line_number(temp_dir: Path, console: Console) -> None:
    """Test that apply works when line_number is None."""
    md_path = temp_dir / "test.md"
    content = """# Test
    
![old alt](image.png)

More content.
"""
    md_path.write_text(content)

    caption_item = utils.AltGenerationResult(
        markdown_file=str(md_path),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=None,
        final_alt="new alt text",
    )

    result = apply._apply_caption_to_file(md_path, caption_item, console)

    assert result is not None
    old_alt, new_alt = result
    assert old_alt == "old alt"
    assert new_alt == "new alt text"

    new_content = md_path.read_text()
    assert "![new alt text](image.png)" in new_content
    assert "![old alt](image.png)" not in new_content


def test_apply_caption_replaces_all_instances(temp_dir: Path, console: Console) -> None:
    """Test that all instances of an asset get replaced."""
    md_path = temp_dir / "test.md"
    content = """# Test

First instance: ![old alt 1](image.png)

Some text.

Second instance: ![old alt 2](image.png)

More text.

Third instance: ![](image.png)
"""
    md_path.write_text(content)

    caption_item = utils.AltGenerationResult(
        markdown_file=str(md_path),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=None,
        final_alt="new alt text",
    )

    result = apply._apply_caption_to_file(md_path, caption_item, console)

    assert result is not None

    new_content = md_path.read_text()
    # All three instances should be updated
    assert new_content.count("![new alt text](image.png)") == 3
    assert "old alt 1" not in new_content
    assert "old alt 2" not in new_content


def test_apply_caption_with_mixed_formats(temp_dir: Path, console: Console) -> None:
    """Test that all formats (markdown, HTML, wikilink) get replaced."""
    md_path = temp_dir / "test.md"
    content = """# Test

Markdown: ![old alt](image.png)

HTML: <img src="image.png" alt="old html alt">

Wikilink: ![[image.png|old wikilink alt]]
"""
    md_path.write_text(content)

    caption_item = utils.AltGenerationResult(
        markdown_file=str(md_path),
        asset_path="image.png",
        suggested_alt="suggested",
        model="test-model",
        context_snippet="context",
        line_number=None,
        final_alt="new alt text",
    )

    result = apply._apply_caption_to_file(md_path, caption_item, console)

    assert result is not None

    new_content = md_path.read_text()
    # All three formats should be updated
    assert "![new alt text](image.png)" in new_content
    assert 'alt="new alt text"' in new_content
    assert "![[image.png|new alt text]]" in new_content
    assert "old alt" not in new_content
    assert "old html alt" not in new_content
    assert "old wikilink alt" not in new_content


"""Tests for video label application functionality."""


class TestApplyHtmlVideoLabel:
    """Test applying aria-label to HTML video elements."""

    @pytest.mark.parametrize(
        "line,asset_path,new_label,expected_old_label,expected_fragment",
        [
            pytest.param(
                '<video src="demo.mp4" controls></video>',
                "demo.mp4", "Product demo", None, 'aria-label="Product demo"',
                id="add_to_bare_video",
            ),
            pytest.param(
                '<video src="demo.mp4" aria-label="old label"></video>',
                "demo.mp4", "New description", "old label", 'aria-label="New description"',
                id="replace_aria_label",
            ),
            pytest.param(
                '<video src="demo.mp4" title="old title"></video>',
                "demo.mp4", "New description", "old title", 'aria-label="New description"',
                id="replace_title",
            ),
            pytest.param(
                '<video controls><source src="demo.mp4" type="video/mp4"></video>',
                "demo.mp4", "Tutorial video", None, 'aria-label="Tutorial video"',
                id="source_child",
            ),
            pytest.param(
                '<video controls>\n  <source src="demo.mp4" type="video/mp4">\n  <source src="demo.webm" type="video/webm">\n</video>',
                "demo.mp4", "Demo video", None, 'aria-label="Demo video"',
                id="multiple_sources",
            ),
            pytest.param(
                '<video src="demo.mp4" aria-describedby="desc-1"></video>',
                "demo.mp4", "New label", "desc-1", 'aria-label="New label"',
                id="aria_describedby",
            ),
            pytest.param(
                '<video src="demo.mp4"></video>',
                "demo.mp4", 'Tutorial: "Getting Started" & More', None, "aria-label=",
                id="special_chars_in_label",
            ),
            pytest.param(
                '<video src="https://example.com/video.mp4"></video>',
                "https://example.com/video.mp4", "Remote video", None, 'aria-label="Remote video"',
                id="url_src",
            ),
            pytest.param(
                "<video src='demo.mp4'></video>",
                "demo.mp4", "Description", None, 'aria-label="Description"',
                id="single_quotes",
            ),
            pytest.param(
                "<video src=demo.mp4></video>",
                "demo.mp4", "Description", None, 'aria-label="Description"',
                id="no_quotes",
            ),
            pytest.param(
                '<video\n  src="demo.mp4"\n  controls\n  autoplay>\n</video>',
                "demo.mp4", "Demo", None, 'aria-label="Demo"',
                id="multiline",
            ),
            pytest.param(
                '<video src="demo.mp4" controls autoplay loop muted playsinline style="width: 80%"></video>',
                "demo.mp4", "Looping demo", None, 'aria-label="Looping demo"',
                id="many_attributes",
            ),
            pytest.param(
                '<video src="demo.mp4"></video>',
                "demo.mp4", "", None, 'aria-label=""',
                id="empty_label",
            ),
            pytest.param(
                '<video src="demo.mp4"></video>',
                "demo.mp4", "Multi\nline\ndescription", None, "aria-label=",
                id="label_with_newlines",
            ),
        ],
    )
    def test_apply_video_label(
        self, line: str, asset_path: str, new_label: str,
        expected_old_label: str | None, expected_fragment: str,
    ) -> None:
        """Test adding/replacing aria-label on video elements."""
        new_line, old_label = apply._apply_html_video_label(line, asset_path, new_label)
        assert expected_fragment in new_line
        assert old_label == expected_old_label

    def test_no_match_returns_unchanged(self):
        """Should return unchanged line if video not found."""
        line = '<video src="other.mp4"></video>'
        new_line, old_label = apply._apply_html_video_label(
            line, "demo.mp4", "Description"
        )
        assert new_line == line
        assert old_label is None

    def test_preserves_other_attributes(self):
        """Should preserve all other video attributes."""
        line = '<video src="demo.mp4" controls autoplay class="video-player"></video>'
        new_line, _ = apply._apply_html_video_label(line, "demo.mp4", "Demo")
        assert "controls" in new_line
        assert "autoplay" in new_line
        assert 'class="video-player"' in new_line

    def test_multiple_videos_in_line(self):
        """Should only modify the matching video."""
        line = '<video src="demo1.mp4"></video><video src="demo2.mp4"></video>'
        new_line, _ = apply._apply_html_video_label(line, "demo1.mp4", "First video")
        assert 'aria-label="First video"' in new_line
        assert new_line.count("aria-label") == 1

    def test_video_with_closing_tag(self):
        """Should handle video with explicit closing tag content."""
        line = '<video src="demo.mp4">Your browser does not support video.</video>'
        new_line, _ = apply._apply_html_video_label(line, "demo.mp4", "Demo video")
        assert 'aria-label="Demo video"' in new_line
        assert "Your browser does not support video." in new_line


# ---------------------------------------------------------------------------
# Edge cases and boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,apply_fn,alt_text,expected",
    [
        pytest.param(
            "![](image.png)",
            apply._apply_markdown_image_alt,
            "日本語の代替テキスト",
            "![日本語の代替テキスト](image.png)",
            id="markdown_unicode",
        ),
        pytest.param(
            '<img src="image.png">',
            apply._apply_html_image_alt,
            "中文替代文字",
            '<img alt="中文替代文字" src="image.png"/>',
            id="html_unicode",
        ),
    ],
)
def test_unicode_alt_text(line: str, apply_fn, alt_text: str, expected: str) -> None:
    """Unicode alt text should be applied correctly in both formats."""
    new_line, _ = apply_fn(line, "image.png", alt_text)
    assert new_line == expected


@pytest.mark.parametrize(
    "source,expect_trailing_newline",
    [
        pytest.param("![old](image.png)\n", True, id="preserves_trailing_newline"),
        pytest.param("![old](image.png)", False, id="preserves_no_trailing_newline"),
    ],
)
def test_apply_preserves_trailing_newline(
    temp_dir: Path, console: Console, source: str, expect_trailing_newline: bool
) -> None:
    """Apply should not alter whether the file ends with a newline."""
    md_path = temp_dir / "test.md"
    md_path.write_text(source, encoding="utf-8")
    caption = utils.AltGenerationResult(
        markdown_file=str(md_path),
        asset_path="image.png",
        suggested_alt="s",
        model="m",
        context_snippet="c",
        final_alt="new",
    )
    apply._apply_caption_to_file(md_path, caption, console)
    assert md_path.read_text().endswith("\n") == expect_trailing_newline


def test_apply_to_nonexistent_file(temp_dir: Path) -> None:
    """Applying to a nonexistent file should warn, not crash."""
    md_path = temp_dir / "nonexistent.md"
    captions_path = temp_dir / "captions.json"
    captions_data = [
        {
            "markdown_file": str(md_path),
            "asset_path": "img.png",
            "line_number": 1,
            "suggested_alt": "s",
            "final_alt": "new",
            "model": "m",
            "context_snippet": "c",
        }
    ]
    captions_path.write_text(json.dumps(captions_data))
    console = Console(file=StringIO())
    result = apply.apply_captions(captions_path, console)
    assert result == 0


def test_apply_caption_asset_not_in_file(temp_dir: Path, console: Console) -> None:
    """Applying caption when asset doesn't exist in the file."""
    md_path = temp_dir / "test.md"
    md_path.write_text("# No images here\n", encoding="utf-8")
    caption = utils.AltGenerationResult(
        markdown_file=str(md_path),
        asset_path="nonexistent.png",
        suggested_alt="s",
        model="m",
        context_snippet="c",
        final_alt="new alt",
    )
    result = apply._apply_caption_to_file(md_path, caption, console)
    assert result is None


def test_apply_with_many_captions(temp_dir: Path) -> None:
    """Apply 100 captions to a file with 100 images."""
    md_path = temp_dir / "test.md"
    lines = [f"![old_{i}](image_{i}.png)" for i in range(100)]
    md_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

    captions_data = [
        {
            "markdown_file": str(md_path),
            "asset_path": f"image_{i}.png",
            "line_number": i * 2 + 1,
            "suggested_alt": f"suggested_{i}",
            "final_alt": f"new_caption_{i}",
            "model": "m",
            "context_snippet": "c",
        }
        for i in range(100)
    ]
    captions_path = temp_dir / "captions.json"
    captions_path.write_text(json.dumps(captions_data))
    console = Console(file=StringIO())
    result = apply.apply_captions(captions_path, console)
    assert result == 100

    content = md_path.read_text()
    for i in range(100):
        assert f"![new_caption_{i}](image_{i}.png)" in content


# ---------------------------------------------------------------------------
# Integration / round-trip tests
# ---------------------------------------------------------------------------


def test_scan_then_apply_roundtrip(temp_dir: Path) -> None:
    """Scan a file, create captions, apply them, then rescan finds nothing."""
    md_path = temp_dir / "article.md"
    md_content = textwrap.dedent("""\
        # My Article

        Here is an image:

        ![](photo.jpg)

        And another:

        <img src="diagram.png">

        And a wikilink:

        ![[chart.svg]]
    """)
    md_path.write_text(md_content, encoding="utf-8")

    queue = scan.build_queue(temp_dir)
    assert len(queue) == 3

    captions_data = [
        {
            "markdown_file": item.markdown_file,
            "asset_path": item.asset_path,
            "line_number": item.line_number,
            "suggested_alt": f"Suggestion for {item.asset_path}",
            "final_alt": f"Alt text for {item.asset_path}",
            "model": "test-model",
            "context_snippet": item.context_snippet,
        }
        for item in queue
    ]

    captions_path = temp_dir / "captions.json"
    captions_path.write_text(json.dumps(captions_data))

    console = Console(file=StringIO())
    applied = apply.apply_captions(captions_path, console)
    assert applied == 3

    new_content = md_path.read_text()
    assert "![Alt text for photo.jpg](photo.jpg)" in new_content
    assert 'alt="Alt text for diagram.png"' in new_content
    assert "![[chart.svg|Alt text for chart.svg]]" in new_content

    # Rescan: all images now have alt text, so queue should be empty
    assert scan.build_queue(temp_dir) == []


def test_large_mixed_file_scan_and_apply(temp_dir: Path) -> None:
    """Large file with mixed formats: scan, mock-caption, apply, verify."""
    lines = ["# Large Document\n"]
    for i in range(30):
        lines.append(f"\nParagraph {i} with some text.\n")
        if i % 3 == 0:
            lines.append(f"\n![](markdown_{i}.png)\n")
        elif i % 3 == 1:
            lines.append(f'\n<img src="html_{i}.jpg">\n')
        else:
            lines.append(f"\n![[wiki_{i}.gif]]\n")

    md_path = temp_dir / "large.md"
    md_path.write_text("".join(lines), encoding="utf-8")

    queue = scan.build_queue(temp_dir)
    assert len(queue) == 30

    captions = [
        {
            "markdown_file": str(md_path),
            "asset_path": item.asset_path,
            "line_number": item.line_number,
            "suggested_alt": "s",
            "final_alt": f"Description {item.asset_path}",
            "model": "m",
            "context_snippet": "c",
        }
        for item in queue
    ]
    cp = temp_dir / "captions.json"
    cp.write_text(json.dumps(captions))
    console = Console(file=StringIO())
    applied = apply.apply_captions(cp, console)
    assert applied == 30

    queue_after = scan.build_queue(temp_dir)
    assert len(queue_after) == 0
