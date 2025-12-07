import textwrap
from pathlib import Path
from markdown_it.token import Token

import pytest

from alt_text_llm import scan


@pytest.mark.parametrize(
    "alt, expected",
    [
        (None, False),
        ("", False),
        ("   ", False),
        ("image", False),
        ("A meaningful description", True),
        ("Meaningful", True),
    ],
)
def test_is_alt_meaningful(alt: str | None, expected: bool) -> None:
    assert scan._is_alt_meaningful(alt) is expected


def _write_md(tmp_path: Path, content: str, name: str = "test.md") -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_build_queue_markdown_asset(tmp_path: Path) -> None:
    md_content = """
Paragraph one.

![](img/foo.png)

Paragraph two.
"""
    _write_md(tmp_path, md_content)
    queue = scan.build_queue(tmp_path)
    assert len(queue) == 1
    item = queue[0]
    assert item.asset_path == "img/foo.png"
    assert item.line_number == 4
    assert "Paragraph one." in item.context_snippet
    assert "Paragraph two." in item.context_snippet


def test_build_queue_html_img_missing_alt(tmp_path: Path) -> None:
    md_content = """
Intro.

<img src=\"assets/pic.jpg\">
"""
    _write_md(tmp_path, md_content, "html.md")
    queue = scan.build_queue(tmp_path)
    assert len(queue) == 1, f"{queue} doesn't have the right elements"
    assert queue[0].asset_path == "assets/pic.jpg"


def test_build_queue_ignores_good_alt(tmp_path: Path) -> None:
    md_content = "![](foo.png)\n\n![Good alt](bar.png)"
    _write_md(tmp_path, md_content)
    queue = scan.build_queue(tmp_path)

    # only the empty alt should be queued
    assert len(queue) == 1, f"{queue} doesn't have the right elements"
    assert queue[0].asset_path == "foo.png"


@pytest.mark.parametrize(
    "content, expected_paths",
    [
        ("![](img/blank.png)", ["img/blank.png"]),
        ("![Good desc](img/good.png)", []),
        (
            '<img src="assets/foo.jpg" alt="photo">\n',
            ["assets/foo.jpg"],
        ),
        (
            '<img src="assets/bar.jpg" alt="Meaningful description">\n',
            [],
        ),
        (
            '<img src="assets/baz.jpg" alt="">\n',
            [],
        ),
    ],
)
def test_queue_expected_paths(
    tmp_path: Path, content: str, expected_paths: list[str]
) -> None:
    """Verify that *build_queue* includes exactly the expected offending assets."""

    file_path = tmp_path / "edge.md"
    file_path.write_text(content, encoding="utf-8")

    queue = scan.build_queue(tmp_path)
    assert sorted(item.asset_path for item in queue) == sorted(expected_paths)


def test_html_img_line_number_fallback(tmp_path: Path) -> None:
    """If markdown-it does not supply *token.map* for an HTML image, the
    fallback logic should locate the correct source line instead of defaulting
    to 1."""

    md_content = textwrap.dedent(
        """
        Intro line.

        <img src="assets/foo.jpg">

        After image.
        """
    )
    _write_md(tmp_path, md_content, "fallback.md")

    queue = scan.build_queue(tmp_path)
    assert len(queue) == 1

    item = queue[0]
    # The <img> tag is on the 4th line of the file (1-based)
    assert item.line_number == 4, f"Expected line 4, got {item.line_number}"


def test_html_img_line_number_with_frontmatter(tmp_path: Path) -> None:
    """Ensure line numbers for HTML images located *after* YAML front-matter
    are computed relative to the full file."""

    md_content = textwrap.dedent(
        """
        ---
        title: Sample
        ---

        Preamble text.

        <img src="assets/bar.jpg">
        """
    )
    file_path = _write_md(tmp_path, md_content, "frontmatter.md")

    # Sanity-check the actual line number of the <img> element
    img_line_no = next(
        idx + 1
        for idx, ln in enumerate(file_path.read_text().splitlines())
        if "<img" in ln
    )

    queue = scan.build_queue(tmp_path)
    assert len(queue) == 1
    assert queue[0].line_number == img_line_no


def test_get_line_number_raises_error_when_asset_not_found(
    tmp_path: Path,
) -> None:
    """Test that _get_line_number raises ValueError when asset can't be found in file."""

    # Create a markdown file without the asset we're looking for
    md_content = textwrap.dedent(
        """
        # Title
        
        Some content here.
        
        ![Different image](other.png)
        """
    )
    file_path = _write_md(tmp_path, md_content, "missing_asset.md")

    # Create a token without map info to force fallback search
    token = Token("image", "", 0)
    token.map = None

    lines = file_path.read_text().splitlines()

    # This should raise ValueError since "nonexistent.png" is not in the file
    with pytest.raises(
        ValueError,
        match="Could not find asset '\\(nonexistent.png\\)' in markdown file",
    ):
        scan._get_line_number(token, lines, "(nonexistent.png)")


def test_html_img_error_when_src_not_in_content(tmp_path: Path) -> None:
    """Test that HTML img with empty alt (decorative) is not queued."""
    md_content = textwrap.dedent(
        """
        # Title
        
        Some content.
        
        <img src="findable.jpg" alt="">
        """
    )
    _write_md(tmp_path, md_content, "html_test.md")

    # Empty alt indicates decorative image, should not be queued
    queue = scan.build_queue(tmp_path)
    assert len(queue) == 0


@pytest.mark.parametrize(
    "content, expected_paths",
    [
        # Wikilink without alt
        ("![[assets/image.png]]", ["assets/image.png"]),
        # Wikilink without meaningful alt
        ("![[assets/image.png|]]", ["assets/image.png"]),
        # Wikilink with meaningful alt
        ("![[assets/image.png|Good description]]", []),
        # Wikilink with placeholder alt
        ("![[assets/image.png|image]]", ["assets/image.png"]),
        # Wikilink with URL
        ("![[https://example.com/image.png]]", ["https://example.com/image.png"]),
        # Multiple wikilinks mixed
        (
            "![[img1.png|image]]\n![[img2.png|Good alt]]\n![[img3.png]]",
            ["img1.png", "img3.png"],
        ),
    ],
)
def test_wikilink_formats(
    tmp_path: Path, content: str, expected_paths: list[str]
) -> None:
    """Test wikilink image detection with various formats."""
    _write_md(tmp_path, content, "wikilink.md")
    queue = scan.build_queue(tmp_path)
    assert sorted(item.asset_path for item in queue) == sorted(expected_paths)


def test_mixed_image_formats(tmp_path: Path) -> None:
    """Test that markdown, HTML, and wikilink images are all detected."""
    md_content = (
        "![](markdown.png)\n"
        '<img src="html.jpg">\n'
        "![[wikilink.gif]]\n"
        "![Good](good-md.png)\n"
        '<img src="good-html.jpg" alt="Good">\n'
        "![[good-wiki.png|Good]]"
    )
    _write_md(tmp_path, md_content, "mixed.md")

    queue = scan.build_queue(tmp_path)
    paths = {item.asset_path for item in queue}
    assert paths == {"markdown.png", "html.jpg", "wikilink.gif"}


def test_build_queue_wikilink_without_alt(tmp_path: Path) -> None:
    """Test that wikilink images without alt text are detected."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        Here is a wikilink image without alt text:
        
        ![[assets/image.png]]
        
        End of document.
        """
    )
    _write_md(tmp_path, md_content, "wikilink.md")

    queue = scan.build_queue(tmp_path)
    assert len(queue) == 1
    assert queue[0].asset_path == "assets/image.png"
    assert "Here is a wikilink image" in queue[0].context_snippet


def test_build_queue_wikilink_with_alt(tmp_path: Path) -> None:
    """Test that wikilink images with meaningful alt text are not queued."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        ![[assets/image.png|A meaningful description of the image]]
        """
    )
    _write_md(tmp_path, md_content, "wikilink_with_alt.md")

    queue = scan.build_queue(tmp_path)
    assert len(queue) == 0


def test_build_queue_wikilink_with_placeholder_alt(tmp_path: Path) -> None:
    """Test that wikilink images with placeholder alt text are queued."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        ![[assets/image.png|image]]
        ![[assets/photo.jpg|photo]]
        ![[assets/pic.png|Good description]]
        """
    )
    _write_md(tmp_path, md_content, "wikilink_placeholder.md")

    queue = scan.build_queue(tmp_path)
    # First two should be queued (placeholder alts), third should not
    assert len(queue) == 2
    paths = {item.asset_path for item in queue}
    assert "assets/image.png" in paths
    assert "assets/photo.jpg" in paths
    assert "assets/pic.png" not in paths


def test_build_queue_wikilink_with_url(tmp_path: Path) -> None:
    """Test that wikilink images with full URLs are detected."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        ![[https://example.com/image.png]]
        ![[https://example.com/photo.jpg|Good alt text]]
        """
    )
    _write_md(tmp_path, md_content, "wikilink_url.md")

    queue = scan.build_queue(tmp_path)
    # Only the first one (without alt) should be queued
    assert len(queue) == 1
    assert queue[0].asset_path == "https://example.com/image.png"


def test_build_queue_mixed_formats(tmp_path: Path) -> None:
    """Test that all image formats (markdown, HTML, wikilink) are detected together."""
    md_content = textwrap.dedent(
        """
        # Mixed Formats
        
        Markdown without alt: ![](markdown.png)
        
        HTML without alt: <img src="html.jpg">
        
        Wikilink without alt: ![[wikilink.gif]]
        
        Markdown with alt: ![Good description](good-markdown.png)
        
        HTML with alt: <img src="good-html.jpg" alt="Good description">
        
        Wikilink with alt: ![[good-wikilink.png|Good description]]
        """
    )
    _write_md(tmp_path, md_content, "mixed.md")

    queue = scan.build_queue(tmp_path)
    # Should find 3 images without meaningful alt text
    assert len(queue) == 3
    paths = {item.asset_path for item in queue}
    assert "markdown.png" in paths
    assert "html.jpg" in paths
    assert "wikilink.gif" in paths
    # Should not include images with good alt text
    assert "good-markdown.png" not in paths
    assert "good-html.jpg" not in paths
    assert "good-wikilink.png" not in paths


def test_wikilink_line_numbers(tmp_path: Path) -> None:
    """Test that line numbers are correctly identified for wikilink images."""
    md_content = textwrap.dedent(
        """
        ---
        title: Test
        ---
        
        First paragraph.
        
        ![[image1.png]]
        
        Second paragraph.
        
        ![[image2.png]]
        
        Third paragraph.
        """
    )
    file_path = _write_md(tmp_path, md_content, "line_numbers.md")

    queue = scan.build_queue(tmp_path)
    assert len(queue) == 2

    # Find actual line numbers in the file
    lines = file_path.read_text().splitlines()
    image1_line = next(i + 1 for i, line in enumerate(lines) if "image1.png" in line)
    image2_line = next(i + 1 for i, line in enumerate(lines) if "image2.png" in line)

    # Check that detected line numbers match
    line_numbers = {item.line_number for item in queue}
    assert image1_line in line_numbers
    assert image2_line in line_numbers


def test_wikilink_document_embeds_not_treated_as_images(tmp_path: Path) -> None:
    """Test that wikilink document embeds (with # but no image extension) are not treated as images."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        This is a document embed (not an image):
        ![[output-feedback-can-obfuscate-chain-of-thought#]]
        
        This is also a document embed with a heading:
        ![[another-document#specific-heading]]
        
        This is also NOT an image (has # fragment):
        ![[diagram.png#light-mode]]
        
        But this IS a regular image:
        ![[photo.jpg]]
        """
    )
    _write_md(tmp_path, md_content, "doc_embeds.md")

    queue = scan.build_queue(tmp_path)

    # Should only find the one actual image (photo.jpg)
    # All wikilinks with # are document/section embeds, not images
    assert len(queue) == 1
    assert queue[0].asset_path == "photo.jpg"

    # Should NOT include any embeds with #
    paths = {item.asset_path for item in queue}
    assert "output-feedback-can-obfuscate-chain-of-thought#" not in paths
    assert "another-document#specific-heading" not in paths
    assert "diagram.png#light-mode" not in paths


def test_wikilink_non_image_files_ignored(tmp_path: Path) -> None:
    """Test that wikilinks to non-image files are ignored."""
    md_content = textwrap.dedent(
        """
        # Test Document
        
        Link to a PDF: ![[document.pdf]]
        
        Link to a markdown file: ![[notes.md]]
        
        Link to a text file: ![[readme.txt]]
        
        But this is an image: ![[photo.png]]
        """
    )
    _write_md(tmp_path, md_content, "non_images.md")

    queue = scan.build_queue(tmp_path)

    # Should only find the actual image
    assert len(queue) == 1
    assert queue[0].asset_path == "photo.png"


@pytest.mark.parametrize(
    "extension", sorted([ext.lstrip(".") for ext in scan.WIKILINK_ASSET_EXTENSIONS])
)
def test_all_image_extensions_recognized(tmp_path: Path, extension: str) -> None:
    """Test that all IMAGE_EXTENSIONS are properly recognized as images in wikilinks."""
    filename = f"test_image.{extension}"
    md_content = f"![[{filename}]]"
    _write_md(tmp_path, md_content, "extension_test.md")

    queue = scan.build_queue(tmp_path)

    # Should find the image regardless of extension
    assert len(queue) == 1
    assert queue[0].asset_path == filename


@pytest.mark.parametrize(
    "extension", sorted([ext.lstrip(".") for ext in scan.WIKILINK_ASSET_EXTENSIONS])
)
def test_image_extensions_with_fragment_not_recognized(
    tmp_path: Path, extension: str
) -> None:
    """Test that wikilinks with # fragments are NOT treated as images (they're document embeds)."""
    filename = f"diagram.{extension}#light-mode"
    md_content = f"![[{filename}]]"
    _write_md(tmp_path, md_content, "fragment_test.md")

    queue = scan.build_queue(tmp_path)

    # Should NOT find images with # fragments - they're document embeds
    assert len(queue) == 0
