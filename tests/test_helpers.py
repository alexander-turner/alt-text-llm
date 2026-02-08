"""Helper functions for test setup."""

from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def create_test_image(
    path: Path,
    size: str,
    *,
    colorspace: str | None = None,
    background: str | None = None,
    draw: str | None = None,
    metadata: str | None = None,
) -> None:
    """Write a minimal dummy file for tests that need an image on disk.

    No external executables are required.  The content is not a valid image,
    but every test that calls this helper mocks ``subprocess.run`` (or
    similar) before actually processing the file.

    Args:
        path: The file path where the image will be saved.
        size: The size of the image in ImageMagick format (e.g., "100x100").
            Accepted for API compatibility but not used.
        colorspace: The colorspace to use (e.g., "sRGB"). Not used.
        background: The background color/type (e.g., "none" for transparency). Not used.
        draw: ImageMagick draw commands to execute. Not used.
        metadata: Metadata to add to the image (e.g., "Artist=Test Artist"). Not used.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


def create_markdown_file(
    path: Path,
    frontmatter: dict[str, Any] | None = None,
    content: str = "# Test",
) -> Path:
    """Create a markdown file with YAML front-matter.

    Args:
        path: Destination *Path*.
        frontmatter: Mapping to serialise as YAML front-matter. If *None*, no
            front-matter is written.
        content: Markdown body to append after the front-matter.
    """
    if frontmatter is not None:
        # Use ruamel.yaml for compatibility with TimeStamp objects
        yaml_parser = YAML(typ="rt")
        yaml_parser.preserve_quotes = True

        from io import StringIO

        stream = StringIO()
        yaml_parser.dump(frontmatter, stream)
        yaml_text = stream.getvalue().strip()

        md_text = f"---\n{yaml_text}\n---\n{content}"
    else:
        md_text = content
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md_text, encoding="utf-8")
    return path
