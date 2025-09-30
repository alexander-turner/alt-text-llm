"""Helper functions for test setup."""

import subprocess
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from alt_text_llm import utils


def create_test_image(
    path: Path,
    size: str,
    *,
    colorspace: str | None = None,
    background: str | None = None,
    draw: str | None = None,
    metadata: str | None = None,
) -> None:
    """
    Creates a test image using ImageMagick.

    Args:
        path (Path): The file path where the image will be saved.
        size (str): The size of the image in ImageMagick format (e.g., "100x100").
        colorspace (str, optional): The colorspace to use (e.g., "sRGB").
        background (str, optional): The background color/type (e.g., "none" for transparency).
        draw (str, optional): ImageMagick draw commands to execute.
        metadata (str, optional): Metadata to add to the image (e.g., "Artist=Test Artist").

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the ImageMagick command fails.
    """
    magick_executable = utils.find_executable("magick")
    command = [magick_executable, "-size", size]

    if background:
        command.extend(["xc:" + background])
    else:
        command.extend(["xc:red"])

    if colorspace:
        command.extend(["-colorspace", colorspace])

    if draw:
        command.extend(["-draw", draw])

    if metadata:
        command.extend(["-set", metadata])

    command.append(str(path))

    subprocess.run(command, check=True)


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
