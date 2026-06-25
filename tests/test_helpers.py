"""Helper functions for test setup."""

import struct
import zlib
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def write_real_png(path: Path, *, color: tuple[int, int, int] = (255, 0, 0)) -> Path:
    """Write a REAL, decodable 1x1 PNG to *path*.

    Builds valid PNG bytes from scratch using ``zlib``/``struct`` so the file
    needs no external tools and is a genuine image (unlike ``create_test_image``
    which writes a dummy non-image blob).

    Args:
        path: Destination path for the PNG file.
        color: RGB tuple for the single pixel.

    Returns:
        The *path* that was written.
    """

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    # 1x1, 8-bit depth, color type 2 (truecolor RGB)
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    # One scanline: filter byte 0 followed by the RGB pixel
    raw = bytes([0, color[0], color[1], color[2]])
    idat = zlib.compress(raw)

    png_bytes = (
        signature
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)
    return path


def write_fake_llm(bin_dir: Path, *, caption: str, fail_on: str | None = None) -> Path:
    """Write a fake ``llm`` executable (a /bin/sh script) into *bin_dir*.

    The script ignores its model/prompt args and echoes a deterministic
    *caption* so the REAL ``generate._run_llm`` subprocess path can run without
    a real LLM or network. If *fail_on* is provided, the script exits non-zero
    (and writes to stderr) whenever any argument contains that substring -- used
    to exercise per-item failure isolation.

    Args:
        bin_dir: Directory to place the ``llm`` script in (created if needed).
        caption: The caption the fake binary echoes on success.
        fail_on: If set, the binary fails when any arg contains this substring.

    Returns:
        Path to the created executable.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    llm_path = bin_dir / "llm"

    if fail_on:
        # Only inspect the attachment passed via ``-a`` so the sentinel must
        # appear in the asset path itself, not merely in the prompt context.
        script = (
            "#!/bin/sh\n"
            "attachment=\n"
            'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "-a" ]; then\n'
            '    attachment="$2"\n'
            "    break\n"
            "  fi\n"
            "  shift\n"
            "done\n"
            f'case "$attachment" in\n'
            f'  *{fail_on}*) echo "fake llm: simulated failure" 1>&2; exit 1;;\n'
            "esac\n"
            f'echo "{caption}"\n'
        )
    else:
        script = "#!/bin/sh\n" f'echo "{caption}"\n'

    llm_path.write_text(script, encoding="utf-8")
    llm_path.chmod(0o755)
    return llm_path


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
