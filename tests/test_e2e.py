"""Real, end-to-end tests for the alt-text-llm pipeline.

These tests exercise the REAL pipeline -- scan.build_queue ->
generate.async_generate_suggestions -> utils.write_output -> apply.apply_captions
-- with no stubbing of production code beyond the single OpenRouter network call,
and no real LLM.

Hermeticity / determinism:
- ``openrouter.generate_caption`` is stubbed (see the ``fake_llm_on_path``
  fixture in conftest.py) so the real ``generate._run_llm`` ->
  ``_run_llm_async`` -> ``async_generate_suggestions`` code path runs without
  network or a real model.
- Assets are real, decodable 1x1 PNGs on disk, so ``download_asset`` resolves
  them locally with no network and no imagemagick/ffmpeg conversion.
- All paths live under pytest's per-test ``tmp_path``, so the tests are safe
  under ``-n auto`` (xdist).
"""

import asyncio
from pathlib import Path

import pytest
from rich.console import Console

from alt_text_llm import apply, generate, openrouter, scan, utils

from tests.conftest import FAKE_LLM_CAPTION
from tests.test_helpers import write_real_png


def _make_options(root: Path, output_path: Path) -> generate.GenerateAltTextOptions:
    return generate.GenerateAltTextOptions(
        root=root,
        model="fake-model",
        max_chars=125,
        timeout=30,
        output_path=output_path,
    )


def _run_full_pipeline(
    root: Path,
    md_path: Path,
    *,
    dry_run: bool = False,
) -> int:
    """Run scan -> generate -> write_output -> apply against *root*.

    Returns the number of captions applied. Assumes a fake ``llm`` is already
    on PATH (caller installs the ``fake_llm_on_path`` fixture).
    """
    queue = scan.build_queue(root)
    assert queue, "expected at least one queued asset"

    output_path = root / "asset_captions.json"
    options = _make_options(root, output_path)

    suggestions = asyncio.run(
        generate.async_generate_suggestions(queue, options)
    )
    assert suggestions, "expected at least one suggestion"

    # Simulate the human/label step accepting each suggestion verbatim.
    for suggestion in suggestions:
        suggestion.final_alt = suggestion.suggested_alt

    utils.write_output(suggestions, output_path)

    return apply.apply_captions(output_path, Console(), dry_run=dry_run)


def test_full_roundtrip_markdown_image(tmp_path: Path, fake_llm_on_path: str):
    """Full real pipeline for a standard markdown ``![](photo.png)`` image."""
    root = tmp_path / "project"
    root.mkdir()
    md_path = root / "post.md"
    md_path.write_text(
        "# Title\n\nSome intro text.\n\n![](photo.png)\n\nTrailing text.\n",
        encoding="utf-8",
    )
    write_real_png(root / "photo.png")

    # --- scan ---
    queue = scan.build_queue(root)
    assert len(queue) == 1
    item = queue[0]
    assert item.asset_path == "photo.png"
    assert item.line_number == 5  # 1-based line of the image
    assert str(md_path) == item.markdown_file

    # --- generate (real subprocess against fake llm) ---
    output_path = root / "asset_captions.json"
    options = _make_options(root, output_path)
    suggestions = asyncio.run(
        generate.async_generate_suggestions(queue, options)
    )
    assert len(suggestions) == 1
    assert suggestions[0].suggested_alt == fake_llm_on_path == FAKE_LLM_CAPTION

    # --- label (accept) + write_output ---
    suggestions[0].final_alt = suggestions[0].suggested_alt
    utils.write_output(suggestions, output_path)
    assert output_path.exists()

    # --- apply ---
    applied = apply.apply_captions(output_path, Console(), dry_run=False)
    assert applied == 1

    final_text = md_path.read_text(encoding="utf-8")
    assert f"![{FAKE_LLM_CAPTION}](photo.png)" in final_text

    # A fresh scan should now find nothing -- the asset has real alt text.
    assert scan.build_queue(root) == []


@pytest.mark.parametrize(
    "image_markup,expected_substring",
    [
        (
            '<img src="photo.png">',
            f'<img alt="{FAKE_LLM_CAPTION}" src="photo.png"/>',
        ),
        (
            "![[photo.png]]",
            f"![[photo.png|{FAKE_LLM_CAPTION}]]",
        ),
    ],
    ids=["html_img", "wikilink"],
)
def test_roundtrip_format_variants(
    tmp_path: Path,
    fake_llm_on_path: str,
    image_markup: str,
    expected_substring: str,
):
    """Real roundtrip for HTML ``<img>`` and Obsidian wikilink formats."""
    root = tmp_path / "project"
    root.mkdir()
    md_path = root / "post.md"
    md_path.write_text(
        f"# Title\n\nIntro.\n\n{image_markup}\n\nOutro.\n",
        encoding="utf-8",
    )
    write_real_png(root / "photo.png")

    applied = _run_full_pipeline(root, md_path)
    assert applied == 1

    final_text = md_path.read_text(encoding="utf-8")
    assert expected_substring in final_text

    # Rescan should be clean now that meaningful alt text exists.
    assert scan.build_queue(root) == []


def test_roundtrip_with_yaml_frontmatter(tmp_path: Path, fake_llm_on_path: str):
    """Roundtrip through a file with YAML frontmatter before the image.

    Exercises the line-number/frontmatter handling end to end.
    """
    root = tmp_path / "project"
    root.mkdir()
    md_path = root / "post.md"
    md_path.write_text(
        "---\n"
        "title: My Post\n"
        "date: 2024-01-01\n"
        "tags:\n"
        "  - one\n"
        "  - two\n"
        "---\n"
        "\n"
        "# Heading\n"
        "\n"
        "Body paragraph before the image.\n"
        "\n"
        "![](photo.png)\n"
        "\n"
        "Body paragraph after.\n",
        encoding="utf-8",
    )
    write_real_png(root / "photo.png")

    queue = scan.build_queue(root)
    assert len(queue) == 1
    # Image is on line 13 (1-based), after 7 frontmatter lines.
    assert queue[0].line_number == 13

    applied = _run_full_pipeline(root, md_path)
    assert applied == 1

    final_text = md_path.read_text(encoding="utf-8")
    assert f"![{FAKE_LLM_CAPTION}](photo.png)" in final_text
    # Frontmatter must be preserved intact.
    assert final_text.startswith("---\ntitle: My Post\n")
    assert scan.build_queue(root) == []


def test_dry_run_does_not_mutate(tmp_path: Path, fake_llm_on_path: str):
    """apply(dry_run=True) must leave the markdown byte-for-byte unchanged."""
    root = tmp_path / "project"
    root.mkdir()
    md_path = root / "post.md"
    original_bytes = (
        b"# Title\n\nIntro.\n\n![](photo.png)\n\nOutro.\n"
    )
    md_path.write_bytes(original_bytes)
    write_real_png(root / "photo.png")

    applied = _run_full_pipeline(root, md_path, dry_run=True)
    # apply reports it *would* apply, but the file is untouched.
    assert applied == 1
    assert md_path.read_bytes() == original_bytes

    # Since nothing was written, the asset still needs alt text.
    assert len(scan.build_queue(root)) == 1


def test_per_item_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """One asset failing generation must not abort the batch.

    Stubs ``openrouter.generate_caption`` to raise for any attachment whose
    filename contains the sentinel substring ``bad``. The good asset must still
    produce a suggestion.
    """
    root = tmp_path / "project"
    root.mkdir()

    md_path = root / "post.md"
    md_path.write_text(
        "# Title\n\nIntro.\n\n![](good.png)\n\nMiddle.\n\n![](bad.png)\n\nEnd.\n",
        encoding="utf-8",
    )
    write_real_png(root / "good.png")
    write_real_png(root / "bad.png")

    def flaky_generate_caption(attachment, prompt, model, timeout):
        if "bad" in Path(attachment).name:
            raise openrouter.OpenRouterError("simulated failure")
        return FAKE_LLM_CAPTION, {"cost": 0.0001}

    monkeypatch.setattr(openrouter, "generate_caption", flaky_generate_caption)

    queue = scan.build_queue(root)
    assert len(queue) == 2

    output_path = root / "asset_captions.json"
    options = _make_options(root, output_path)
    suggestions = asyncio.run(
        generate.async_generate_suggestions(queue, options)
    )

    # The bad asset is skipped; the good one still succeeds.
    assert len(suggestions) == 1
    assert suggestions[0].asset_path == "good.png"
    assert suggestions[0].suggested_alt == FAKE_LLM_CAPTION
