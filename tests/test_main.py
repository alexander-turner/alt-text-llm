"""Tests for main.py CLI dispatch, argument handling, and UX guards."""

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

import alt_text_llm
from alt_text_llm import main


def _run_main(argv: list[str]) -> None:
    """Run main() with a patched sys.argv."""
    with patch("sys.argv", ["alt-text-llm", *argv]):
        main.main()


# ---------------------------------------------------------------------------
# Subcommand dispatch
# ---------------------------------------------------------------------------


def test_scan_command_writes_queue_and_prints_count(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """scan writes a JSON queue and prints the item count."""
    md = temp_dir / "post.md"
    md.write_text(
        "# Title\n\nSome text.\n\n![](photo.png)\n", encoding="utf-8"
    )
    output = temp_dir / "queue.json"

    _run_main(["scan", "--root", str(temp_dir), "--output", str(output)])

    captured = capsys.readouterr().out
    assert output.exists()
    queue = json.loads(output.read_text(encoding="utf-8"))
    assert len(queue) == 1
    assert queue[0]["asset_path"] == "photo.png"
    assert f"Wrote {len(queue)} queue item(s)" in captured


def test_apply_command_mutates_markdown(temp_dir: Path) -> None:
    """apply rewrites the markdown image alt text in place."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")

    captions = temp_dir / "captions.json"
    captions.write_text(
        json.dumps(
            [
                {
                    "markdown_file": str(md),
                    "asset_path": "photo.png",
                    "suggested_alt": "A photo",
                    "final_alt": "A descriptive photo",
                    "model": "test-model",
                    "context_snippet": "ctx",
                    "line_number": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    _run_main(["apply", "--captions-file", str(captions)])

    assert "![A descriptive photo](photo.png)" in md.read_text(encoding="utf-8")


def test_apply_dry_run_does_not_mutate(temp_dir: Path) -> None:
    """apply --dry-run leaves the markdown file untouched."""
    md = temp_dir / "post.md"
    original = "![](photo.png)\n"
    md.write_text(original, encoding="utf-8")

    captions = temp_dir / "captions.json"
    captions.write_text(
        json.dumps(
            [
                {
                    "markdown_file": str(md),
                    "asset_path": "photo.png",
                    "suggested_alt": "A photo",
                    "final_alt": "A descriptive photo",
                    "model": "test-model",
                    "context_snippet": "ctx",
                    "line_number": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    _run_main(["apply", "--captions-file", str(captions), "--dry-run"])

    assert md.read_text(encoding="utf-8") == original


def test_label_command_dispatch(temp_dir: Path) -> None:
    """label dispatches to label_from_suggestions_file with parsed args."""
    suggestions = temp_dir / "sugg.json"
    suggestions.write_text("[]", encoding="utf-8")
    output = temp_dir / "out.json"

    with patch.object(main.label, "label_from_suggestions_file") as mock_label:
        _run_main(
            [
                "label",
                "--suggestions-file",
                str(suggestions),
                "--output",
                str(output),
                "--no-skip-existing",
            ]
        )

    mock_label.assert_called_once()
    kwargs = mock_label.call_args.kwargs
    assert kwargs["skip_existing"] is False


# ---------------------------------------------------------------------------
# CLI UX: no subcommand, --version
# ---------------------------------------------------------------------------


def test_no_subcommand_exits_with_error() -> None:
    """Bare invocation exits via SystemExit (argparse, code 2)."""
    with pytest.raises(SystemExit) as exc_info:
        _run_main([])
    assert exc_info.value.code == 2


def test_version_flag_prints_version_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--version prints the package version and exits 0."""
    with pytest.raises(SystemExit) as exc_info:
        _run_main(["--version"])
    assert exc_info.value.code == 0
    assert alt_text_llm.__version__ in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Friendly errors / exit codes
# ---------------------------------------------------------------------------


def test_scan_missing_root_exits_nonzero(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """scan with a non-existent --root exits non-zero with a friendly error."""
    missing = temp_dir / "does-not-exist"
    with pytest.raises(SystemExit) as exc_info:
        _run_main(["scan", "--root", str(missing)])
    assert exc_info.value.code != 0
    assert "Root directory not found" in capsys.readouterr().out


def test_label_missing_file_exits_nonzero(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """label with a missing suggestions file exits non-zero."""
    missing = temp_dir / "missing.json"
    with pytest.raises(SystemExit) as exc_info:
        _run_main(["label", "--suggestions-file", str(missing)])
    assert exc_info.value.code != 0
    assert "not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# generate: estimate-only and confirmation
# ---------------------------------------------------------------------------


def test_generate_estimate_only_unknown_model(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """generate --estimate-only with an unknown model reports unavailable cost."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")

    with patch.object(main.generate, "async_generate_suggestions") as mock_gen:
        _run_main(
            [
                "generate",
                "--root",
                str(temp_dir),
                "--model",
                "totally-unknown-model",
                "--captions",
                str(temp_dir / "captions.json"),
                "--suggestions-file",
                str(temp_dir / "sugg.json"),
                "--estimate-only",
            ]
        )

    assert "Cost estimation not available" in capsys.readouterr().out
    mock_gen.assert_not_called()


def test_generate_with_yes_skips_confirmation(temp_dir: Path) -> None:
    """generate --yes proceeds without prompting and writes suggestions."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")
    suggestions_file = temp_dir / "sugg.json"

    async def fake_generate(queue_items, opts):
        return [
            main.utils.AltGenerationResult(
                markdown_file=str(md),
                asset_path="photo.png",
                suggested_alt="generated alt",
                model=opts.model,
                context_snippet="ctx",
                line_number=1,
            )
        ]

    with (
        patch.object(
            main.generate,
            "async_generate_suggestions",
            side_effect=fake_generate,
        ),
        patch("builtins.input") as mock_input,
    ):
        _run_main(
            [
                "generate",
                "--root",
                str(temp_dir),
                "--model",
                "gemini-2.5-flash",
                "--captions",
                str(temp_dir / "captions.json"),
                "--suggestions-file",
                str(suggestions_file),
                "--yes",
            ]
        )

    mock_input.assert_not_called()
    saved = json.loads(suggestions_file.read_text(encoding="utf-8"))
    assert saved[0]["suggested_alt"] == "generated alt"


def test_generate_declined_confirmation_does_not_generate(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interactive 'n' answer aborts generation."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")
    suggestions_file = temp_dir / "sugg.json"

    with (
        patch("sys.stdin.isatty", return_value=True),
        patch("sys.stdout.isatty", return_value=True),
        patch("builtins.input", return_value="n"),
        patch.object(
            main.generate, "async_generate_suggestions"
        ) as mock_gen,
    ):
        _run_main(
            [
                "generate",
                "--root",
                str(temp_dir),
                "--model",
                "gemini-2.5-flash",
                "--captions",
                str(temp_dir / "captions.json"),
                "--suggestions-file",
                str(suggestions_file),
            ]
        )

    mock_gen.assert_not_called()
    assert "Aborted" in capsys.readouterr().out
    assert not suggestions_file.exists()
