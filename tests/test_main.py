"""Tests for the CLI entry point: dispatch, parsing, completion, key gating."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import alt_text_llm
from alt_text_llm import generate, main, openrouter, scan


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def test_build_parser_subcommands() -> None:
    parser = main._build_parser()
    assert isinstance(parser, argparse.ArgumentParser)

    args = parser.parse_args(["generate", "--model", "google/gemini-2.5-flash"])
    assert args.command == "generate"
    assert args.model == "google/gemini-2.5-flash"


def test_generate_model_argument_has_completer() -> None:
    """The --model action must expose the OpenRouter completer for argcomplete."""
    parser = main._build_parser()
    generate_action = None
    for action in parser._subparsers._group_actions:  # type: ignore[attr-defined]
        choices = getattr(action, "choices", {})
        if "generate" in choices:
            generate_action = choices["generate"]
            break
    assert generate_action is not None

    model_action = next(
        a for a in generate_action._actions if a.dest == "model"
    )
    assert getattr(model_action, "completer", None) is main._model_completer


# ---------------------------------------------------------------------------
# Model completer
# ---------------------------------------------------------------------------


def test_model_completer_filters_by_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        openrouter,
        "list_model_ids",
        lambda: [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-sonnet-4.5",
        ],
    )

    assert main._model_completer("google/") == [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
    ]
    assert main._model_completer("anthropic/") == ["anthropic/claude-sonnet-4.5"]
    assert main._model_completer("") == [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4.5",
    ]


def test_model_completer_offline_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(openrouter, "list_model_ids", lambda: [])
    assert main._model_completer("google/") == []


def test_argcomplete_end_to_end(tmp_path: Path) -> None:
    """Drive argcomplete's real protocol: `--model google/<TAB>` completes ids.

    This exercises the full path: the ``# PYTHON_ARGCOMPLETE_OK`` marker,
    ``argcomplete.autocomplete(parser)``, and the ``--model`` completer.
    """
    pytest.importorskip("argcomplete")

    outfile = tmp_path / "completions"
    line = "alt-text-llm generate --model google/"
    env = {
        **os.environ,
        "_ARGCOMPLETE": "1",
        "_ARGCOMPLETE_SHELL": "bash",
        "COMP_LINE": line,
        "COMP_POINT": str(len(line)),
        "_ARGCOMPLETE_COMP_WORDBREAKS": " \t\n\"'><=;|&(:",
        "_ARGCOMPLETE_STDOUT_FILENAME": str(outfile),
    }
    # Stub the catalogue so completion needs no network.
    shim = (
        "from unittest.mock import patch\n"
        "from alt_text_llm import openrouter, main\n"
        "ids = ['google/gemini-2.5-flash', 'google/gemini-2.5-pro',"
        " 'anthropic/claude-x']\n"
        "with patch.object(openrouter, 'list_model_ids', lambda: ids):\n"
        "    main.main()\n"
    )
    subprocess.run(
        [sys.executable, "-c", shim], env=env, check=False, timeout=60
    )

    completions = outfile.read_text().split("\x0b")
    assert "google/gemini-2.5-flash" in completions
    assert "google/gemini-2.5-pro" in completions
    assert "anthropic/claude-x" not in completions  # prefix filtered


# ---------------------------------------------------------------------------
# generate command: API-key gating
# ---------------------------------------------------------------------------


def _generate_args(tmp: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "root": tmp,
        "model": "google/gemini-2.5-flash",
        "max_chars": 300,
        "timeout": 120,
        "captions": tmp / "asset_captions.json",
        "suggestions_file": tmp / "suggested_alts.json",
        "skip_existing": True,
        "estimate_only": False,
        "yes": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_generate_command_aborts_without_api_key(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation must fail fast (and not scan) when the key is missing."""
    monkeypatch.delenv(openrouter.API_KEY_ENV_VAR, raising=False)

    build_queue_called = False

    def tracking_build_queue(root):
        nonlocal build_queue_called
        build_queue_called = True
        return []

    monkeypatch.setattr(scan, "build_queue", tracking_build_queue)

    printed: list[str] = []
    console = Mock()
    console.print = lambda msg, *a, **k: printed.append(str(msg))
    monkeypatch.setattr(main, "Console", lambda: console)

    main._generate_command(_generate_args(temp_dir, estimate_only=False))

    assert not build_queue_called  # aborted before scanning
    assert any(openrouter.API_KEY_ENV_VAR in line for line in printed)


def test_estimate_only_does_not_require_api_key(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--estimate-only must work with no key (public pricing catalogue)."""
    monkeypatch.delenv(openrouter.API_KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(scan, "build_queue", lambda root: [])
    monkeypatch.setattr(
        generate, "estimate_cost", lambda model, n: "Estimated cost: $0.000"
    )

    key_checked = False

    def tracking_get_api_key():
        nonlocal key_checked
        key_checked = True
        raise openrouter.OpenRouterError("should not be called")

    monkeypatch.setattr(openrouter, "get_api_key", tracking_get_api_key)
    monkeypatch.setattr(main, "Console", lambda: Mock())

    # Should not raise and should not consult the API key.
    main._generate_command(_generate_args(temp_dir, estimate_only=True))
    assert key_checked is False


def test_generate_command_warns_on_unknown_model(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model id absent from the catalogue triggers a warning but proceeds."""
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    monkeypatch.setattr(
        scan,
        "build_queue",
        lambda root: [
            scan.QueueItem(
                markdown_file="t.md",
                asset_path="i.jpg",
                line_number=1,
                context_snippet="c",
            )
        ],
    )
    monkeypatch.setattr(generate, "estimate_cost", lambda m, n: "Estimated cost: $0")
    monkeypatch.setattr(
        openrouter, "list_model_ids", lambda: ["google/gemini-2.5-flash"]
    )

    # Stop before real generation (must be a coroutine for asyncio.run).
    async def fake_generate(*a, **k):
        return []

    monkeypatch.setattr(generate, "async_generate_suggestions", fake_generate)

    printed: list[str] = []
    console = Mock()
    console.print = lambda msg, *a, **k: printed.append(str(msg))
    monkeypatch.setattr(main, "Console", lambda: console)

    args = _generate_args(temp_dir, model="gemini-3.0-flash", skip_existing=False)
    main._generate_command(args)

    assert any("not a known OpenRouter model id" in line for line in printed)


# ---------------------------------------------------------------------------
# Subcommand dispatch (integration via main())
# ---------------------------------------------------------------------------


def _run_main(argv: list[str]) -> None:
    """Run main() with a patched sys.argv."""
    with patch("sys.argv", ["alt-text-llm", *argv]):
        main.main()


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
# generate: estimate-only and confirmation (integration via main())
# ---------------------------------------------------------------------------


def test_generate_estimate_only_unknown_model(
    temp_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate --estimate-only with an unknown model reports unavailable cost."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")

    # Unknown model -> no live pricing; stubbed to avoid a network call.
    monkeypatch.setattr(main.openrouter, "get_pricing", lambda model: None)

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


def test_generate_with_yes_skips_confirmation(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """generate --yes proceeds without prompting and writes suggestions."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")
    suggestions_file = temp_dir / "sugg.json"

    # Provide a key and stub the network-backed helpers.
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    monkeypatch.setattr(
        main.generate, "estimate_cost", lambda model, n: "Estimated cost: $0.00"
    )
    monkeypatch.setattr(main.openrouter, "list_model_ids", lambda: [])

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
                "google/gemini-2.5-flash",
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
    temp_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interactive 'n' answer aborts generation."""
    md = temp_dir / "post.md"
    md.write_text("![](photo.png)\n", encoding="utf-8")
    suggestions_file = temp_dir / "sugg.json"

    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    monkeypatch.setattr(
        main.generate, "estimate_cost", lambda model, n: "Estimated cost: $0.00"
    )
    monkeypatch.setattr(main.openrouter, "list_model_ids", lambda: [])

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
                "google/gemini-2.5-flash",
                "--captions",
                str(temp_dir / "captions.json"),
                "--suggestions-file",
                str(suggestions_file),
            ]
        )

    mock_gen.assert_not_called()
    assert "Aborted" in capsys.readouterr().out
    assert not suggestions_file.exists()
