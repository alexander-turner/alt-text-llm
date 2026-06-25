"""Tests for the CLI entry point: argument parsing, completion, key gating."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

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
