"""Main entry point for alt text generation and labeling workflows."""

# PYTHON_ARGCOMPLETE_OK

import argparse
import asyncio
import json
import sys
from enum import StrEnum
from pathlib import Path

from rich.console import Console

import alt_text_llm
from alt_text_llm import apply, generate, label, openrouter, scan, utils

_JSON_INDENT: int = 2

# Default OpenRouter model for `generate`. A cheap, video-capable vision model
# (~4x cheaper than gemini-2.5-pro) that handles both images and videos (only
# Google Gemini models can caption videos on OpenRouter). Override with
# --model — e.g. 'google/gemini-2.5-flash-lite' (cheapest) or
# 'google/gemini-2.5-pro' (highest quality). Model ids must be the full
# 'provider/slug' form; a bare slug yields "Unknown model" from OpenRouter.
_DEFAULT_MODEL: str = "google/gemini-2.5-flash"


class Command(StrEnum):
    """Available commands for alt text workflows."""

    SCAN = "scan"
    GENERATE = "generate"
    LABEL = "label"
    APPLY = "apply"


def _validate_root(root: Path, console: Console) -> None:
    """Exit non-zero with a friendly message if *root* is missing."""
    if not root.exists():
        console.print(f"[red]Error: Root directory not found: {root}[/red]")
        raise SystemExit(1)


def _scan_command(args: argparse.Namespace) -> None:
    """Execute the scan sub-command."""
    console = Console()
    _validate_root(args.root, console)
    output_path = args.output
    queue_items = scan.build_queue(args.root)

    output_path.write_text(
        json.dumps(
            [item.to_json() for item in queue_items],
            indent=_JSON_INDENT,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(queue_items)} queue item(s) to {output_path}")


def _suggest_model_id(model: str, known: list[str]) -> str | None:
    """Suggest a catalogue id for a likely-mistyped *model*.

    The common mistake is a bare slug missing its ``provider/`` prefix
    (e.g. ``gemini-2.5-flash`` instead of ``google/gemini-2.5-flash``), which
    OpenRouter rejects at generation time with "Unknown model". Returns the
    prefixed id when one exists, else ``None``.
    """
    if "/" in model:
        return None
    prefixed = f"google/{model}"
    if prefixed in known:
        return prefixed
    matches = [m for m in known if m.split("/", 1)[-1] == model]
    return matches[0] if matches else None


def _model_is_usable(model: str, console: Console) -> bool:
    """Return whether *model* should be used, aborting early on a bad id.

    Validates against OpenRouter's live catalogue so an invalid id fails fast
    with a fix-it hint, instead of letting every queue item error one-by-one
    mid-run. When the catalogue is unreachable (empty list), validation is
    skipped and generation proceeds.
    """
    known = openrouter.list_model_ids()
    if not known or model in known:
        return True
    suggestion = _suggest_model_id(model, known)
    if suggestion:
        hint = f" Did you mean '{suggestion}'?"
    else:
        hint = (
            " Model ids look like 'google/gemini-2.5-flash'; browse "
            "https://openrouter.ai/models or tab-complete --model."
        )
    console.print(
        f"[red]'{model}' is not a valid OpenRouter model id.{hint}[/red]"
    )
    return False


def _generate_command(args: argparse.Namespace) -> None:
    """Execute the generate sub-command."""
    opts = generate.GenerateAltTextOptions(
        root=args.root,
        model=args.model,
        max_chars=args.max_chars,
        timeout=args.timeout,
        output_path=args.captions,
        skip_existing=args.skip_existing,
    )

    suggestions_path = args.suggestions_file
    console = Console()
    _validate_root(opts.root, console)

    # Fail fast (before scanning/generating) if the key is missing or the model
    # id is invalid, unless we are only estimating cost — the public model
    # catalogue needs no key, and estimate-only already reports unpriceable
    # models via the cost line. Validating the model here means a bad id (e.g.
    # a bare slug missing its `provider/` prefix) aborts immediately with a
    # suggestion instead of failing every queue item one-by-one mid-run.
    if not args.estimate_only:
        try:
            openrouter.get_api_key()
        except openrouter.OpenRouterError as err:
            console.print(f"[red]{err}[/red]")
            return
        if not _model_is_usable(opts.model, console):
            return

    with console.status("Scanning markdown files for assets…"):
        queue_items = scan.build_queue(opts.root)

    if opts.skip_existing:
        queue_items = generate.filter_existing_captions(
            queue_items,
            [opts.output_path, suggestions_path],
            console,
            verbose=False if args.estimate_only else True,
        )

    # Show cost estimate
    cost_est = generate.estimate_cost(opts.model, len(queue_items))
    console.print(
        f"[bold blue]{len(queue_items)} items → {cost_est} using model '{opts.model}'[/bold blue]"
    )

    # If estimate-only mode, exit here
    if args.estimate_only:
        return

    # Run generation
    if not queue_items:
        console.print("[yellow]No items to process.[/yellow]")
        return

    # Confirm before spending money, unless explicitly skipped or running
    # in a non-interactive context (automation/tests).
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if not args.yes and interactive:
        answer = input(
            f"Generate {len(queue_items)} suggestions with '{opts.model}'? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            console.print("[yellow]Aborted; no suggestions generated.[/yellow]")
            return

    console.print(
        f"[bold green]Generating {len(queue_items)} suggestions with '{opts.model}'[/bold green]"
    )

    suggestions = []
    try:
        suggestions = asyncio.run(
            generate.async_generate_suggestions(queue_items, opts)
        )
    finally:
        utils.write_output(suggestions, suggestions_path, append_mode=True)
        console.print(
            f"[green]Saved {len(suggestions)} suggestions to {suggestions_path}[/green]"
        )


def _model_completer(prefix: str, **_kwargs: object) -> list[str]:
    """Shell-completion callback: suggest OpenRouter model ids matching *prefix*."""
    return [
        model_id
        for model_id in openrouter.list_model_ids()
        if model_id.startswith(prefix)
    ]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for all alt text workflows."""
    parser = argparse.ArgumentParser(
        description="Alt text generation and labeling workflows"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {alt_text_llm.__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # ---------------------------------------------------------------------------
    # scan sub-command
    # ---------------------------------------------------------------------------
    scan_parser = subparsers.add_parser(
        Command.SCAN,
        help="Scan markdown files for assets without meaningful alt text",
    )
    scan_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory to search (default: current directory)",
    )
    scan_parser.add_argument(
        "--output",
        type=Path,
        default=Path("asset_queue.json"),
        help="Path for output JSON file",
    )

    # ---------------------------------------------------------------------------
    # generate sub-command
    # ---------------------------------------------------------------------------
    generate_parser = subparsers.add_parser(
        Command.GENERATE, help="Generate AI alt text suggestions"
    )
    generate_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Markdown root directory (default: current directory)",
    )
    model_arg = generate_parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"OpenRouter model id, full 'provider/slug' form "
        f"(default: '{_DEFAULT_MODEL}'). Only Gemini models can caption videos.",
    )
    # Enables `--model <TAB>` to complete live OpenRouter model ids.
    model_arg.completer = _model_completer  # type: ignore[attr-defined]
    generate_parser.add_argument(
        "--max-chars",
        type=int,
        default=300,
        help="Max characters for generated alt text",
    )
    generate_parser.add_argument(
        "--timeout", type=int, default=120, help="LLM command timeout seconds"
    )
    generate_parser.add_argument(
        "--captions",
        type=Path,
        default=Path("asset_captions.json"),
        help="Existing/final captions JSON path (used to skip existing unless --process-existing)",
    )
    generate_parser.add_argument(
        "--suggestions-file",
        type=Path,
        default=Path("suggested_alts.json"),
        help="Path to read/write suggestions JSON",
    )
    generate_parser.add_argument(
        "--process-existing",
        dest="skip_existing",
        action="store_false",
        help="Also process assets that already have captions (default is to skip)",
    )
    generate_parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate cost without generating suggestions",
    )
    generate_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the cost-confirmation prompt before generating",
    )
    generate_parser.set_defaults(skip_existing=True)

    # ---------------------------------------------------------------------------
    # label sub-command
    # ---------------------------------------------------------------------------
    label_parser = subparsers.add_parser(
        Command.LABEL, help="Interactively label alt text suggestions"
    )
    label_parser.add_argument(
        "--suggestions-file",
        type=Path,
        default=Path("suggested_alts.json"),
        help="Path to read suggestions JSON",
    )
    label_parser.add_argument(
        "--output",
        type=Path,
        default=Path("asset_captions.json"),
        help="Final captions JSON path",
    )
    label_parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip captions already present in output file",
    )
    label_parser.add_argument(
        "--vi-mode",
        action="store_true",
        default=False,
        help="Enable vi keybindings for text editing (default: disabled)",
    )

    # ---------------------------------------------------------------------------
    # apply sub-command
    # ---------------------------------------------------------------------------
    apply_parser = subparsers.add_parser(
        Command.APPLY, help="Apply labeled captions to markdown files"
    )
    apply_parser.add_argument(
        "--captions-file",
        type=Path,
        default=Path("asset_captions.json"),
        help="Path to the captions JSON file with final_alt populated",
    )
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be changed without modifying files",
    )

    return parser


def _parse_args() -> argparse.Namespace:
    """Build the parser, enable shell completion, and parse arguments."""
    parser = _build_parser()

    # Activate argcomplete if installed. Completion runs only when the shell
    # invokes us in completion mode, so this is a no-op for normal runs.
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    return parser.parse_args()


def main() -> None:
    """Main entry point for alt text workflows."""
    args = _parse_args()

    if args.command == Command.SCAN:
        _scan_command(args)
    elif args.command == Command.GENERATE:
        _generate_command(args)
    elif args.command == Command.LABEL:
        label.label_from_suggestions_file(
            args.suggestions_file,
            args.output,
            skip_existing=args.skip_existing,
            vi_mode=args.vi_mode,
        )
    elif args.command == Command.APPLY:
        apply.apply_from_captions_file(args.captions_file, args.dry_run)
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == "__main__":
    main()
