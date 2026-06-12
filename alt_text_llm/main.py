"""Main entry point for alt text generation and labeling workflows."""

import argparse
import asyncio
import json
from enum import StrEnum
from pathlib import Path

from rich.console import Console

from alt_text_llm import apply, generate, label, scan, utils

_JSON_INDENT: int = 2


class Command(StrEnum):
    """Available commands for alt text workflows."""

    SCAN = "scan"
    GENERATE = "generate"
    LABEL = "label"
    APPLY = "apply"


def _scan_command(args: argparse.Namespace) -> None:
    """Execute the scan sub-command."""
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for all alt text workflows."""
    parser = argparse.ArgumentParser(
        description="Alt text generation and labeling workflows"
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
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
    generate_parser.add_argument(
        "--model", required=True, help="LLM model to use for generation"
    )
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
            args.skip_existing,
            args.vi_mode,
        )
    elif args.command == Command.APPLY:
        apply.apply_from_captions_file(args.captions_file, args.dry_run)
    else:
        raise ValueError(f"Invalid command: {args.command}")


if __name__ == "__main__":
    main()
