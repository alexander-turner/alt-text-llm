"""Interactive labeling interface for alt text suggestions."""

import json
import os
import platform
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Optional, Sequence
import sys

import requests
from prompt_toolkit import prompt
from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from alt_text_llm import generate, scan, utils

UNDO_REQUESTED = "UNDO_REQUESTED"
QUIT_REQUESTED = "QUIT_REQUESTED"
SKIP_REQUESTED = "SKIP_REQUESTED"

# Callback used to report preparation progress (e.g. to a spinner).
ProgressCallback = Callable[[str], None]


@dataclass(slots=True)
class DisplayPlan:
    """How to display a prepared asset, with all heavy work already done.

    ``imgcat_path`` is the file to render inline with imgcat (the image itself,
    or a generated video-preview GIF). ``external_path`` is the original video
    to open in an external player when inline display is impossible or fails.
    ``note`` is an optional message to surface to the user.
    """

    imgcat_path: Optional[Path]
    external_path: Optional[Path]
    note: Optional[str] = None


def _queue_item_for(
    suggestion: utils.AltGenerationResult,
) -> "scan.QueueItem":
    """Rebuild the scan.QueueItem a suggestion was generated from."""
    return scan.QueueItem(
        markdown_file=suggestion.markdown_file,
        asset_path=suggestion.asset_path,
        line_number=suggestion.line_number or 1,
        context_snippet=suggestion.context_snippet,
    )


def _build_video_preview(video_path: Path, workspace: Path) -> Path:
    """Render a short, downscaled animated GIF preview of *video_path*.

    Only the first few seconds are sampled so the preview stays small and
    quick to build. Raises ValueError (caught by callers) on failure.
    """
    ffmpeg = utils.find_executable("ffmpeg")
    preview = workspace / "preview.gif"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                # Input-side -t only decodes the first 5s, keeping it fast.
                "-t",
                "5",
                "-i",
                str(video_path),
                "-vf",
                "fps=10,scale=480:-2:flags=lanczos",
                "-loop",
                "0",
                str(preview),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
        raise ValueError(f"Failed to build video preview: {err}") from err
    return preview


def prepare_display(
    queue_item: "scan.QueueItem",
    workspace: Path,
    progress: Optional[ProgressCallback] = None,
) -> DisplayPlan:
    """Download *queue_item*'s asset and pre-render anything needed to show it.

    This does all the slow work (network download, ffmpeg conversion) and is
    safe to run off the main thread so the next asset can be buffered while the
    user labels the current one. Display errors are folded into the returned
    plan; only genuine acquisition failures (missing file, network) propagate.
    """
    if progress is not None:
        progress("Downloading asset…")
    asset = utils.download_asset(queue_item, workspace)

    if not utils.is_video_asset(str(asset)):
        return DisplayPlan(imgcat_path=asset, external_path=None)

    # imgcat's inline protocol can't render under tmux, so skip the (slow)
    # conversion entirely and arrange to open the video externally instead.
    if "TMUX" in os.environ:
        return DisplayPlan(
            imgcat_path=None,
            external_path=asset,
            note="Cannot preview video inline in tmux; opening externally",
        )

    try:
        if progress is not None:
            progress("Building video preview…")
        gif = _build_video_preview(asset, workspace)
    except Exception as err:  # noqa: BLE001 - inline preview is best-effort
        return DisplayPlan(
            imgcat_path=None,
            external_path=asset,
            note=f"Could not preview video inline ({err}); opening externally",
        )
    return DisplayPlan(imgcat_path=gif, external_path=asset)


class AssetPrefetcher:
    """Downloads and pre-renders upcoming assets on a background thread.

    Each suggestion index gets its own workspace subdirectory so prepared
    assets can coexist. A single worker keeps at most one download/conversion
    in flight, and a sliding window bounds disk use while still letting recent
    items be re-shown instantly after an undo.
    """

    _KEEP_BEHIND = 3

    def __init__(
        self,
        suggestions: Sequence[utils.AltGenerationResult],
        workspace_root: Path,
    ) -> None:
        self._suggestions = suggestions
        self._root = workspace_root
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: dict[int, "Future[DisplayPlan]"] = {}

    def _workspace_for(self, index: int) -> Path:
        workspace = self._root / f"item-{index}"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def schedule(self, index: int) -> None:
        """Begin preparing the asset at *index* in the background, if valid."""
        if not (0 <= index < len(self._suggestions)):
            return
        if index in self._futures:
            return
        queue_item = _queue_item_for(self._suggestions[index])
        workspace = self._workspace_for(index)
        self._futures[index] = self._executor.submit(
            prepare_display, queue_item, workspace
        )

    def get(
        self, index: int, on_status: Optional[ProgressCallback] = None
    ) -> DisplayPlan:
        """Return the prepared plan for *index*, computing it if not buffered.

        Re-raises any acquisition error stored by the background worker so the
        labeling loop's existing error handling still applies.
        """
        future = self._futures.get(index)
        if future is None:
            queue_item = _queue_item_for(self._suggestions[index])
            workspace = self._workspace_for(index)
            return prepare_display(queue_item, workspace, progress=on_status)
        if on_status is not None and not future.done():
            on_status("Preparing preview…")
        return future.result()

    def retain_window(self, index: int) -> None:
        """Drop buffered items far from *index* to bound disk usage."""
        low, high = index - self._KEEP_BEHIND, index + 1
        for stale in [i for i in self._futures if i < low or i > high]:
            self._discard(stale)

    def _discard(self, index: int) -> None:
        future = self._futures.pop(index, None)
        if future is not None:
            future.cancel()
        shutil.rmtree(self._root / f"item-{index}", ignore_errors=True)

    def shutdown(self) -> None:
        """Cancel pending work and stop the background worker."""
        self._executor.shutdown(wait=False, cancel_futures=True)


class LabelingQuit(Exception):
    """Raised internally to cleanly stop the labeling loop and save progress."""


class LabelingSession:
    """Manages the labeling session state and navigation."""

    def __init__(self, suggestions: Sequence[utils.AltGenerationResult]) -> None:
        self.suggestions: list[utils.AltGenerationResult] = list(suggestions)
        self.current_index = 0
        self.processed_results: list[utils.AltGenerationResult] = []

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return len(self.processed_results) > 0

    def undo(self) -> utils.AltGenerationResult | None:
        """Undo the last processed result and return to previous item."""
        if not self.can_undo():
            return None

        undone_result = self.processed_results.pop()
        self.current_index = max(0, self.current_index - 1)
        return undone_result

    def add_result(self, result: utils.AltGenerationResult) -> None:
        """Add a processed result and advance to next item."""
        self.processed_results.append(result)
        self.current_index += 1

    def get_current_suggestion(self) -> utils.AltGenerationResult | None:
        """Get the current suggestion to process."""
        if self.current_index >= len(self.suggestions):
            return None
        return self.suggestions[self.current_index]

    def is_complete(self) -> bool:
        """Check if all suggestions have been processed."""
        return self.current_index >= len(self.suggestions)

    def get_progress(self) -> tuple[int, int]:
        """Get current position and total count."""
        return self.current_index + 1, len(self.suggestions)

    def skip_current(self) -> None:
        """Skip the current suggestion due to error and advance index."""
        self.current_index += 1


class DisplayManager:
    """Handles rich console display operations."""

    def __init__(self, console: Console, vi_mode: bool = False) -> None:
        self.console = console
        self.vi_mode = vi_mode

    def show_context(self, queue_item: "scan.QueueItem") -> None:
        """Display context information for the queue item."""
        context = utils.generate_article_context(
            queue_item, max_before=4, max_after=1, trim_frontmatter=True
        )
        rendered_context = Markdown(context)
        basename = Path(queue_item.markdown_file).name
        self.console.print(
            Panel(
                rendered_context,
                title="Context",
                subtitle=f"{basename}:{queue_item.line_number}",
                box=ROUNDED,
            )
        )

    def render(self, plan: DisplayPlan) -> None:
        """Display a prepared asset: inline via imgcat, else open externally.

        All heavy work happened in ``prepare_display``; this only performs the
        terminal I/O. Display failures are best-effort and never raise.
        """
        if plan.note:
            self.console.print(f"[yellow]{plan.note}[/yellow]")

        if plan.imgcat_path is not None:
            try:
                self.show_image(plan.imgcat_path)
                return
            except Exception as err:  # noqa: BLE001 - display is best-effort
                if plan.external_path is None:
                    self.console.print(
                        f"[yellow]Could not display asset: {err}; "
                        "continuing[/yellow]"
                    )
                    return
                self.console.print(
                    f"[yellow]Could not preview inline ({err}); "
                    "opening externally[/yellow]"
                )

        if plan.external_path is not None:
            self._open_externally(plan.external_path)

    def show_image(self, path: Path) -> None:
        """Display the image using imgcat."""
        if "TMUX" in os.environ:
            raise ValueError("Cannot open image in tmux")
        try:
            subprocess.run(["imgcat", str(path)], check=True)
        except subprocess.CalledProcessError as err:
            raise ValueError(
                f"Failed to open image: {err}; is imgcat installed?"
            ) from err

    def _open_externally(self, path: Path) -> None:
        """Open *path* with the OS default application without stealing focus.

        Uses a non-blocking, detached launch so the labeling prompt keeps
        terminal focus. On macOS ``open -g`` opens in the background; other
        platforms have no portable background flag, but their openers return
        immediately and leave the terminal in the foreground.
        """
        opener = self._external_open_command(path)
        try:
            subprocess.Popen(
                opener,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError) as err:
            self.console.print(
                f"[yellow]Could not open video externally: {err}[/yellow]"
            )

    @staticmethod
    def _external_open_command(path: Path) -> list[str]:
        """Build the platform-specific command to open *path* externally."""
        system = platform.system()
        if system == "Darwin":
            # -g keeps the launched app in the background so the terminal
            # retains focus.
            return ["open", "-g", str(path)]
        if system == "Windows":
            return ["cmd", "/c", "start", "", str(path)]
        return ["xdg-open", str(path)]

    def show_progress(self, current: int, total: int) -> None:
        """Display progress information."""
        percent = (current / total * 100) if total else 0.0
        progress_text = f"Progress: {current}/{total} ({percent:.1f}%)"
        self.console.print(f"[dim]{progress_text}[/dim]")

    def prompt_for_edit(
        self,
        suggestion: str,
        current: int | None = None,
        total: int | None = None,
    ) -> str:
        """Prompt user to edit the suggestion with prefilled editable text."""
        # Show progress if provided
        if current is not None and total is not None:
            self.show_progress(current, total)

        self.console.print(
            "\n[bold blue]Edit alt text, or press Enter to accept. "
            "Commands: 'undo'/'u' go back, 'skip'/'s' skip, 'quit'/'q' save & exit, "
            "'help'/'?' for help. Exiting will save your progress.[/bold blue]"
        )

        while True:
            # Use prompt_toolkit for reliable prefilling across all shells
            try:
                result = prompt(
                    "> ",
                    default=suggestion,
                    vi_mode=self.vi_mode,
                    multiline=False,
                )
            except EOFError as err:
                # Treat Ctrl+D like Ctrl+C so callers save progress
                raise KeyboardInterrupt from err

            command = result.strip().lower()

            # Check for undo command
            if command in ("undo", "u"):
                return UNDO_REQUESTED
            if command in ("quit", "q"):
                return QUIT_REQUESTED
            if command in ("skip", "s"):
                return SKIP_REQUESTED
            if command in ("help", "?"):
                self._show_command_help()
                continue

            return result if result.strip() else suggestion

    def _show_command_help(self) -> None:
        """Print available editing commands."""
        self.console.print(
            "[bold]Commands:[/bold]\n"
            "  Enter        accept the prefilled/edited text\n"
            "  undo, u      undo the previous item and re-label it\n"
            "  skip, s      skip this item without recording it\n"
            "  quit, q      save progress and exit\n"
            "  help, ?      show this help"
        )

    def show_rule(self, title: str) -> None:
        """Display a separator rule."""
        self.console.print()
        self.console.rule(f"[bold]Asset: {title}[/bold]")

    def show_error(self, error_message: str) -> None:
        """Display error message."""
        self.console.print(
            Panel(
                error_message,
                title="Alt generation error",
                box=ROUNDED,
                style="red",
            )
        )


def _process_single_suggestion_for_labeling(
    suggestion_data: utils.AltGenerationResult,
    display: DisplayManager,
    prefetcher: "AssetPrefetcher",
    index: int,
    current: int | None = None,
    total: int | None = None,
) -> utils.AltGenerationResult:
    queue_item = _queue_item_for(suggestion_data)
    console = display.console

    # Acquire the (possibly pre-buffered) asset. A spinner reports progress
    # whenever the main thread has to wait - the first item, a deep undo, or a
    # background prepare that hasn't finished yet.
    with console.status("[bold]Preparing asset…") as status:
        plan = prefetcher.get(
            index, on_status=lambda msg: status.update(f"[bold]{msg}")
        )

    # Start buffering the next asset while the user edits this one, and free
    # assets we've moved well past.
    prefetcher.schedule(index + 1)
    prefetcher.retain_window(index)

    display.show_rule(queue_item.asset_path)
    display.show_context(queue_item)
    display.render(plan)

    # Allow user to edit the suggestion
    prefill_text = (
        suggestion_data.final_alt
        if suggestion_data.final_alt is not None
        else suggestion_data.suggested_alt
    )
    final_alt = prefill_text
    if sys.stdout.isatty():
        final_alt = display.prompt_for_edit(prefill_text, current, total)

    return replace(suggestion_data, final_alt=final_alt)


def _handle_undo_request(
    session: LabelingSession,
    console: Console,
) -> None:
    """Handle undo request by reverting to previous suggestion."""
    undone_result = session.undo()

    if undone_result is None:
        console.print("[yellow]Nothing to undo - at the beginning[/yellow]")
        return

    console.print(f"[yellow]Undoing: {undone_result.asset_path}[/yellow]")

    # Prefill with the previous final_alt value
    prefill_text = (
        undone_result.final_alt
        if undone_result.final_alt is not None
        else undone_result.suggested_alt
    )
    session.suggestions[session.current_index] = replace(
        session.suggestions[session.current_index],
        final_alt=prefill_text,
    )


def _process_labeling_loop(
    session: LabelingSession,
    display: DisplayManager,
    console: Console,
    prefetcher: "AssetPrefetcher",
) -> None:
    """Process all suggestions in the labeling session."""
    while not session.is_complete():
        current_suggestion = session.get_current_suggestion()
        if current_suggestion is None:
            break

        try:
            current, total = session.get_progress()
            index = session.current_index
            result = _process_single_suggestion_for_labeling(
                current_suggestion,
                display,
                prefetcher,
                index,
                current=current,
                total=total,
            )

            if result.final_alt == UNDO_REQUESTED:
                _handle_undo_request(session, console)
            elif result.final_alt == QUIT_REQUESTED:
                raise LabelingQuit
            elif result.final_alt == SKIP_REQUESTED:
                console.print(
                    f"[yellow]Skipping: {current_suggestion.asset_path}[/yellow]"
                )
                session.skip_current()
            else:
                session.add_result(result)

        except (
            utils.AltGenerationError,
            FileNotFoundError,
            requests.RequestException,
        ) as err:
            display.show_error(str(err))
            session.skip_current()


def label_suggestions(
    suggestions: Sequence[utils.AltGenerationResult],
    console: Console,
    output_path: Path,
    skip_existing: bool,
    vi_mode: bool = False,
) -> int:
    """Load suggestions and allow user to label them, collecting results.

    ``skip_existing`` controls whether suggestions already present in
    ``output_path`` are filtered out before labeling. It is intentionally
    decoupled from how results are written: results are ALWAYS appended to the
    output file so previously-labeled captions are never overwritten, even when
    ``skip_existing`` is False (a re-label session).
    """
    console.print(f"\n[bold blue]Labeling {len(suggestions)} suggestions[/bold blue]\n")

    if not sys.stdout.isatty():
        console.print(
            "[yellow]Non-interactive terminal: accepting suggestions as-is[/yellow]"
        )

    suggestions_to_process = (
        generate.filter_existing_captions(suggestions, [output_path], console)
        if skip_existing
        else list(suggestions)
    )

    session = LabelingSession(suggestions_to_process)
    display = DisplayManager(console, vi_mode=vi_mode)

    # A session-scoped workspace lets the prefetcher buffer the next asset
    # before the current item's temp dir would otherwise be torn down.
    with TemporaryDirectory(prefix="alt-text-llm-") as temp_root:
        prefetcher = AssetPrefetcher(session.suggestions, Path(temp_root))
        try:
            _process_labeling_loop(session, display, console, prefetcher)
        except (KeyboardInterrupt, LabelingQuit):
            console.print("\n[yellow]Saving progress...[/yellow]")
        finally:
            prefetcher.shutdown()
            if session.processed_results:
                # Always append so prior labels are never lost.
                utils.write_output(
                    session.processed_results, output_path, append_mode=True
                )
                console.print(
                    f"[green]Saved {len(session.processed_results)} results to {output_path}[/green]"
                )

    return len(session.processed_results)


def label_from_suggestions_file(
    suggestions_file: Path,
    output_path: Path,
    skip_existing: bool = False,
    vi_mode: bool = False,
) -> None:
    """Load suggestions from file and start labeling process."""
    console = Console()

    if not suggestions_file.exists():
        console.print(
            f"[red]Error: Suggestions file not found: {suggestions_file}[/red]"
        )
        raise SystemExit(1)

    try:
        with open(suggestions_file, encoding="utf-8") as f:
            suggestions_from_file = json.load(f)
    except json.JSONDecodeError as err:
        console.print(
            f"[red]Error: Invalid JSON in {suggestions_file}: {err}[/red]"
        )
        raise SystemExit(1) from err

    # Convert loaded data to AltGenerationResult, ignoring unknown fields
    suggestions = [
        utils.AltGenerationResult.from_json(suggestion)
        for suggestion in suggestions_from_file
    ]

    console.print(
        f"[green]Loaded {len(suggestions)} suggestions from {suggestions_file}[/green]"
    )

    processed_count = label_suggestions(
        suggestions, console, output_path, skip_existing=skip_existing, vi_mode=vi_mode
    )

    # Write final results
    console.print(
        f"\n[green]Completed! Wrote {processed_count} results to {output_path}[/green]"
    )
