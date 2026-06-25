"""Generate AI alt text suggestions for assets lacking meaningful alt text."""

import asyncio
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from rich.console import Console
from rich.progress import Progress

from alt_text_llm import scan, utils

# Approximate cost estimates per 1000 tokens (as of Sep 2025)
MODEL_COSTS = {
    # https://www.helicone.ai/llm-cost
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0025},
    "gemini-2.5-flash-lite": {"input": 0.00001, "output": 0.00004},
    # https://developers.googleblog.com/en/continuing-to-bring-you-our-latest-models-with-an-improved-gemini-2-5-flash-and-flash-lite-release/?ref=testingcatalog.com
    "gemini-2.5-flash-lite-preview-09-2025": {
        "input": 0.00001,
        "output": 0.00004,
    },
    "gemini-2.5-flash-preview-09-2025": {"input": 0.00001, "output": 0.00004},
    # Approximate per-1K-token costs for a few widely-used non-Gemini models.
    # These are rough public list prices (USD per 1K tokens) and may drift over
    # time as providers change pricing -- treat them as estimates only.
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
}


def _run_llm(
    attachment: Path,
    prompt: str,
    model: str,
    timeout: int,
) -> str:
    """Execute LLM command and return generated caption."""
    llm_path = utils.find_executable("llm")

    try:
        result = subprocess.run(
            [llm_path, "-m", model, "-a", str(attachment), "--usage", prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        # Convert to a skippable error so a single slow item does not abort
        # the entire async batch.
        raise utils.AltGenerationError(
            f"LLM timed out after {timeout}s for {attachment}"
        ) from err

    if result.returncode != 0:
        error_output = result.stderr.strip() or result.stdout.strip()
        raise utils.AltGenerationError(
            f"Caption generation failed for {attachment}: {error_output}"
        )

    cleaned = result.stdout.strip()
    if not cleaned:
        raise utils.AltGenerationError("LLM returned empty caption")
    return cleaned


@dataclass(slots=True)
class GenerateAltTextOptions:
    """Options for generating alt text."""

    root: Path
    model: str
    max_chars: int
    timeout: int
    output_path: Path
    skip_existing: bool = False


def estimate_cost(
    model: str,
    queue_count: int,
    avg_prompt_tokens: int = 4500,
    avg_output_tokens: int = 1500,
) -> str:
    """Estimate the cost of processing the queue with the given model."""
    model_lower = model.lower()
    if model_lower not in MODEL_COSTS:
        return f"Cost estimation not available for model: {model}"

    cost_info = MODEL_COSTS[model_lower]
    input_cost = (avg_prompt_tokens * queue_count / 1000) * cost_info["input"]
    output_cost = (avg_output_tokens * queue_count / 1000) * cost_info["output"]
    total_cost = input_cost + output_cost
    return f"Estimated cost: ${total_cost:.3f} (${input_cost:.3f} input + ${output_cost:.3f} output)"


_HasAssetPath = TypeVar(
    "_HasAssetPath", "scan.QueueItem", utils.AltGenerationResult
)


def filter_existing_captions(
    queue_items: Sequence[_HasAssetPath],
    output_paths: Sequence[Path],
    console: Console,
    verbose: bool = True,
) -> list[_HasAssetPath]:
    """Filter out items that already have captions in the output paths."""
    existing_captions = set()
    for output_path in output_paths:
        existing_captions.update(utils.load_existing_captions(output_path))
    original_count = len(queue_items)
    filtered_items = [
        item for item in queue_items if item.asset_path not in existing_captions
    ]
    skipped_count = original_count - len(filtered_items)
    if skipped_count > 0 and verbose:
        console.print(
            f"[dim]Skipped {skipped_count} items with existing captions[/dim]"
        )
    return filtered_items


# ---------------------------------------------------------------------------
# Async helpers for parallel LLM calls
# ---------------------------------------------------------------------------


_CONCURRENCY_LIMIT = 32


async def _run_llm_async(
    index: int,
    queue_item: "scan.QueueItem",
    options: GenerateAltTextOptions,
    sem: asyncio.Semaphore,
) -> tuple[int, utils.AltGenerationResult]:
    """Download asset, run LLM in a thread; clean up; return suggestion payload."""
    # Create the temp dir inside the semaphore so that a large queue does not
    # spawn thousands of temp directories upfront (all tasks are created
    # eagerly). Only up to _CONCURRENCY_LIMIT workspaces exist at once.
    async with sem:
        workspace = Path(tempfile.mkdtemp())
        try:
            attachment = await asyncio.to_thread(
                utils.download_asset, queue_item, workspace
            )
            prompt = utils.build_prompt(queue_item, options.max_chars)
            caption = await asyncio.to_thread(
                _run_llm,
                attachment,
                prompt,
                options.model,
                options.timeout,
            )
            return index, utils.AltGenerationResult(
                markdown_file=queue_item.markdown_file,
                asset_path=queue_item.asset_path,
                suggested_alt=caption,
                model=options.model,
                context_snippet=queue_item.context_snippet,
                line_number=queue_item.line_number,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


async def async_generate_suggestions(
    queue_items: Sequence["scan.QueueItem"],
    options: GenerateAltTextOptions,
) -> list[utils.AltGenerationResult]:
    """Generate suggestions concurrently for *queue_items*."""
    if not queue_items:
        return []

    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)
    tasks = [
        asyncio.create_task(_run_llm_async(index, qi, options, sem))
        for index, qi in enumerate(queue_items)
    ]

    # Collect by original index so output order is deterministic (matches input
    # order) regardless of completion order. Failed/skipped items are omitted.
    completed: dict[int, utils.AltGenerationResult] = {}
    with Progress() as progress:
        task_id = progress.add_task("Generating alt text", total=len(tasks))
        try:
            for finished in asyncio.as_completed(tasks):
                try:
                    index, result = await finished
                    completed[index] = result
                except (
                    utils.AltGenerationError,
                    FileNotFoundError,
                ) as err:
                    # Skip individual items that fail (e.g., unsupported file types)
                    progress.console.print(f"Skipped item due to error: {err}")
                progress.advance(task_id)
        except asyncio.CancelledError:
            progress.update(
                task_id,
                description="Generating alt text (cancelled, finishing up...)",
            )

    return [completed[index] for index in sorted(completed)]
