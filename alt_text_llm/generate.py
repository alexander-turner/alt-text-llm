"""Generate AI alt text suggestions for assets lacking meaningful alt text."""

import asyncio
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from rich.console import Console
from rich.progress import Progress

from alt_text_llm import openrouter, scan, utils


def _run_llm(
    attachment: Path,
    prompt: str,
    model: str,
    timeout: int,
) -> tuple[str, float | None]:
    """Generate a caption via OpenRouter.

    Returns the caption text and the request's actual cost in USD (``None`` if
    OpenRouter did not report a cost for the request).
    """
    caption, usage = openrouter.generate_caption(
        attachment, prompt, model, timeout
    )
    cost = usage.get("cost")
    return caption, float(cost) if isinstance(cost, (int, float)) else None


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
    """Estimate the cost of processing the queue with the given model.

    Pricing is pulled live from OpenRouter's model catalogue (USD per token).
    """
    pricing = openrouter.get_pricing(model)
    if pricing is None:
        return f"Cost estimation not available for model: {model}"

    input_cost = avg_prompt_tokens * queue_count * pricing["input"]
    output_cost = avg_output_tokens * queue_count * pricing["output"]
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
    queue_item: "scan.QueueItem",
    options: GenerateAltTextOptions,
    sem: asyncio.Semaphore,
) -> tuple[utils.AltGenerationResult, float | None]:
    """Download asset, run LLM in a thread; clean up; return suggestion payload.

    Returns the suggestion alongside the request's actual cost in USD (``None``
    when OpenRouter did not report a cost).
    """
    workspace = Path(tempfile.mkdtemp())
    try:
        async with sem:
            attachment = await asyncio.to_thread(
                utils.download_asset, queue_item, workspace
            )
            prompt = utils.build_prompt(queue_item, options.max_chars)
            caption, cost = await asyncio.to_thread(
                _run_llm,
                attachment,
                prompt,
                options.model,
                options.timeout,
            )
        result = utils.AltGenerationResult(
            markdown_file=queue_item.markdown_file,
            asset_path=queue_item.asset_path,
            suggested_alt=caption,
            model=options.model,
            context_snippet=queue_item.context_snippet,
            line_number=queue_item.line_number,
        )
        return result, cost
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


async def async_generate_suggestions(
    queue_items: Sequence["scan.QueueItem"],
    options: GenerateAltTextOptions,
) -> list[utils.AltGenerationResult]:
    """Generate suggestions concurrently for *queue_items*.

    Prints the total actual cost (summed from OpenRouter's per-request ``cost``)
    once generation finishes.
    """
    if not queue_items:
        return []

    sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)
    tasks = [
        asyncio.create_task(_run_llm_async(qi, options, sem))
        for qi in queue_items
    ]

    suggestions: list[utils.AltGenerationResult] = []
    total_cost = 0.0
    with Progress() as progress:
        task_id = progress.add_task("Generating alt text", total=len(tasks))
        try:
            for finished in asyncio.as_completed(tasks):
                try:
                    result, cost = await finished
                    suggestions.append(result)
                    if cost is not None:
                        total_cost += cost
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

    if total_cost > 0:
        Console().print(
            f"[bold blue]Actual cost: ${total_cost:.4f}[/bold blue]"
        )

    return suggestions
