"""Tests for generate.py module."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from alt_text_llm import generate, openrouter, scan, utils

# Live per-token pricing (USD) as returned by OpenRouter's catalogue.
_FAKE_PRICING = {
    "google/gemini-2.5-flash": {"input": 0.0000003, "output": 0.0000025},
    "google/gemini-2.5-flash-lite": {"input": 0.00000001, "output": 0.00000004},
}


@pytest.fixture
def _patch_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch openrouter.get_pricing with a deterministic local price table."""

    def fake_get_pricing(model: str) -> dict[str, float] | None:
        return _FAKE_PRICING.get(model)

    monkeypatch.setattr(openrouter, "get_pricing", fake_get_pricing)


@pytest.mark.usefixtures("_patch_pricing")
@pytest.mark.parametrize(
    "model, queue_count, avg_prompt_tokens, avg_output_tokens",
    [
        ("google/gemini-2.5-flash", 10, 300, 50),
        ("google/gemini-2.5-flash-lite", 100, 300, 50),
        ("google/gemini-2.5-flash", 1, 200, 30),
        ("google/gemini-2.5-flash-lite", 50, 400, 80),
    ],
)
def test_estimate_cost_calculation_parametrized(
    model: str,
    queue_count: int,
    avg_prompt_tokens: int,
    avg_output_tokens: int,
) -> None:
    pricing = _FAKE_PRICING[model]

    expected_input = avg_prompt_tokens * queue_count * pricing["input"]
    expected_output = avg_output_tokens * queue_count * pricing["output"]
    expected_total = expected_input + expected_output

    result = generate.estimate_cost(
        model, queue_count, avg_prompt_tokens, avg_output_tokens
    )

    assert f"${expected_total:.3f}" in result
    assert f"${expected_input:.3f} input" in result
    assert f"${expected_output:.3f} output" in result


@pytest.mark.usefixtures("_patch_pricing")
@pytest.mark.parametrize(
    "model, queue_count",
    [
        ("google/gemini-2.5-flash", 1),
        ("google/gemini-2.5-flash", 10),
        ("google/gemini-2.5-flash-lite", 5),
        ("google/gemini-2.5-flash-lite", 100),
    ],
)
def test_estimate_cost_format_consistency(
    model: str, queue_count: int
) -> None:
    """Test that cost estimation returns consistently formatted results."""
    result = generate.estimate_cost(model, queue_count)

    # Check format consistency
    assert result.startswith("Estimated cost: $")
    assert " input + $" in result
    assert " output)" in result
    assert result.count("$") == 3  # Total, input, output


@pytest.mark.usefixtures("_patch_pricing")
def test_estimate_cost_invalid_model() -> None:
    """Cost estimation with an unpriced model returns an informative message."""
    result = generate.estimate_cost("invalid/model", 10)

    assert result.startswith("Cost estimation not available for model")


def test_estimate_cost_uses_live_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """estimate_cost must consult openrouter.get_pricing (live catalogue)."""
    seen: list[str] = []

    def fake_get_pricing(model: str) -> dict[str, float]:
        seen.append(model)
        return {"input": 0.000001, "output": 0.000002}

    monkeypatch.setattr(openrouter, "get_pricing", fake_get_pricing)

    result = generate.estimate_cost("any/model", 2, 1000, 500)

    assert seen == ["any/model"]
    # input: 1000 * 2 * 1e-6 = 0.002 ; output: 500 * 2 * 2e-6 = 0.002
    assert "$0.004" in result
    assert "$0.002 input" in result
    assert "$0.002 output" in result


def test_run_llm_success(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_run_llm returns the caption and the request's actual cost."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")
    prompt = "Generate alt text for this image"
    model = "google/gemini-2.5-flash"

    captured: dict[str, object] = {}

    def fake_generate_caption(att, p, m, t):
        captured.update(attachment=att, prompt=p, model=m, timeout=t)
        return "Generated alt text", {"cost": 0.00042}

    monkeypatch.setattr(openrouter, "generate_caption", fake_generate_caption)

    caption, cost = generate._run_llm(attachment, prompt, model, 60)

    assert caption == "Generated alt text"
    assert cost == 0.00042
    assert captured == {
        "attachment": attachment,
        "prompt": prompt,
        "model": model,
        "timeout": 60,
    }


def test_run_llm_missing_cost_returns_none(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When OpenRouter omits a cost, _run_llm reports None."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")

    monkeypatch.setattr(
        openrouter,
        "generate_caption",
        lambda *a, **k: ("caption", {"prompt_tokens": 10}),
    )

    caption, cost = generate._run_llm(attachment, "p", "m", 60)
    assert caption == "caption"
    assert cost is None


def test_run_llm_propagates_openrouter_error(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors from OpenRouter surface as AltGenerationError subclasses."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")

    def boom(*a, **k):
        raise openrouter.OpenRouterError("Unknown model: bogus")

    monkeypatch.setattr(openrouter, "generate_caption", boom)

    with pytest.raises(utils.AltGenerationError, match="Unknown model"):
        generate._run_llm(attachment, "p", "bogus", 60)


def test_filter_existing_captions_filters_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue_items = [
        scan.QueueItem(
            markdown_file="test1.md",
            asset_path="image1.jpg",
            line_number=1,
            context_snippet="context1",
        ),
        scan.QueueItem(
            markdown_file="test2.md",
            asset_path="image2.jpg",
            line_number=2,
            context_snippet="context2",
        ),
    ]

    def fake_load_existing_captions(_path: Path) -> set[str]:
        return {"image1.jpg"}

    monkeypatch.setattr(
        utils,
        "load_existing_captions",
        fake_load_existing_captions,
    )

    console_mock = Mock()
    console_mock.print = Mock()

    filtered = generate.filter_existing_captions(
        queue_items,
        [Path("captions.json")],
        console_mock,
    )

    assert len(filtered) == 1
    assert filtered[0].asset_path == "image2.jpg"
    console_mock.print.assert_called_once()


@pytest.mark.asyncio
async def test_async_generate_suggestions(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    queue_items = [
        scan.QueueItem(
            markdown_file="test1.md",
            asset_path="image1.jpg",
            line_number=1,
            context_snippet="context1",
        ),
        scan.QueueItem(
            markdown_file="test2.md",
            asset_path="image2.jpg",
            line_number=2,
            context_snippet="context2",
        ),
    ]

    def fake_download_asset(
        queue_item: scan.QueueItem, workspace: Path
    ) -> Path:
        asset_filename = Path(queue_item.asset_path).name or "asset"
        target_path = workspace / asset_filename
        target_path.write_bytes(b"data")
        return target_path

    monkeypatch.setattr(
        utils,
        "download_asset",
        fake_download_asset,
    )

    def fake_run_llm(
        attachment: Path, prompt: str, model: str, timeout: int
    ) -> tuple[str, float | None]:
        return f"{attachment.name}-caption", 0.001

    monkeypatch.setattr(generate, "_run_llm", fake_run_llm)

    def fake_generate_article_context(
        queue_item: scan.QueueItem,
        max_before: int | None = None,
        max_after: int = 2,
        trim_frontmatter: bool = False,
    ) -> str:
        return queue_item.context_snippet

    monkeypatch.setattr(
        utils,
        "generate_article_context",
        fake_generate_article_context,
    )

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test-model",
        max_chars=50,
        timeout=10,
        output_path=temp_dir / "captions.json",
        skip_existing=False,
    )

    results = await generate.async_generate_suggestions(queue_items, options)

    assert len(results) == len(queue_items)
    result_asset_paths = {result.asset_path for result in results}
    expected_asset_paths = {item.asset_path for item in queue_items}
    assert result_asset_paths == expected_asset_paths

    expected_suggestions = {
        f"{Path(item.asset_path).name}-caption" for item in queue_items
    }
    actual_suggestions = {result.suggested_alt for result in results}
    assert actual_suggestions == expected_suggestions


@pytest.mark.asyncio
async def test_async_generate_reports_total_cost(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    """The summed per-request cost is printed after generation."""
    queue_items = [
        scan.QueueItem(
            markdown_file="t.md",
            asset_path=f"image{i}.jpg",
            line_number=i + 1,
            context_snippet="ctx",
        )
        for i in range(3)
    ]

    def fake_download(qi, workspace):
        target = workspace / "asset.jpg"
        target.write_bytes(b"data")
        return target

    monkeypatch.setattr(utils, "download_asset", fake_download)
    monkeypatch.setattr(
        utils, "generate_article_context", lambda qi, **k: "ctx"
    )
    # Two items report a cost; one reports None and must be ignored.
    costs = iter([0.001, None, 0.0025])
    monkeypatch.setattr(
        generate, "_run_llm", lambda *a, **k: ("caption", next(costs))
    )

    printed: list[str] = []
    console = Mock()
    console.print = lambda msg: printed.append(str(msg))
    monkeypatch.setattr(generate, "Console", lambda: console)

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="m",
        max_chars=50,
        timeout=10,
        output_path=temp_dir / "out.json",
    )
    results = await generate.async_generate_suggestions(queue_items, options)

    assert len(results) == 3
    # 0.001 + 0.0025 = 0.0035 -> "$0.0035"
    assert any("Actual cost: $0.0035" in line for line in printed)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_generate_empty_queue(temp_dir: Path) -> None:
    """Async generation with empty queue should return empty list."""
    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test",
        max_chars=100,
        timeout=10,
        output_path=temp_dir / "out.json",
    )
    results = await generate.async_generate_suggestions([], options)
    assert results == []


@pytest.mark.asyncio
async def test_async_generate_with_individual_errors(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async generation should handle individual failures gracefully."""
    queue_items = [
        scan.QueueItem(
            markdown_file="test.md",
            asset_path=f"image{i}.jpg",
            line_number=i + 1,
            context_snippet=f"ctx{i}",
        )
        for i in range(5)
    ]

    def failing_download(qi, workspace):
        if "image2" in qi.asset_path:
            raise FileNotFoundError("Not found")
        target = workspace / "asset.jpg"
        target.write_bytes(b"data")
        return target

    def fake_run_llm(attachment, prompt, model, timeout):
        return "caption", None

    def fake_context(qi, **kwargs):
        return qi.context_snippet

    monkeypatch.setattr(utils, "download_asset", failing_download)
    monkeypatch.setattr(generate, "_run_llm", fake_run_llm)
    monkeypatch.setattr(utils, "generate_article_context", fake_context)

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test",
        max_chars=100,
        timeout=10,
        output_path=temp_dir / "out.json",
    )
    results = await generate.async_generate_suggestions(queue_items, options)
    assert len(results) == 4  # 4 out of 5 should succeed


@pytest.mark.asyncio
async def test_async_generate_bounds_concurrent_temp_dirs(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At most _CONCURRENCY_LIMIT temp dirs should exist simultaneously."""
    item_count = generate._CONCURRENCY_LIMIT * 3
    queue_items = [
        scan.QueueItem(
            markdown_file="test.md",
            asset_path=f"image{i}.jpg",
            line_number=i + 1,
            context_snippet=f"ctx{i}",
        )
        for i in range(item_count)
    ]

    live = 0
    peak = 0
    real_mkdtemp = tempfile.mkdtemp

    def counting_mkdtemp(*args, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        return real_mkdtemp(*args, **kwargs)

    real_rmtree = generate.shutil.rmtree

    def counting_rmtree(path, *args, **kwargs):
        nonlocal live
        live -= 1
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(tempfile, "mkdtemp", counting_mkdtemp)
    monkeypatch.setattr(generate.shutil, "rmtree", counting_rmtree)

    def fake_download_asset(qi: scan.QueueItem, workspace: Path) -> Path:
        target = workspace / "asset.jpg"
        target.write_bytes(b"data")
        return target

    async def fake_run_llm(attachment, prompt, model, timeout):
        # Yield control so many coroutines interleave inside the semaphore.
        await asyncio.sleep(0.01)
        return "caption"

    def sync_run_llm(attachment, prompt, model, timeout):
        # _run_llm is invoked via asyncio.to_thread; emulate a brief delay.
        import time

        time.sleep(0.01)
        return "caption", None

    monkeypatch.setattr(utils, "download_asset", fake_download_asset)
    monkeypatch.setattr(generate, "_run_llm", sync_run_llm)
    monkeypatch.setattr(utils, "build_prompt", lambda qi, mc: qi.context_snippet)

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test",
        max_chars=100,
        timeout=10,
        output_path=temp_dir / "out.json",
    )

    results = await generate.async_generate_suggestions(queue_items, options)
    assert len(results) == item_count
    assert peak <= generate._CONCURRENCY_LIMIT
    assert live == 0  # every workspace was cleaned up


@pytest.mark.asyncio
async def test_async_generate_preserves_input_order(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Results are returned in input order despite varying completion times."""
    item_count = 8
    queue_items = [
        scan.QueueItem(
            markdown_file="test.md",
            asset_path=f"image{i}.jpg",
            line_number=i + 1,
            context_snippet=f"ctx{i}",
        )
        for i in range(item_count)
    ]

    def fake_download_asset(qi: scan.QueueItem, workspace: Path) -> Path:
        target = workspace / Path(qi.asset_path).name
        target.write_bytes(b"data")
        return target

    def fake_run_llm(attachment, prompt, model, timeout):
        # Sleep inversely to index so later items finish first.
        import time

        index = int(attachment.stem.replace("image", ""))
        time.sleep((item_count - index) * 0.005)
        return f"{attachment.name}-caption", None

    monkeypatch.setattr(utils, "download_asset", fake_download_asset)
    monkeypatch.setattr(generate, "_run_llm", fake_run_llm)
    monkeypatch.setattr(utils, "build_prompt", lambda qi, mc: qi.context_snippet)

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test",
        max_chars=100,
        timeout=10,
        output_path=temp_dir / "out.json",
    )

    results = await generate.async_generate_suggestions(queue_items, options)
    assert [r.asset_path for r in results] == [
        qi.asset_path for qi in queue_items
    ]


@pytest.mark.asyncio
async def test_async_generate_survives_timeout(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One timing-out item is skipped; other items still succeed."""
    queue_items = [
        scan.QueueItem(
            markdown_file="test.md",
            asset_path=f"image{i}.jpg",
            line_number=i + 1,
            context_snippet=f"ctx{i}",
        )
        for i in range(3)
    ]

    def fake_download_asset(qi: scan.QueueItem, workspace: Path) -> Path:
        target = workspace / Path(qi.asset_path).name
        target.write_bytes(b"data")
        return target

    def fake_run_llm(attachment, prompt, model, timeout):
        if attachment.name == "image1.jpg":
            raise utils.AltGenerationError("LLM timed out after 5s")
        return f"{attachment.name}-caption", None

    monkeypatch.setattr(utils, "download_asset", fake_download_asset)
    monkeypatch.setattr(generate, "_run_llm", fake_run_llm)
    monkeypatch.setattr(utils, "build_prompt", lambda qi, mc: qi.context_snippet)

    options = generate.GenerateAltTextOptions(
        root=temp_dir,
        model="test",
        max_chars=100,
        timeout=5,
        output_path=temp_dir / "out.json",
    )

    results = await generate.async_generate_suggestions(queue_items, options)
    asset_paths = [r.asset_path for r in results]
    assert asset_paths == ["image0.jpg", "image2.jpg"]
