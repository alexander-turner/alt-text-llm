"""Tests for generate.py module."""

import asyncio
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from alt_text_llm import generate, scan, utils


@pytest.mark.parametrize(
    "model, queue_count, avg_prompt_tokens, avg_output_tokens",
    [
        ("gemini-2.5-flash", 10, 300, 50),
        ("gemini-2.5-flash-lite", 100, 300, 50),
        ("gemini-2.5-flash", 1, 200, 30),
        ("gemini-2.5-flash-lite", 50, 400, 80),
    ],
)
def test_estimate_cost_calculation_parametrized(
    model: str,
    queue_count: int,
    avg_prompt_tokens: int,
    avg_output_tokens: int,
) -> None:
    # Retrieve costs from the actual MODEL_COSTS constant
    model_costs = generate.MODEL_COSTS[model]
    input_cost_per_1k = model_costs["input"]
    output_cost_per_1k = model_costs["output"]

    expected_input = (
        avg_prompt_tokens * queue_count / 1000
    ) * input_cost_per_1k
    expected_output = (
        avg_output_tokens * queue_count / 1000
    ) * output_cost_per_1k
    expected_total = expected_input + expected_output

    result = generate.estimate_cost(
        model, queue_count, avg_prompt_tokens, avg_output_tokens
    )

    assert f"${expected_total:.3f}" in result
    assert f"${expected_input:.3f} input" in result
    assert f"${expected_output:.3f} output" in result


@pytest.mark.parametrize(
    "model, queue_count",
    [
        ("gemini-2.5-flash", 1),
        ("gemini-2.5-flash", 10),
        ("gemini-2.5-flash-lite", 5),
        ("gemini-2.5-flash-lite", 100),
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


def test_estimate_cost_invalid_model() -> None:
    """Test cost estimation with invalid model returns informative message."""
    result = generate.estimate_cost("invalid-model", 10)

    assert result.startswith("Cost estimation not available for model")


@pytest.mark.parametrize(
    "model",
    ["gpt-4o-mini", "claude-3-5-sonnet", "claude-3-5-haiku"],
)
def test_estimate_cost_common_models(model: str) -> None:
    """Newly-added common models should return a dollar estimate."""
    assert model in generate.MODEL_COSTS
    result = generate.estimate_cost(model, 10)
    assert result.startswith("Estimated cost: $")
    assert " input + $" in result
    assert " output)" in result


def test_run_llm_success(temp_dir: Path) -> None:
    """Test successful LLM execution."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")
    prompt = "Generate alt text for this image"
    model = "gemini-2.5-flash"
    timeout = 60

    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = "Generated alt text"
    mock_result.stderr = ""

    with (
        patch("alt_text_llm.utils.find_executable", return_value="/usr/bin/llm"),
        patch("subprocess.run", return_value=mock_result) as mock_run,
    ):
        result = generate._run_llm(attachment, prompt, model, timeout)

        assert result == "Generated alt text"
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/usr/bin/llm"
        assert "-m" in call_args
        assert model in call_args
        assert "-a" in call_args
        assert str(attachment) in call_args
        assert prompt in call_args


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


@pytest.mark.parametrize(
    "returncode,stdout,stderr,expected_match",
    [
        pytest.param(1, "", "LLM error", "Caption generation failed", id="failure"),
        pytest.param(0, "   ", "", "LLM returned empty caption", id="empty_output"),
    ],
)
def test_run_llm_error_cases(
    temp_dir: Path, returncode: int, stdout: str, stderr: str, expected_match: str,
) -> None:
    """Test LLM execution failure and empty output."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")
    mock_result = Mock()
    mock_result.returncode = returncode
    mock_result.stdout = stdout
    mock_result.stderr = stderr

    with (
        patch("alt_text_llm.utils.find_executable", return_value="/usr/bin/llm"),
        patch("subprocess.run", return_value=mock_result),
        pytest.raises(utils.AltGenerationError, match=expected_match),
    ):
        generate._run_llm(attachment, "Generate alt text", "gemini-2.5-flash", 60)


def test_run_llm_timeout_raises_alt_generation_error(temp_dir: Path) -> None:
    """A subprocess timeout becomes a skippable AltGenerationError."""
    attachment = temp_dir / "test.jpg"
    attachment.write_bytes(b"fake image")

    with (
        patch("alt_text_llm.utils.find_executable", return_value="/usr/bin/llm"),
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="llm", timeout=5),
        ),
        pytest.raises(utils.AltGenerationError, match="timed out after 5s"),
    ):
        generate._run_llm(attachment, "Generate alt text", "gemini-2.5-flash", 5)


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
    ) -> str:
        return f"{attachment.name}-caption"

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
        return "caption"

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
    lock = asyncio.Lock()
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
        return "caption"

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
        return f"{attachment.name}-caption"

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
        return f"{attachment.name}-caption"

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
