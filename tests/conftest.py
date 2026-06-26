"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path

import pytest

from alt_text_llm import openrouter, scan, utils

FAKE_LLM_CAPTION = "A friendly robot waving hello"


@pytest.fixture(autouse=True)
def _clear_executable_cache():
    """Clear the executable path cache between tests."""
    utils._executable_cache.clear()
    yield
    utils._executable_cache.clear()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def fake_llm_on_path(monkeypatch: pytest.MonkeyPatch) -> str:
    """Stub OpenRouter generation with a deterministic caption.

    Patches ``openrouter.generate_caption`` so the REAL generate pipeline
    (``generate._run_llm`` -> ``_run_llm_async`` -> ``async_generate_suggestions``)
    runs without network or a real LLM. Returns the caption string the stub
    emits.
    """

    def fake_generate_caption(attachment, prompt, model, timeout):
        return FAKE_LLM_CAPTION, {"cost": 0.0001}

    monkeypatch.setattr(openrouter, "generate_caption", fake_generate_caption)
    return FAKE_LLM_CAPTION


@pytest.fixture
def base_queue_item(temp_dir: Path) -> scan.QueueItem:
    """Provides a base QueueItem for testing."""
    return scan.QueueItem(
        markdown_file=str(temp_dir / "test.md"),
        asset_path="image.jpg",
        line_number=5,
        context_snippet="This is a test image context.",
    )
