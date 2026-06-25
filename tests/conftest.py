"""Shared pytest fixtures for all tests."""

import os
import tempfile
from pathlib import Path

import pytest

from alt_text_llm import scan, utils

from tests.test_helpers import write_fake_llm

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
def fake_llm_on_path(tmp_path: Path):
    """Install a fake ``llm`` binary on PATH that echoes a deterministic caption.

    Prepends a temp bin dir holding the fake ``llm`` to ``PATH`` and clears the
    executable cache so the REAL subprocess code in ``generate._run_llm`` runs
    against it. PATH and the cache are restored on teardown. Yields the caption
    string the fake binary emits.
    """
    bin_dir = tmp_path / "fake_bin"
    write_fake_llm(bin_dir, caption=FAKE_LLM_CAPTION)

    original_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{original_path}"
    utils._executable_cache.clear()
    try:
        yield FAKE_LLM_CAPTION
    finally:
        os.environ["PATH"] = original_path
        utils._executable_cache.clear()


@pytest.fixture
def base_queue_item(temp_dir: Path) -> scan.QueueItem:
    """Provides a base QueueItem for testing."""
    return scan.QueueItem(
        markdown_file=str(temp_dir / "test.md"),
        asset_path="image.jpg",
        line_number=5,
        context_snippet="This is a test image context.",
    )
