"""Shared pytest fixtures for all tests."""

import tempfile
from pathlib import Path

import pytest

from alt_text_llm import scan, utils


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
def base_queue_item(temp_dir: Path) -> scan.QueueItem:
    """Provides a base QueueItem for testing."""
    return scan.QueueItem(
        markdown_file=str(temp_dir / "test.md"),
        asset_path="image.jpg",
        line_number=5,
        context_snippet="This is a test image context.",
    )
