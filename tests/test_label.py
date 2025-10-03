"""Tests for label.py module."""

import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich import console
from rich.console import Console

from alt_text_llm import label, scan, utils
from tests.test_helpers import create_markdown_file, create_test_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_alt(
    idx: int, *, final_alt: str | None = None
) -> utils.AltGenerationResult:
    """Factory for AltGenerationResult with deterministic dummy fields."""
    return utils.AltGenerationResult(
        markdown_file=f"test{idx}.md",
        asset_path=f"image{idx}.jpg",
        suggested_alt=f"suggestion {idx}",
        final_alt=final_alt,
        model="test-model",
        context_snippet=f"context {idx}",
        line_number=idx,
    )


@pytest.fixture
def test_suggestions() -> list[utils.AltGenerationResult]:
    """Test suggestions for error handling tests."""
    return [
        utils.AltGenerationResult(
            markdown_file="test1.md",
            asset_path="image1.jpg",
            suggested_alt="First",
            model="test",
            context_snippet="ctx1",
            line_number=1,
        ),
        utils.AltGenerationResult(
            markdown_file="test2.md",
            asset_path="image2.jpg",
            suggested_alt="Second",
            model="test",
            context_snippet="ctx2",
            line_number=2,
        ),
    ]


@contextmanager
def _setup_error_mocks(error_type, error_on_item: str):
    """Helper to set up mocks that raise errors on specific items."""

    def mock_download_asset(queue_item, workspace):
        if error_on_item in queue_item.asset_path:
            raise error_type(f"Error on {queue_item.asset_path}")
        test_file = workspace / "test.jpg"
        test_file.write_bytes(b"fake image")
        return test_file

    with (
        patch("sys.stdout.isatty", return_value=False),
        patch.object(
            utils,
            "download_asset",
            side_effect=mock_download_asset,
        ),
        patch.object(label.DisplayManager, "show_error"),
        patch.object(label.DisplayManager, "show_context"),
        patch.object(label.DisplayManager, "show_rule"),
        patch.object(label.DisplayManager, "show_image"),
    ):
        yield


def _maybe_assert_saved_results(
    output_file: Path, expected_count: int
) -> None:
    """Helper to assert saved results match expectations."""
    if expected_count > 0:
        assert output_file.exists()
        with output_file.open("r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert len(saved_data) == expected_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisplayManager:
    """Test the DisplayManager class."""

    @pytest.fixture
    def display_manager(self) -> label.DisplayManager:
        """Create a DisplayManager with mocked console for testing."""
        richConsole = console.Console(file=Mock())
        return label.DisplayManager(richConsole)

    def test_display_manager_creation(self) -> None:
        richConsole = console.Console()
        display = label.DisplayManager(richConsole)
        assert display.console is richConsole

    def test_show_context(
        self,
        display_manager: label.DisplayManager,
        base_queue_item: scan.QueueItem,
    ) -> None:
        # Create the markdown file that the queue item references
        markdown_file = Path(base_queue_item.markdown_file)
        create_markdown_file(
            markdown_file, content="Test content for context display."
        )

        # Should not raise an exception
        display_manager.show_context(base_queue_item)

    def test_show_image_not_tty(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        test_image = temp_dir / "test.jpg"
        create_test_image(test_image, "100x100")

        with (
            patch("sys.stdout.isatty", return_value=False),
            patch.dict("os.environ", {}, clear=True),  # Clear TMUX env var
            patch("subprocess.run") as mock_run,
        ):
            # Should not raise an exception and should call imgcat
            display_manager.show_image(test_image)
            mock_run.assert_called_once_with(
                ["imgcat", str(test_image)], check=True
            )

    def test_show_image_success(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        test_image = temp_dir / "test.jpg"
        create_test_image(test_image, "100x100")

        with (
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {}, clear=True),  # Clear TMUX env var
        ):
            display_manager.show_image(test_image)

            # Should have called imgcat with the image path
            mock_run.assert_called_once_with(
                ["imgcat", str(test_image)], check=True
            )

    def test_show_image_subprocess_error(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        test_image = temp_dir / "test.jpg"
        create_test_image(test_image, "100x100")

        with (
            patch("subprocess.run") as mock_run,
            patch.dict("os.environ", {}, clear=True),  # Clear TMUX env var
        ):
            mock_run.side_effect = subprocess.CalledProcessError(
                1, ["imgcat", str(test_image)]
            )
            with pytest.raises(ValueError):
                display_manager.show_image(test_image)

    def test_show_image_tmux_error(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        test_image = temp_dir / "test.jpg"
        create_test_image(test_image, "100x100")

        with patch.dict("os.environ", {"TMUX": "1"}):
            with pytest.raises(ValueError, match="Cannot open image in tmux"):
                display_manager.show_image(test_image)


def test_label_suggestions_handles_file_errors(
    temp_dir: Path,
    test_suggestions: list[utils.AltGenerationResult],
) -> None:
    """Test that individual file errors are handled gracefully and processing continues."""
    output_file = temp_dir / "test_output.json"

    with _setup_error_mocks(FileNotFoundError, "image2.jpg"):
        result_count = label.label_suggestions(
            test_suggestions, Mock(), output_file, append_mode=False
        )

    assert result_count == 1  # Only first item processed successfully
    _maybe_assert_saved_results(output_file, 1)


def test_label_suggestions_saves_on_keyboard_interrupt(
    temp_dir: Path,
    test_suggestions: list[utils.AltGenerationResult],
) -> None:
    """Test that results are saved when KeyboardInterrupt occurs during processing."""
    output_file = temp_dir / "test_output.json"

    with _setup_error_mocks(KeyboardInterrupt, "image2.jpg"):
        # KeyboardInterrupt is caught and handled gracefully, no exception raised
        label.label_suggestions(
            test_suggestions, Mock(), output_file, append_mode=False
        )

    _maybe_assert_saved_results(output_file, 1)


def test_label_suggestions_saves_on_runtime_error(
    temp_dir: Path,
    test_suggestions: list[utils.AltGenerationResult],
) -> None:
    """Test that results are saved when RuntimeError occurs during processing."""
    output_file = temp_dir / "test_output.json"

    with _setup_error_mocks(RuntimeError, "image1.jpg"):
        # RuntimeError is not caught, so it should still raise
        with pytest.raises(RuntimeError):
            label.label_suggestions(
                test_suggestions, Mock(), output_file, append_mode=False
            )

    _maybe_assert_saved_results(output_file, 0)


def test_label_from_suggestions_file_loads_and_filters_data(
    temp_dir: Path,
) -> None:
    """Test that label_from_suggestions_file loads suggestions and preserves final_alt if present."""
    suggestions_file = temp_dir / "suggestions.json"
    output_file = temp_dir / "output.json"

    suggestions_data = [
        {
            "markdown_file": "test.md",
            "asset_path": "image.jpg",
            "suggested_alt": "Test suggestion",
            "final_alt": "Previously labeled alt text",  # Should be preserved
            "model": "test-model",
            "context_snippet": "context",
            "line_number": 10,
        }
    ]

    suggestions_file.write_text(json.dumps(suggestions_data), encoding="utf-8")

    with patch.object(label, "label_suggestions") as mock_label:
        mock_label.return_value = 1
        label.label_from_suggestions_file(
            suggestions_file, output_file, skip_existing=False
        )

    loaded_suggestions = mock_label.call_args[0][0]
    assert len(loaded_suggestions) == 1
    assert loaded_suggestions[0].asset_path == "image.jpg"
    assert loaded_suggestions[0].line_number == 10
    assert loaded_suggestions[0].final_alt == "Previously labeled alt text"


def test_label_from_suggestions_file_without_final_alt_field(
    temp_dir: Path,
) -> None:
    """Test that suggestions without final_alt field are loaded correctly."""
    suggestions_file = temp_dir / "suggestions.json"
    output_file = temp_dir / "output.json"

    suggestions_data = [
        {
            "markdown_file": "test.md",
            "asset_path": "image.jpg",
            "suggested_alt": "Test suggestion",
            # No final_alt field at all
            "model": "test-model",
            "context_snippet": "context",
            "line_number": 10,
        }
    ]

    suggestions_file.write_text(json.dumps(suggestions_data), encoding="utf-8")

    with patch.object(label, "label_suggestions") as mock_label:
        mock_label.return_value = 1
        label.label_from_suggestions_file(
            suggestions_file, output_file, skip_existing=False
        )

    loaded_suggestions = mock_label.call_args[0][0]
    assert len(loaded_suggestions) == 1
    assert loaded_suggestions[0].final_alt is None


@pytest.mark.parametrize(
    "error,file_content",
    [
        (json.JSONDecodeError, "invalid json"),
        (FileNotFoundError, None),  # File doesn't exist
        (
            TypeError,
            '[{"markdown_file": "test.md"}]',
        ),  # Missing required fields
    ],
)
def test_label_from_suggestions_file_error_handling(
    temp_dir: Path, error: type, file_content: str | None
) -> None:
    """Test error handling for various file and data issues."""
    suggestions_file = temp_dir / "suggestions.json"

    if file_content is not None:
        suggestions_file.write_text(file_content, encoding="utf-8")

    with pytest.raises(error):
        label.label_from_suggestions_file(
            suggestions_file, temp_dir / "output.json", skip_existing=False
        )


@pytest.mark.parametrize("user_input", ["undo", "u", "UNDO"])
def test_prompt_for_edit_undo_command(user_input: str) -> None:
    """prompt_for_edit returns sentinel on various undo inputs."""
    console = Console()
    display = label.DisplayManager(console)

    with patch("builtins.input", return_value=user_input):
        result = display.prompt_for_edit("test suggestion")
        assert result == label.UNDO_REQUESTED


def test_labeling_session() -> None:
    """Test the LabelingSession helper class."""
    suggestions = [create_alt(1), create_alt(2)]

    session = label.LabelingSession(suggestions)

    # Initial state
    assert not session.is_complete()
    assert not session.can_undo()
    assert session.get_progress() == (1, 2)
    assert session.get_current_suggestion() == suggestions[0]

    # Process first item
    result1 = create_alt(1, final_alt="final 1")
    session.add_result(result1)

    # After processing first item
    assert not session.is_complete()
    assert session.can_undo()
    assert session.get_progress() == (2, 2)
    assert session.get_current_suggestion() == suggestions[1]

    # Test undo
    undone = session.undo()
    assert undone == result1
    assert session.get_progress() == (1, 2)
    assert session.get_current_suggestion() == suggestions[0]
    assert not session.can_undo()

    # Process both items
    session.add_result(result1)
    result2 = create_alt(2, final_alt="final 2")
    session.add_result(result2)

    # Complete
    assert session.is_complete()
    assert session.get_current_suggestion() is None
    assert len(session.processed_results) == 2


@pytest.mark.parametrize(
    "sequence,expected_saved",
    [
        # Undo in middle then accept second item
        (
            [
                "accepted 1",
                label.UNDO_REQUESTED,
                "modified 1",
                "accepted 2",
            ],
            ["modified 1", "accepted 2"],
        ),
        # Undo at beginning then accept
        (
            [label.UNDO_REQUESTED, "accepted"],
            ["accepted"],
        ),
    ],
)
def test_label_suggestions_sequences(
    temp_dir: Path, sequence: list[str], expected_saved: list[str]
) -> None:
    """Parametrized test covering various undo/accept sequences."""

    console = Console()
    output_path = temp_dir / "output.json"

    # Build suggestions equal to length of unique images needed (max 3)
    suggestions = [create_alt(i + 1) for i in range(max(3, len(sequence)))]

    call_count = 0

    def mock_process_single_suggestion(
        suggestion_data, display, current=None, total=None
    ):
        nonlocal call_count
        final = (
            sequence[call_count]
            if call_count < len(sequence)
            else "accepted tail"
        )
        call_count += 1
        return create_alt(suggestion_data.line_number, final_alt=final)

    with patch.object(
        label,
        "_process_single_suggestion_for_labeling",
        side_effect=mock_process_single_suggestion,
    ):
        label.label_suggestions(
            suggestions, console, output_path, append_mode=True
        )

    saved = [
        r["final_alt"]
        for r in json.loads(output_path.read_text(encoding="utf-8"))
    ]
    assert saved[: len(expected_saved)] == expected_saved


def test_prefill_after_undo(temp_dir: Path) -> None:
    """Ensure that after an undo, the previous final_alt is used as prefill."""

    console = Console()
    output_path = temp_dir / "output.json"

    suggestions = [create_alt(1), create_alt(2)]

    # Sequence: accept → undo → modify → accept next
    sequence: list[str] = [
        "accepted first",
        label.UNDO_REQUESTED,
        "modified first",
        "accepted second",
    ]

    call_index = 0
    observed_final_alts: list[str | None] = []

    def mock_process_single_suggestion(
        suggestion_data, display, current=None, total=None
    ):
        nonlocal call_index
        # Record the final_alt that arrives as prefill for this prompt
        observed_final_alts.append(suggestion_data.final_alt)

        final = (
            sequence[call_index]
            if call_index < len(sequence)
            else "accepted tail"
        )
        call_index += 1
        return create_alt(suggestion_data.line_number, final_alt=final)

    with patch.object(
        label,
        "_process_single_suggestion_for_labeling",
        side_effect=mock_process_single_suggestion,
    ):
        label.label_suggestions(
            suggestions, console, output_path, append_mode=False
        )

    # First prompt: no prefill; re-prompt after undo: prefilled with prior accepted text
    assert [observed_final_alts[0], observed_final_alts[2]] == [
        None,
        "accepted first",
    ]
