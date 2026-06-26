"""Tests for label.py module."""

import json
import subprocess
from contextlib import ExitStack, contextmanager
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

    def test_render_image_uses_imgcat(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        """render() on an image plan displays it via show_image."""
        image = temp_dir / "asset.jpg"
        image.write_bytes(b"data")
        plan = label.DisplayPlan(imgcat_path=image, external_path=None)

        with patch.object(label.DisplayManager, "show_image") as mock_image:
            display_manager.render(plan)

        mock_image.assert_called_once_with(image)

    def test_render_video_inline(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        """render() shows the preview GIF and does not open externally."""
        gif = temp_dir / "preview.gif"
        gif.write_bytes(b"gif")
        plan = label.DisplayPlan(
            imgcat_path=gif, external_path=temp_dir / "clip.mp4"
        )

        with (
            patch.object(label.DisplayManager, "show_image") as mock_image,
            patch.object(label.DisplayManager, "_open_externally") as mock_open,
        ):
            display_manager.render(plan)

        mock_image.assert_called_once_with(gif)
        mock_open.assert_not_called()

    def test_render_falls_back_to_external_when_inline_fails(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        """If imgcat fails, render() opens the video externally."""
        gif = temp_dir / "preview.gif"
        video = temp_dir / "clip.mp4"
        plan = label.DisplayPlan(imgcat_path=gif, external_path=video)

        with (
            patch.object(
                label.DisplayManager,
                "show_image",
                side_effect=ValueError("tmux"),
            ),
            patch.object(label.DisplayManager, "_open_externally") as mock_open,
        ):
            display_manager.render(plan)

        mock_open.assert_called_once_with(video)

    def test_render_external_only(
        self, display_manager: label.DisplayManager, temp_dir: Path
    ) -> None:
        """A plan with no inline path opens externally directly."""
        video = temp_dir / "clip.mp4"
        plan = label.DisplayPlan(
            imgcat_path=None, external_path=video, note="tmux"
        )

        with (
            patch.object(label.DisplayManager, "show_image") as mock_image,
            patch.object(label.DisplayManager, "_open_externally") as mock_open,
        ):
            display_manager.render(plan)

        mock_image.assert_not_called()
        mock_open.assert_called_once_with(video)

    @pytest.mark.parametrize(
        "system,expected_prefix",
        [
            ("Darwin", ["open", "-g"]),
            ("Linux", ["xdg-open"]),
            ("Windows", ["cmd", "/c", "start", ""]),
        ],
    )
    def test_external_open_command_per_platform(
        self, system: str, expected_prefix: list[str]
    ) -> None:
        """The external opener is platform-appropriate and keeps terminal focus."""
        path = Path("/tmp/clip.mp4")
        with patch("platform.system", return_value=system):
            command = label.DisplayManager._external_open_command(path)
        assert command[: len(expected_prefix)] == expected_prefix
        assert command[-1] == str(path)

    def test_open_externally_is_non_blocking(
        self, display_manager: label.DisplayManager
    ) -> None:
        """_open_externally launches detached via Popen (does not block)."""
        video = Path("/tmp/clip.mp4")
        with (
            patch("platform.system", return_value="Linux"),
            patch("subprocess.Popen") as mock_popen,
        ):
            display_manager._open_externally(video)

        mock_popen.assert_called_once()
        assert mock_popen.call_args.args[0] == ["xdg-open", str(video)]


# ---------------------------------------------------------------------------
# Asset preparation (download + preview) and prefetching
# ---------------------------------------------------------------------------


class TestPrepareDisplay:
    """Test prepare_display, which does the heavy work off the main thread."""

    def _queue_item(self, asset_path: str, temp_dir: Path) -> scan.QueueItem:
        return scan.QueueItem(
            markdown_file=str(temp_dir / "test.md"),
            asset_path=asset_path,
            line_number=1,
            context_snippet="ctx",
        )

    def test_image_plan(self, temp_dir: Path) -> None:
        """An image is shown inline directly, with no external fallback."""
        item = self._queue_item("image.jpg", temp_dir)

        def fake_download(queue_item, workspace):
            f = workspace / "image.jpg"
            f.write_bytes(b"img")
            return f

        with patch.object(utils, "download_asset", side_effect=fake_download):
            plan = label.prepare_display(item, temp_dir)

        assert plan.imgcat_path is not None
        assert plan.imgcat_path.suffix == ".jpg"
        assert plan.external_path is None

    def test_video_plan_builds_preview(self, temp_dir: Path) -> None:
        """A video yields an inline GIF preview plus an external fallback."""
        item = self._queue_item("clip.mp4", temp_dir)

        def fake_download(queue_item, workspace):
            f = workspace / "clip.mp4"
            f.write_bytes(b"vid")
            return f

        with (
            patch.dict("os.environ", {}, clear=True),  # no TMUX
            patch.object(utils, "download_asset", side_effect=fake_download),
            patch.object(
                utils, "find_executable", return_value="/usr/bin/ffmpeg"
            ),
            patch("subprocess.run") as mock_run,
        ):
            plan = label.prepare_display(item, temp_dir)

        # ffmpeg was invoked to build the preview gif.
        assert any(
            call.args[0][0] == "/usr/bin/ffmpeg"
            for call in mock_run.call_args_list
        )
        assert plan.imgcat_path is not None
        assert plan.imgcat_path.suffix == ".gif"
        assert plan.external_path is not None
        assert plan.external_path.suffix == ".mp4"

    def test_video_plan_tmux_skips_conversion(self, temp_dir: Path) -> None:
        """Under tmux we don't convert; the plan opens the video externally."""
        item = self._queue_item("clip.mp4", temp_dir)

        def fake_download(queue_item, workspace):
            f = workspace / "clip.mp4"
            f.write_bytes(b"vid")
            return f

        with (
            patch.dict("os.environ", {"TMUX": "1"}),
            patch.object(utils, "download_asset", side_effect=fake_download),
            patch("subprocess.run") as mock_run,
        ):
            plan = label.prepare_display(item, temp_dir)

        mock_run.assert_not_called()
        assert plan.imgcat_path is None
        assert plan.external_path is not None
        assert plan.note and "tmux" in plan.note

    def test_video_plan_ffmpeg_missing_falls_back(self, temp_dir: Path) -> None:
        """Missing ffmpeg degrades to an external-open plan with a note."""
        item = self._queue_item("clip.mp4", temp_dir)

        def fake_download(queue_item, workspace):
            f = workspace / "clip.mp4"
            f.write_bytes(b"vid")
            return f

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(utils, "download_asset", side_effect=fake_download),
            patch.object(
                utils,
                "find_executable",
                side_effect=FileNotFoundError("no ffmpeg"),
            ),
        ):
            plan = label.prepare_display(item, temp_dir)

        assert plan.imgcat_path is None
        assert plan.external_path is not None
        assert plan.note is not None

    def test_download_error_propagates(self, temp_dir: Path) -> None:
        """Acquisition failures are not swallowed - they propagate to callers."""
        item = self._queue_item("missing.jpg", temp_dir)
        with patch.object(
            utils, "download_asset", side_effect=FileNotFoundError("gone")
        ):
            with pytest.raises(FileNotFoundError):
                label.prepare_display(item, temp_dir)


class TestAssetPrefetcher:
    """Test the background prefetch/buffering of upcoming assets."""

    def _suggestions(self, n: int) -> list[utils.AltGenerationResult]:
        return [create_alt(i + 1) for i in range(n)]

    def test_get_without_schedule_computes_synchronously(
        self, temp_dir: Path
    ) -> None:
        suggestions = self._suggestions(2)
        prefetcher = label.AssetPrefetcher(suggestions, temp_dir)

        sentinel = label.DisplayPlan(imgcat_path=None, external_path=None)
        with patch.object(
            label, "prepare_display", return_value=sentinel
        ) as mock_prepare:
            plan = prefetcher.get(0)

        assert plan is sentinel
        mock_prepare.assert_called_once()
        prefetcher.shutdown()

    def test_schedule_then_get_uses_cached_result(self, temp_dir: Path) -> None:
        """A scheduled item is prepared once; get() reuses that result."""
        suggestions = self._suggestions(2)
        prefetcher = label.AssetPrefetcher(suggestions, temp_dir)

        call_count = 0

        def fake_prepare(queue_item, workspace, progress=None):
            nonlocal call_count
            call_count += 1
            return label.DisplayPlan(imgcat_path=None, external_path=None)

        with patch.object(label, "prepare_display", side_effect=fake_prepare):
            prefetcher.schedule(1)
            first = prefetcher.get(1)
            second = prefetcher.get(1)  # cached future, no recompute

        assert first is second or first == second
        assert call_count == 1
        prefetcher.shutdown()

    def test_get_reraises_background_error(self, temp_dir: Path) -> None:
        """An error during background prep is re-raised at get() time."""
        suggestions = self._suggestions(1)
        prefetcher = label.AssetPrefetcher(suggestions, temp_dir)

        with patch.object(
            label, "prepare_display", side_effect=FileNotFoundError("boom")
        ):
            prefetcher.schedule(0)
            with pytest.raises(FileNotFoundError):
                prefetcher.get(0)
        prefetcher.shutdown()

    def test_schedule_out_of_range_is_noop(self, temp_dir: Path) -> None:
        suggestions = self._suggestions(1)
        prefetcher = label.AssetPrefetcher(suggestions, temp_dir)
        with patch.object(label, "prepare_display") as mock_prepare:
            prefetcher.schedule(5)
            prefetcher.schedule(-1)
        mock_prepare.assert_not_called()
        prefetcher.shutdown()

    def test_retain_window_discards_distant_items(self, temp_dir: Path) -> None:
        """Items far behind the cursor are dropped to bound disk use."""
        suggestions = self._suggestions(10)
        prefetcher = label.AssetPrefetcher(suggestions, temp_dir)

        def fake_prepare(queue_item, workspace, progress=None):
            return label.DisplayPlan(imgcat_path=None, external_path=None)

        with patch.object(label, "prepare_display", side_effect=fake_prepare):
            for i in range(6):
                prefetcher.schedule(i)
                prefetcher.get(i)
            prefetcher.retain_window(5)

        # KEEP_BEHIND=3, so item 0 (>3 behind index 5) is discarded.
        assert 0 not in prefetcher._futures
        assert 5 in prefetcher._futures
        prefetcher.shutdown()


@pytest.mark.parametrize(
    "error_type,error_on,expected_result_count,expected_saved,should_raise",
    [
        pytest.param(
            FileNotFoundError, "image2.jpg", 1, 1, False,
            id="file_error_graceful",
        ),
        pytest.param(
            KeyboardInterrupt, "image2.jpg", None, 1, False,
            id="keyboard_interrupt_saves",
        ),
        pytest.param(
            RuntimeError, "image1.jpg", None, 0, True,
            id="runtime_error_propagates",
        ),
    ],
)
def test_label_suggestions_error_handling(
    temp_dir: Path,
    test_suggestions: list[utils.AltGenerationResult],
    error_type: type,
    error_on: str,
    expected_result_count: int | None,
    expected_saved: int,
    should_raise: bool,
) -> None:
    """Test error handling during labeling: graceful recovery or propagation."""
    output_file = temp_dir / "test_output.json"
    with _setup_error_mocks(error_type, error_on):
        if should_raise:
            with pytest.raises(error_type):
                label.label_suggestions(
                    test_suggestions, Console(), output_file, skip_existing=False
                )
        else:
            result_count = label.label_suggestions(
                test_suggestions, Console(), output_file, skip_existing=False
            )
            if expected_result_count is not None:
                assert result_count == expected_result_count
    _maybe_assert_saved_results(output_file, expected_saved)


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
        # Invalid JSON and missing files now produce friendly errors and exit
        # non-zero via SystemExit (mirroring apply.py behavior).
        (SystemExit, "invalid json"),
        (SystemExit, None),  # File doesn't exist
        (
            TypeError,
            '[{"markdown_file": "test.md"}]',
        ),  # Missing required fields (unguarded, raw exception)
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


def test_label_from_suggestions_file_friendly_errors_exit_nonzero(
    temp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Missing file and bad JSON print a red error and exit non-zero."""
    missing = temp_dir / "missing.json"
    with pytest.raises(SystemExit) as exc_info:
        label.label_from_suggestions_file(
            missing, temp_dir / "output.json", skip_existing=False
        )
    assert exc_info.value.code != 0
    assert "not found" in capsys.readouterr().out

    bad = temp_dir / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        label.label_from_suggestions_file(
            bad, temp_dir / "output.json", skip_existing=False
        )
    assert exc_info.value.code != 0
    assert "Invalid JSON" in capsys.readouterr().out


@pytest.mark.parametrize("user_input", ["undo", "u", "UNDO"])
def test_prompt_for_edit_undo_command(user_input: str) -> None:
    """prompt_for_edit returns sentinel on various undo inputs."""
    console = Console()
    display = label.DisplayManager(console)

    with patch("alt_text_llm.label.prompt", return_value=user_input):
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
        suggestion_data, display, prefetcher=None, index=None, current=None, total=None
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
            suggestions, console, output_path, skip_existing=True
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
        suggestion_data, display, prefetcher=None, index=None, current=None, total=None
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
            suggestions, console, output_path, skip_existing=False
        )

    # First prompt: no prefill; re-prompt after undo: prefilled with prior accepted text
    assert [observed_final_alts[0], observed_final_alts[2]] == [
        None,
        "accepted first",
    ]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_labeling_session_empty_suggestions() -> None:
    """LabelingSession with no suggestions should be immediately complete."""
    session = label.LabelingSession([])
    assert session.is_complete()
    assert session.get_current_suggestion() is None
    assert session.get_progress() == (1, 0)


def test_labeling_session_multiple_undos() -> None:
    """Multiple consecutive undos should work correctly."""
    suggestions = [create_alt(i) for i in range(1, 4)]
    session = label.LabelingSession(suggestions)

    for s in suggestions:
        session.add_result(create_alt(s.line_number, final_alt=f"final_{s.line_number}"))

    assert session.is_complete()

    undone3 = session.undo()
    assert undone3 is not None
    assert undone3.final_alt == "final_3"

    undone2 = session.undo()
    assert undone2 is not None
    assert undone2.final_alt == "final_2"

    undone1 = session.undo()
    assert undone1 is not None
    assert undone1.final_alt == "final_1"

    assert session.undo() is None
    assert session.current_index == 0


def test_label_suggestions_empty_list(temp_dir: Path) -> None:
    """Labeling with empty suggestions should return 0."""
    from io import StringIO

    output = temp_dir / "output.json"
    result = label.label_suggestions(
        [], Console(file=StringIO()), output, skip_existing=False
    )
    assert result == 0


# ---------------------------------------------------------------------------
# New UX / robustness behaviors
# ---------------------------------------------------------------------------


def test_show_progress_reaches_100_and_guards_zero() -> None:
    """Progress hits 100% on the last item and never divides by zero."""
    from io import StringIO

    out = StringIO()
    display = label.DisplayManager(Console(file=out))

    display.show_progress(1, 3)
    display.show_progress(3, 3)
    display.show_progress(1, 0)  # must not raise ZeroDivisionError

    text = out.getvalue()
    assert "1/3 (33.3%)" in text
    assert "3/3 (100.0%)" in text
    assert "1/0 (0.0%)" in text


@pytest.mark.parametrize(
    "user_input,sentinel",
    [
        ("q", label.QUIT_REQUESTED),
        ("quit", label.QUIT_REQUESTED),
        ("s", label.SKIP_REQUESTED),
        ("skip", label.SKIP_REQUESTED),
    ],
)
def test_prompt_for_edit_quit_skip_commands(
    user_input: str, sentinel: str
) -> None:
    """prompt_for_edit returns quit/skip sentinels."""
    display = label.DisplayManager(Console())
    with patch("alt_text_llm.label.prompt", return_value=user_input):
        assert display.prompt_for_edit("suggestion") == sentinel


def test_prompt_for_edit_help_reprompts() -> None:
    """help/? prints commands and re-prompts, then returns the accepted text."""
    from io import StringIO

    out = StringIO()
    display = label.DisplayManager(Console(file=out))
    responses = iter(["?", "final text"])

    with patch(
        "alt_text_llm.label.prompt", side_effect=lambda *a, **k: next(responses)
    ):
        result = display.prompt_for_edit("suggestion")

    assert result == "final text"
    assert "Commands:" in out.getvalue()


def _label_with_single_input(
    suggestions, output_path, user_input, *, mock_show_image=True
):
    """Run label_suggestions through a real prompt that returns user_input.

    Downloads are mocked and the prompt returns user_input. show_image is
    mocked by default; pass mock_show_image=False to exercise the real method.
    """
    def mock_download_asset(queue_item, workspace):
        test_file = workspace / "test.jpg"
        test_file.write_bytes(b"fake image")
        return test_file

    patches = [
        patch("sys.stdout.isatty", return_value=True),
        patch.object(utils, "download_asset", side_effect=mock_download_asset),
        patch.object(label.DisplayManager, "show_context"),
        patch.object(label.DisplayManager, "show_rule"),
        patch("alt_text_llm.label.prompt", return_value=user_input),
    ]
    if mock_show_image:
        patches.append(patch.object(label.DisplayManager, "show_image"))

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        return label.label_suggestions(
            suggestions, Console(), output_path, skip_existing=False
        )


def test_quit_command_stops_loop_and_saves(temp_dir: Path) -> None:
    """A quit command saves progress for prior items and stops labeling."""
    output_path = temp_dir / "out.json"
    suggestions = [create_alt(1), create_alt(2), create_alt(3)]

    # Accept the first item, then quit on the second.
    responses = iter(["accepted one", "q", "should not reach"])

    def mock_download_asset(queue_item, workspace):
        f = workspace / "test.jpg"
        f.write_bytes(b"fake")
        return f

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch.object(utils, "download_asset", side_effect=mock_download_asset),
        patch.object(label.DisplayManager, "show_context"),
        patch.object(label.DisplayManager, "show_rule"),
        patch.object(label.DisplayManager, "show_image"),
        patch(
            "alt_text_llm.label.prompt",
            side_effect=lambda *a, **k: next(responses),
        ),
    ):
        count = label.label_suggestions(
            suggestions, Console(), output_path, skip_existing=False
        )

    assert count == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["final_alt"] for r in saved] == ["accepted one"]


def test_skip_command_advances_without_recording(temp_dir: Path) -> None:
    """A skip command advances without recording the item."""
    output_path = temp_dir / "out.json"
    suggestions = [create_alt(1), create_alt(2)]

    responses = iter(["s", "kept two"])

    def mock_download_asset(queue_item, workspace):
        f = workspace / "test.jpg"
        f.write_bytes(b"fake")
        return f

    with (
        patch("sys.stdout.isatty", return_value=True),
        patch.object(utils, "download_asset", side_effect=mock_download_asset),
        patch.object(label.DisplayManager, "show_context"),
        patch.object(label.DisplayManager, "show_rule"),
        patch.object(label.DisplayManager, "show_image"),
        patch(
            "alt_text_llm.label.prompt",
            side_effect=lambda *a, **k: next(responses),
        ),
    ):
        count = label.label_suggestions(
            suggestions, Console(), output_path, skip_existing=False
        )

    assert count == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert [r["final_alt"] for r in saved] == ["kept two"]


def test_image_display_failure_does_not_crash_labeling(temp_dir: Path) -> None:
    """In tmux (show_image raises ValueError) labeling still records captions."""
    output_path = temp_dir / "out.json"
    suggestions = [create_alt(1)]

    # Real show_image is used; TMUX env makes it raise ValueError.
    with patch.dict("os.environ", {"TMUX": "1"}):
        count = _label_with_single_input(
            suggestions,
            output_path,
            "edited despite tmux",
            mock_show_image=False,
        )

    assert count == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved[0]["final_alt"] == "edited despite tmux"


def test_non_interactive_warns_and_autoaccepts(temp_dir: Path) -> None:
    """Non-tty labeling warns once and accepts the suggestion as-is."""
    from io import StringIO

    output_path = temp_dir / "out.json"
    suggestions = [create_alt(1)]
    out = StringIO()

    def mock_download_asset(queue_item, workspace):
        f = workspace / "test.jpg"
        f.write_bytes(b"fake")
        return f

    with (
        patch("sys.stdout.isatty", return_value=False),
        patch.object(utils, "download_asset", side_effect=mock_download_asset),
        patch.object(label.DisplayManager, "show_context"),
        patch.object(label.DisplayManager, "show_rule"),
        patch.object(label.DisplayManager, "show_image"),
    ):
        count = label.label_suggestions(
            suggestions, Console(file=out), output_path, skip_existing=False
        )

    assert count == 1
    assert "Non-interactive terminal" in out.getvalue()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    # suggested_alt for create_alt(1) is "suggestion 1"
    assert saved[0]["final_alt"] == "suggestion 1"


def test_no_skip_existing_relabel_preserves_prior_captions(
    temp_dir: Path,
) -> None:
    """--no-skip-existing relabel must never overwrite prior captions.

    Pre-populate the output file with a prior caption, then run a labeling
    session with skip_existing=False (the --no-skip-existing path). The prior
    caption must still be present afterwards (appended, not overwritten).
    """
    output_path = temp_dir / "captions.json"
    prior = create_alt(99, final_alt="prior caption that must survive")
    utils.write_output([prior], output_path, append_mode=False)

    suggestions = [create_alt(1)]
    count = _label_with_single_input(
        suggestions, output_path, "newly labeled"
    )

    assert count == 1
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    final_alts = [r["final_alt"] for r in saved]
    assert "prior caption that must survive" in final_alts
    assert "newly labeled" in final_alts
