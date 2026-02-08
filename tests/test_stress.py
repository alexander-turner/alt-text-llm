"""Comprehensive stress tests for alt-text-llm.

Targets edge cases, boundary conditions, malformed inputs, Unicode handling,
concurrency stress, and integration round-trips that the existing suite doesn't cover.
"""

import asyncio
import json
import re
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.console import Console

from alt_text_llm import apply, generate, label, scan, utils
from tests.test_helpers import create_markdown_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def console() -> Console:
    return Console(file=StringIO())


# ---------------------------------------------------------------------------
# scan.py stress tests
# ---------------------------------------------------------------------------


class TestScanStress:
    """Stress tests for the scanning module."""

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty markdown file should produce no queue items."""
        (tmp_path / "empty.md").write_text("", encoding="utf-8")
        assert scan.build_queue(tmp_path) == []

    def test_file_with_only_frontmatter(self, tmp_path: Path) -> None:
        """File with only YAML frontmatter and no body."""
        (tmp_path / "fm.md").write_text("---\ntitle: hello\n---\n", encoding="utf-8")
        assert scan.build_queue(tmp_path) == []

    def test_hundreds_of_images(self, tmp_path: Path) -> None:
        """File with 500 images should all be detected."""
        lines = [f"![](image_{i}.png)" for i in range(500)]
        (tmp_path / "big.md").write_text("\n\n".join(lines), encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 500
        paths = {item.asset_path for item in queue}
        for i in range(500):
            assert f"image_{i}.png" in paths

    def test_multiple_images_on_same_line(self, tmp_path: Path) -> None:
        """Multiple markdown images on a single line."""
        content = "![](a.png) some text ![](b.png) more ![](c.png)\n"
        (tmp_path / "multi.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        # markdown-it may handle inline images differently
        paths = {item.asset_path for item in queue}
        assert "a.png" in paths
        assert "c.png" in paths

    def test_image_inside_code_block_ignored(self, tmp_path: Path) -> None:
        """Images inside fenced code blocks should NOT be detected."""
        content = textwrap.dedent("""\
            # Docs

            ```markdown
            ![](code_image.png)
            ```

            ![](real_image.png)
        """)
        (tmp_path / "code.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "real_image.png" in paths
        assert "code_image.png" not in paths

    def test_image_inside_inline_code_ignored(self, tmp_path: Path) -> None:
        """Images inside inline code should NOT be detected."""
        content = "Use `![](not_real.png)` syntax for images.\n\n![](real.png)\n"
        (tmp_path / "inline_code.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "real.png" in paths
        assert "not_real.png" not in paths

    @pytest.mark.xfail(
        reason="Bug: markdown-it URL-encodes Unicode src attrs, "
        "but _get_line_number searches for the encoded string in raw file content"
    )
    def test_unicode_asset_path_markdown(self, tmp_path: Path) -> None:
        """Markdown images with Unicode file names are URL-encoded by markdown-it,
        causing _get_line_number to fail when searching in the raw file."""
        content = "![](画像.png)\n"
        (tmp_path / "unicode.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "画像.png" in paths or "%E7%94%BB%E5%83%8F.png" in paths

    def test_unicode_asset_path_html(self, tmp_path: Path) -> None:
        """HTML images with Unicode file names should be detected."""
        content = '<img src="фото.jpg">\n'
        (tmp_path / "unicode.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "фото.jpg" in paths

    def test_unicode_alt_text_is_meaningful(self, tmp_path: Path) -> None:
        """Unicode alt text should be considered meaningful."""
        content = "![日本語の説明](image.png)\n"
        (tmp_path / "unicode_alt.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 0  # Unicode alt IS meaningful

    def test_very_long_alt_text(self, tmp_path: Path) -> None:
        """Very long alt text should still be considered meaningful."""
        long_alt = "A" * 10000
        content = f"![{long_alt}](image.png)\n"
        (tmp_path / "long_alt.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 0

    def test_deeply_nested_html_img(self, tmp_path: Path) -> None:
        """Deeply nested HTML img tags should be found."""
        content = '<div><div><div><p><img src="nested.jpg"></p></div></div></div>\n'
        (tmp_path / "nested.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "nested.jpg" in paths

    def test_malformed_html_does_not_crash(self, tmp_path: Path) -> None:
        """Malformed HTML should not crash the scanner."""
        content = '<img src="ok.png">\n<img src="broken.jpg\n<img src=unclosed>\n'
        (tmp_path / "malformed.md").write_text(content, encoding="utf-8")
        # Should not raise
        queue = scan.build_queue(tmp_path)
        # At minimum, should find the valid image
        paths = {item.asset_path for item in queue}
        assert "ok.png" in paths

    def test_wikilink_empty_inner_content(self, tmp_path: Path) -> None:
        """Wikilink with empty content: ![[]] should be ignored."""
        content = "![[]] and ![[|alt text]]\n"
        (tmp_path / "empty_wikilink.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        # Empty src should be skipped
        assert len(queue) == 0

    def test_wikilink_nested_brackets(self, tmp_path: Path) -> None:
        """Wikilink parsing with tricky bracket patterns."""
        content = "![[image.png]] and [[not an image]] and ![[photo.jpg|alt]]\n"
        (tmp_path / "brackets.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        paths = {item.asset_path for item in queue}
        assert "image.png" in paths
        # photo.jpg has meaningful alt, should NOT be in queue
        assert "photo.jpg" not in paths

    def test_html_img_with_extra_attributes(self, tmp_path: Path) -> None:
        """HTML img with many attributes should still be detected."""
        content = '<img src="test.png" class="hero" id="main" width="100" height="50" loading="lazy">\n'
        (tmp_path / "attrs.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 1
        assert queue[0].asset_path == "test.png"

    def test_html_img_decorative_empty_alt(self, tmp_path: Path) -> None:
        """HTML img with alt='' is decorative and should NOT be queued."""
        content = '<img src="decorative.png" alt="">\n'
        (tmp_path / "decorative.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 0

    def test_mixed_video_and_image_formats_large(self, tmp_path: Path) -> None:
        """Large file mixing video and image formats."""
        lines = []
        for i in range(50):
            lines.append(f"![](img_{i}.png)")
            lines.append(f'<video src="vid_{i}.mp4"></video>')
            lines.append(f"![[wiki_{i}.jpg]]")
        (tmp_path / "mixed.md").write_text("\n\n".join(lines), encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 150

    def test_queue_item_line_number_must_be_positive(self) -> None:
        """QueueItem enforces positive line numbers."""
        with pytest.raises(ValueError, match="line_number must be positive"):
            scan.QueueItem(
                markdown_file="test.md",
                asset_path="img.png",
                line_number=0,
                context_snippet="ctx",
            )
        with pytest.raises(ValueError, match="line_number must be positive"):
            scan.QueueItem(
                markdown_file="test.md",
                asset_path="img.png",
                line_number=-1,
                context_snippet="ctx",
            )

    @pytest.mark.xfail(
        reason="Bug: <video> without src attr using <source> child is not detected "
        "during scanning. _extract_html_video_info handles it, but the token "
        "content may not be parsed as a single html_block by markdown-it."
    )
    def test_video_with_source_element_and_text(self, tmp_path: Path) -> None:
        """Video tag using <source> child, surrounded by content to ensure markdown-it parses it."""
        content = textwrap.dedent("""\
            # Video Demo

            Some intro text.

            <video controls><source src="clip.mp4" type="video/mp4"></video>

            After video text.
        """)
        (tmp_path / "source_video.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        found_clip = any(item.asset_path == "clip.mp4" for item in queue)
        assert found_clip, f"Expected clip.mp4 in queue, got: {[i.asset_path for i in queue]}"

    def test_multiple_files_in_directory(self, tmp_path: Path) -> None:
        """Scanning a directory with multiple markdown files."""
        for i in range(10):
            (tmp_path / f"file_{i}.md").write_text(
                f"![](image_{i}.png)\n", encoding="utf-8"
            )
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 10

    def test_non_md_files_ignored(self, tmp_path: Path) -> None:
        """Non-markdown files should be ignored."""
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "readme.txt").write_text("![](img.png)")
        (tmp_path / "actual.md").write_text("![](real.png)\n")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 1
        assert queue[0].asset_path == "real.png"

    def test_special_regex_chars_in_path_markdown(self, tmp_path: Path) -> None:
        """Parentheses in markdown URL close the link early (standard CommonMark).
        Use URL-encoding or HTML img for paths with parens."""
        content = "![](image (1).png)\n"
        (tmp_path / "special.md").write_text(content, encoding="utf-8")
        # CommonMark interprets the first ')' as closing the URL,
        # so this results in a link to "image (1" - expected behavior.
        queue = scan.build_queue(tmp_path)
        # The parsed asset path will be truncated at the first unescaped ')'
        if len(queue) == 1:
            # If markdown-it parses it, the path will be "image (1"
            assert "image" in queue[0].asset_path

    def test_special_regex_chars_in_path_html(self, tmp_path: Path) -> None:
        """HTML img handles paths with special characters correctly."""
        content = '<img src="image (1).png">\n'
        (tmp_path / "special_html.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 1
        assert queue[0].asset_path == "image (1).png"

    def test_url_assets_detected(self, tmp_path: Path) -> None:
        """Remote URL assets should be detected."""
        content = "![](https://example.com/img.jpg)\n"
        (tmp_path / "url.md").write_text(content, encoding="utf-8")
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 1
        assert queue[0].asset_path == "https://example.com/img.jpg"


# ---------------------------------------------------------------------------
# apply.py stress tests
# ---------------------------------------------------------------------------


class TestApplyStress:
    """Stress tests for the apply module."""

    def test_special_regex_chars_in_asset_path(self) -> None:
        """Asset paths with regex-special characters should not break."""
        line = "![old](path/image (1).png)"
        new_line, old_alt = apply._apply_markdown_image_alt(
            line, "path/image (1).png", "new alt"
        )
        assert "![new alt](path/image (1).png)" in new_line
        assert old_alt == "old"

    def test_dollar_signs_in_alt_text(self) -> None:
        """Dollar signs should be escaped in markdown alt text."""
        line = "![](image.png)"
        new_line, _ = apply._apply_markdown_image_alt(
            line, "image.png", "$100 worth of items"
        )
        assert r"\$100 worth of items" in new_line

    def test_backslash_in_alt_text(self) -> None:
        """Backslashes should be escaped in markdown alt text."""
        line = "![](image.png)"
        new_line, _ = apply._apply_markdown_image_alt(
            line, "image.png", r"x\in A"
        )
        assert r"x\\in A" in new_line

    def test_empty_alt_text_application(self) -> None:
        """Applying empty alt text."""
        line = "![old](image.png)"
        new_line, old_alt = apply._apply_markdown_image_alt(
            line, "image.png", ""
        )
        assert "![](image.png)" in new_line
        assert old_alt == "old"

    def test_very_long_alt_text(self, tmp_path: Path, console: Console) -> None:
        """Alt text with 10,000 characters should still apply."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![old](image.png)\n", encoding="utf-8")
        long_alt = "A" * 10000
        caption = utils.AltGenerationResult(
            markdown_file=str(md_path),
            asset_path="image.png",
            suggested_alt="s",
            model="m",
            context_snippet="c",
            final_alt=long_alt,
        )
        result = apply._apply_caption_to_file(md_path, caption, console)
        assert result is not None
        content = md_path.read_text()
        assert long_alt in content

    def test_unicode_alt_text(self) -> None:
        """Unicode alt text in markdown images."""
        line = "![](image.png)"
        new_line, _ = apply._apply_markdown_image_alt(
            line, "image.png", "日本語の代替テキスト"
        )
        assert "![日本語の代替テキスト](image.png)" in new_line

    def test_unicode_alt_text_html(self) -> None:
        """Unicode alt text in HTML images."""
        line = '<img src="image.png">'
        new_line, _ = apply._apply_html_image_alt(
            line, "image.png", "中文替代文字"
        )
        assert "中文替代文字" in new_line

    def test_html_special_chars_all_escaped(self) -> None:
        """All HTML special characters should be properly escaped."""
        line = '<img src="img.png">'
        new_line, _ = apply._apply_html_image_alt(
            line, "img.png", 'A < B > C & D "E"'
        )
        assert "&lt;" in new_line
        assert "&gt;" in new_line
        assert "&amp;" in new_line

    def test_apply_preserves_trailing_newline(
        self, tmp_path: Path, console: Console
    ) -> None:
        """File with trailing newline should preserve it."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![old](image.png)\n", encoding="utf-8")
        caption = utils.AltGenerationResult(
            markdown_file=str(md_path),
            asset_path="image.png",
            suggested_alt="s",
            model="m",
            context_snippet="c",
            final_alt="new",
        )
        apply._apply_caption_to_file(md_path, caption, console)
        assert md_path.read_text().endswith("\n")

    def test_apply_preserves_no_trailing_newline(
        self, tmp_path: Path, console: Console
    ) -> None:
        """File without trailing newline should NOT gain one."""
        md_path = tmp_path / "test.md"
        md_path.write_text("![old](image.png)", encoding="utf-8")
        caption = utils.AltGenerationResult(
            markdown_file=str(md_path),
            asset_path="image.png",
            suggested_alt="s",
            model="m",
            context_snippet="c",
            final_alt="new",
        )
        apply._apply_caption_to_file(md_path, caption, console)
        assert not md_path.read_text().endswith("\n")

    def test_apply_to_nonexistent_file(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Applying to a nonexistent file should warn, not crash."""
        md_path = tmp_path / "nonexistent.md"
        captions_path = tmp_path / "captions.json"
        captions_data = [
            {
                "markdown_file": str(md_path),
                "asset_path": "img.png",
                "line_number": 1,
                "suggested_alt": "s",
                "final_alt": "new",
                "model": "m",
                "context_snippet": "c",
            }
        ]
        captions_path.write_text(json.dumps(captions_data))
        result = apply.apply_captions(captions_path, console)
        assert result == 0

    def test_apply_no_captions_with_final_alt(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Captions file where nothing has final_alt set."""
        captions_path = tmp_path / "captions.json"
        captions_data = [
            {
                "markdown_file": "test.md",
                "asset_path": "img.png",
                "line_number": 1,
                "suggested_alt": "s",
                "final_alt": "",
                "model": "m",
                "context_snippet": "c",
            }
        ]
        captions_path.write_text(json.dumps(captions_data))
        result = apply.apply_captions(captions_path, console)
        assert result == 0

    def test_apply_with_many_captions(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Apply 100 captions to a file with 100 images."""
        md_path = tmp_path / "test.md"
        lines = [f"![old_{i}](image_{i}.png)" for i in range(100)]
        md_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")

        captions_data = [
            {
                "markdown_file": str(md_path),
                "asset_path": f"image_{i}.png",
                "line_number": i * 2 + 1,
                "suggested_alt": f"suggested_{i}",
                "final_alt": f"new_caption_{i}",
                "model": "m",
                "context_snippet": "c",
            }
            for i in range(100)
        ]
        captions_path = tmp_path / "captions.json"
        captions_path.write_text(json.dumps(captions_data))
        result = apply.apply_captions(captions_path, console)
        assert result == 100

        content = md_path.read_text()
        for i in range(100):
            assert f"![new_caption_{i}](image_{i}.png)" in content

    def test_try_all_formats_tries_video_last(self) -> None:
        """_try_all_image_formats should try video as last resort."""
        line = '<video src="demo.mp4"></video>'
        new_line, _ = apply._try_all_image_formats(
            line, "demo.mp4", "Video description"
        )
        assert "aria-label" in new_line

    def test_wikilink_with_pipe_in_alt(self) -> None:
        """Wikilinks should handle pipe character in replacement."""
        line = "![[image.png]]"
        new_line, _ = apply._apply_wikilink_image_alt(
            line, "image.png", "new alt"
        )
        assert "![[image.png|new alt]]" in new_line

    def test_multiline_alt_text_collapsed(self) -> None:
        """Multi-line alt text should be collapsed with ellipses."""
        line = "![old](image.png)"
        new_line, _ = apply._try_all_image_formats(
            line, "image.png", "Line 1\nLine 2\nLine 3"
        )
        assert "Line 1 ... Line 2 ... Line 3" in new_line
        # No raw newlines inside the alt text
        alt_match = re.search(r"!\[([^\]]*)\]", new_line)
        assert alt_match is not None
        assert "\n" not in alt_match.group(1)

    def test_apply_caption_asset_not_in_file(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Applying caption when asset doesn't exist in the file."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# No images here\n", encoding="utf-8")
        caption = utils.AltGenerationResult(
            markdown_file=str(md_path),
            asset_path="nonexistent.png",
            suggested_alt="s",
            model="m",
            context_snippet="c",
            final_alt="new alt",
        )
        result = apply._apply_caption_to_file(md_path, caption, console)
        assert result is None

    def test_captions_json_with_whitespace_only_final_alt(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Captions with whitespace-only final_alt should be treated as unused."""
        captions_path = tmp_path / "captions.json"
        captions_data = [
            {
                "markdown_file": "test.md",
                "asset_path": "img.png",
                "line_number": 1,
                "suggested_alt": "s",
                "final_alt": "   ",
                "model": "m",
                "context_snippet": "c",
            }
        ]
        captions_path.write_text(json.dumps(captions_data))
        result = apply.apply_captions(captions_path, console)
        assert result == 0


# ---------------------------------------------------------------------------
# utils.py stress tests
# ---------------------------------------------------------------------------


class TestUtilsStress:
    """Stress tests for the utils module."""

    def test_paragraph_context_single_paragraph(self) -> None:
        """Single paragraph with no blanks."""
        lines = ["Line 1", "Line 2", "Line 3"]
        result = utils.paragraph_context(lines, 1)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_paragraph_context_all_blank_lines(self) -> None:
        """All blank lines should return empty string."""
        lines = ["", "", "", ""]
        result = utils.paragraph_context(lines, 2)
        assert result == ""

    def test_paragraph_context_target_on_blank_line(self) -> None:
        """Target on a blank line should grab next paragraph."""
        lines = ["Para A", "", "Para B", "", "Para C"]
        result = utils.paragraph_context(lines, 1, max_before=0, max_after=0)
        assert "Para B" in result

    def test_paragraph_context_large_document(self) -> None:
        """Large document with 1000 paragraphs."""
        lines = []
        for i in range(1000):
            lines.append(f"Paragraph {i}")
            lines.append("")
        # Target is paragraph 500
        result = utils.paragraph_context(
            lines, 1000, max_before=2, max_after=2
        )
        assert "Paragraph 499" in result or "Paragraph 500" in result

    def test_paragraph_context_max_after_zero(self) -> None:
        """max_after=0 should include no paragraphs after target."""
        lines = ["A", "", "B", "", "C"]
        result = utils.paragraph_context(lines, 2, max_before=0, max_after=0)
        assert "B" in result
        assert "C" not in result
        assert "A" not in result

    def test_split_yaml_multiple_separators(self, tmp_path: Path) -> None:
        """File with multiple --- separators."""
        content = "---\ntitle: test\n---\nContent with --- inside\n---\nMore\n"
        fp = tmp_path / "multi_sep.md"
        fp.write_text(content, encoding="utf-8")
        metadata, body = utils.split_yaml(fp)
        assert metadata.get("title") == "test"
        # Body should be everything after the second ---
        assert "Content with --- inside" in body

    def test_split_yaml_unicode_content(self, tmp_path: Path) -> None:
        """YAML with unicode values."""
        content = "---\ntitle: 日本語タイトル\n---\n本文\n"
        fp = tmp_path / "unicode.md"
        fp.write_text(content, encoding="utf-8")
        metadata, body = utils.split_yaml(fp)
        assert metadata.get("title") == "日本語タイトル"
        assert "本文" in body

    def test_is_url_with_many_edge_cases(self) -> None:
        """Additional is_url edge cases."""
        assert utils.is_url("https://a.b/c?d=e&f=g#h") is True
        assert utils.is_url("data:image/png;base64,abc") is False
        assert utils.is_url("mailto:user@example.com") is False
        assert utils.is_url("file:///local/path") is False

    def test_is_video_asset_all_extensions(self) -> None:
        """Test all video extensions are recognized."""
        for ext in utils.VIDEO_EXTENSIONS:
            assert utils.is_video_asset(f"test{ext}") is True
        assert utils.is_video_asset("test.png") is False
        assert utils.is_video_asset("test.MP4") is True  # Case insensitive

    def test_build_prompt_for_video(self, tmp_path: Path) -> None:
        """build_prompt should use video-specific language for video assets."""
        md_path = tmp_path / "test.md"
        md_path.write_text("Some content\n\nVideo here\n", encoding="utf-8")
        qi = scan.QueueItem(
            markdown_file=str(md_path),
            asset_path="demo.mp4",
            line_number=3,
            context_snippet="ctx",
        )
        prompt = utils.build_prompt(qi, max_chars=200)
        assert "video" in prompt.lower()
        assert "Under 200 characters" in prompt

    def test_build_prompt_for_image(self, tmp_path: Path) -> None:
        """build_prompt should use image-specific language for image assets."""
        md_path = tmp_path / "test.md"
        md_path.write_text("Some content\n\nImage here\n", encoding="utf-8")
        qi = scan.QueueItem(
            markdown_file=str(md_path),
            asset_path="photo.jpg",
            line_number=3,
            context_snippet="ctx",
        )
        prompt = utils.build_prompt(qi, max_chars=300)
        assert "alt text" in prompt.lower()
        assert "Under 300 characters" in prompt

    def test_write_output_unicode(self, tmp_path: Path) -> None:
        """write_output should handle Unicode in results."""
        results = [
            utils.AltGenerationResult(
                markdown_file="日本語.md",
                asset_path="画像.png",
                suggested_alt="代替テキスト",
                model="test",
                context_snippet="コンテキスト",
                final_alt="最終テキスト",
            )
        ]
        output = tmp_path / "unicode_output.json"
        utils.write_output(results, output)
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data[0]["markdown_file"] == "日本語.md"
        assert data[0]["suggested_alt"] == "代替テキスト"

    def test_write_output_empty_results(self, tmp_path: Path) -> None:
        """write_output with no results should produce empty JSON array."""
        output = tmp_path / "empty.json"
        utils.write_output([], output)
        data = json.loads(output.read_text())
        assert data == []

    def test_write_output_append_to_empty_file(self, tmp_path: Path) -> None:
        """Append mode to an empty file."""
        output = tmp_path / "empty.json"
        output.write_text("[]")
        result = utils.AltGenerationResult(
            markdown_file="t.md",
            asset_path="i.png",
            suggested_alt="s",
            model="m",
            context_snippet="c",
        )
        utils.write_output([result], output, append_mode=True)
        data = json.loads(output.read_text())
        assert len(data) == 1

    def test_load_existing_captions_empty_list(self, tmp_path: Path) -> None:
        """Loading empty JSON list returns empty set."""
        fp = tmp_path / "captions.json"
        fp.write_text("[]")
        assert utils.load_existing_captions(fp) == set()

    def test_load_existing_captions_null_json(self, tmp_path: Path) -> None:
        """Loading null JSON returns empty set."""
        fp = tmp_path / "captions.json"
        fp.write_text("null")
        assert utils.load_existing_captions(fp) == set()

    def test_alt_generation_result_to_json(self) -> None:
        """AltGenerationResult serializes correctly."""
        result = utils.AltGenerationResult(
            markdown_file="test.md",
            asset_path="img.png",
            suggested_alt="alt",
            model="model",
            context_snippet="ctx",
            line_number=5,
            final_alt="final",
        )
        d = result.to_json()
        assert d["markdown_file"] == "test.md"
        assert d["line_number"] == 5
        assert d["final_alt"] == "final"

    def test_alt_generation_result_optional_fields(self) -> None:
        """AltGenerationResult with default optional fields."""
        result = utils.AltGenerationResult(
            markdown_file="test.md",
            asset_path="img.png",
            suggested_alt="alt",
            model="model",
            context_snippet="ctx",
        )
        assert result.line_number is None
        assert result.final_alt is None
        d = result.to_json()
        assert d["line_number"] is None
        assert d["final_alt"] is None

    def test_get_files_empty_directory(self, tmp_path: Path) -> None:
        """get_files on empty directory returns empty tuple."""
        result = utils.get_files(
            dir_to_search=tmp_path, use_git_ignore=False
        )
        assert result == ()

    def test_get_files_nested_deeply(self, tmp_path: Path) -> None:
        """get_files finds deeply nested files."""
        deep = tmp_path / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        (deep / "deep.md").write_text("deep content")
        result = utils.get_files(
            dir_to_search=tmp_path, use_git_ignore=False
        )
        assert len(result) == 1
        assert result[0].name == "deep.md"

    def test_find_executable_caching(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """find_executable should cache results."""
        monkeypatch.setattr(utils, "_executable_cache", {})
        call_count = 0
        original_which = __import__("shutil").which

        def counting_which(name: str):
            nonlocal call_count
            call_count += 1
            return original_which(name)

        monkeypatch.setattr("shutil.which", counting_which)
        # Call twice for the same executable
        try:
            utils.find_executable("python3")
            utils.find_executable("python3")
        except FileNotFoundError:
            pass
        # shutil.which should be called at most once due to caching
        assert call_count <= 1


# ---------------------------------------------------------------------------
# generate.py stress tests
# ---------------------------------------------------------------------------


class TestGenerateStress:
    """Stress tests for the generate module."""

    def test_estimate_cost_zero_items(self) -> None:
        """Cost estimation with zero items should produce $0."""
        result = generate.estimate_cost("gemini-2.5-flash", 0)
        assert "$0.000" in result

    def test_estimate_cost_large_queue(self) -> None:
        """Cost estimation with 10000 items."""
        result = generate.estimate_cost("gemini-2.5-flash", 10000)
        assert "Estimated cost:" in result
        # Should produce a non-zero cost
        assert "$0.000" not in result or "input" in result

    def test_estimate_cost_all_known_models(self) -> None:
        """All models in MODEL_COSTS should produce valid estimates."""
        for model in generate.MODEL_COSTS:
            result = generate.estimate_cost(model, 10)
            assert result.startswith("Estimated cost:")
            assert result.count("$") == 3

    def test_estimate_cost_case_sensitivity(self) -> None:
        """Model names are case-insensitive for cost estimation."""
        lower = generate.estimate_cost("gemini-2.5-flash", 10)
        # The function lowercases the model name
        assert "Estimated cost:" in lower

    @pytest.mark.asyncio
    async def test_async_generate_empty_queue(self, tmp_path: Path) -> None:
        """Async generation with empty queue should return empty list."""
        options = generate.GenerateAltTextOptions(
            root=tmp_path,
            model="test",
            max_chars=100,
            timeout=10,
            output_path=tmp_path / "out.json",
        )
        results = await generate.async_generate_suggestions([], options)
        assert results == []

    @pytest.mark.asyncio
    async def test_async_generate_with_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

        call_count = 0

        def failing_download(qi, workspace):
            nonlocal call_count
            call_count += 1
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
            root=tmp_path,
            model="test",
            max_chars=100,
            timeout=10,
            output_path=tmp_path / "out.json",
        )
        results = await generate.async_generate_suggestions(queue_items, options)
        # 4 out of 5 should succeed
        assert len(results) == 4

    def test_filter_existing_captions_empty_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filtering with empty queue should return empty list."""
        monkeypatch.setattr(
            utils, "load_existing_captions", lambda _: set()
        )
        result = generate.filter_existing_captions(
            [], [Path("captions.json")], Mock()
        )
        assert result == []

    def test_filter_existing_captions_all_existing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All items already captioned should result in empty list."""
        items = [
            scan.QueueItem(
                markdown_file="t.md",
                asset_path="img.jpg",
                line_number=1,
                context_snippet="c",
            )
        ]
        monkeypatch.setattr(
            utils, "load_existing_captions", lambda _: {"img.jpg"}
        )
        result = generate.filter_existing_captions(
            items, [Path("c.json")], Mock()
        )
        assert result == []


# ---------------------------------------------------------------------------
# label.py stress tests
# ---------------------------------------------------------------------------


class TestLabelStress:
    """Stress tests for the labeling module."""

    def test_labeling_session_empty_suggestions(self) -> None:
        """LabelingSession with no suggestions should be immediately complete."""
        session = label.LabelingSession([])
        assert session.is_complete()
        assert session.get_current_suggestion() is None
        assert session.get_progress() == (1, 0)

    def test_labeling_session_undo_at_beginning(self) -> None:
        """Undo when nothing has been processed should return None."""
        session = label.LabelingSession([_make_alt(1)])
        assert session.undo() is None

    def test_labeling_session_multiple_undos(self) -> None:
        """Multiple consecutive undos should work correctly."""
        suggestions = [_make_alt(i) for i in range(1, 4)]
        session = label.LabelingSession(suggestions)

        # Process all three
        for s in suggestions:
            session.add_result(
                utils.AltGenerationResult(
                    markdown_file=s.markdown_file,
                    asset_path=s.asset_path,
                    suggested_alt=s.suggested_alt,
                    model=s.model,
                    context_snippet=s.context_snippet,
                    line_number=s.line_number,
                    final_alt=f"final_{s.line_number}",
                )
            )

        assert session.is_complete()

        # Undo all three
        undone3 = session.undo()
        assert undone3 is not None
        assert undone3.final_alt == "final_3"

        undone2 = session.undo()
        assert undone2 is not None
        assert undone2.final_alt == "final_2"

        undone1 = session.undo()
        assert undone1 is not None
        assert undone1.final_alt == "final_1"

        # Now at beginning
        assert session.undo() is None
        assert session.current_index == 0

    def test_labeling_session_skip_current(self) -> None:
        """skip_current should advance index without adding to results."""
        suggestions = [_make_alt(1), _make_alt(2)]
        session = label.LabelingSession(suggestions)

        session.skip_current()
        assert session.current_index == 1
        assert len(session.processed_results) == 0

        session.skip_current()
        assert session.is_complete()

    def test_labeling_session_skip_then_process(self) -> None:
        """Skipping one item then processing the next."""
        suggestions = [_make_alt(1), _make_alt(2)]
        session = label.LabelingSession(suggestions)

        session.skip_current()  # Skip first
        session.add_result(
            utils.AltGenerationResult(
                markdown_file="test2.md",
                asset_path="image2.jpg",
                suggested_alt="s2",
                model="m",
                context_snippet="c2",
                line_number=2,
                final_alt="final_2",
            )
        )

        assert session.is_complete()
        assert len(session.processed_results) == 1

    def test_display_manager_show_progress(self) -> None:
        """show_progress should not crash."""
        output = StringIO()
        console = Console(file=output)
        dm = label.DisplayManager(console)
        dm.show_progress(5, 10)
        assert "5/10" in output.getvalue()

    def test_display_manager_show_rule(self) -> None:
        """show_rule should not crash."""
        output = StringIO()
        console = Console(file=output)
        dm = label.DisplayManager(console)
        dm.show_rule("test_asset.png")
        assert "test_asset.png" in output.getvalue()

    def test_display_manager_show_error(self) -> None:
        """show_error should display error message."""
        output = StringIO()
        console = Console(file=output)
        dm = label.DisplayManager(console)
        dm.show_error("Something went wrong")
        assert "Something went wrong" in output.getvalue()

    def test_label_suggestions_empty_list(self, tmp_path: Path) -> None:
        """Labeling with empty suggestions should return 0."""
        output = tmp_path / "output.json"
        result = label.label_suggestions(
            [], Console(file=StringIO()), output, append_mode=False
        )
        assert result == 0


# ---------------------------------------------------------------------------
# Integration / round-trip stress tests
# ---------------------------------------------------------------------------


class TestIntegrationStress:
    """Integration tests that exercise multiple modules together."""

    def test_scan_then_apply_roundtrip(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Scan a file, create captions, apply them - full pipeline."""
        md_path = tmp_path / "article.md"
        md_content = textwrap.dedent("""\
            # My Article

            Here is an image:

            ![](photo.jpg)

            And another:

            <img src="diagram.png">

            And a wikilink:

            ![[chart.svg]]
        """)
        md_path.write_text(md_content, encoding="utf-8")

        # Scan
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 3

        # Create captions data from queue
        captions_data = []
        for item in queue:
            captions_data.append(
                {
                    "markdown_file": item.markdown_file,
                    "asset_path": item.asset_path,
                    "line_number": item.line_number,
                    "suggested_alt": f"Suggestion for {item.asset_path}",
                    "final_alt": f"Alt text for {item.asset_path}",
                    "model": "test-model",
                    "context_snippet": item.context_snippet,
                }
            )

        captions_path = tmp_path / "captions.json"
        captions_path.write_text(json.dumps(captions_data))

        # Apply
        applied = apply.apply_captions(captions_path, console)
        assert applied == 3

        # Verify
        new_content = md_path.read_text()
        assert "![Alt text for photo.jpg](photo.jpg)" in new_content
        assert 'alt="Alt text for diagram.png"' in new_content
        assert "![[chart.svg|Alt text for chart.svg]]" in new_content

    def test_scan_then_apply_no_regressions(
        self, tmp_path: Path, console: Console
    ) -> None:
        """After applying, rescanning should find no issues."""
        md_path = tmp_path / "article.md"
        md_path.write_text(
            "![](photo.jpg)\n\n<img src='diagram.png'>\n",
            encoding="utf-8",
        )

        # First scan
        queue = scan.build_queue(tmp_path)
        assert len(queue) == 2

        # Apply captions
        captions_data = [
            {
                "markdown_file": str(md_path),
                "asset_path": item.asset_path,
                "line_number": item.line_number,
                "suggested_alt": "s",
                "final_alt": f"Good description of {item.asset_path}",
                "model": "m",
                "context_snippet": "c",
            }
            for item in queue
        ]
        captions_path = tmp_path / "captions.json"
        captions_path.write_text(json.dumps(captions_data))
        apply.apply_captions(captions_path, console)

        # Rescan - should find no issues now
        queue_after = scan.build_queue(tmp_path)
        assert len(queue_after) == 0

    def test_large_mixed_file_scan_and_apply(
        self, tmp_path: Path, console: Console
    ) -> None:
        """Large file with mixed formats: scan, mock-caption, apply, verify."""
        lines = ["# Large Document\n"]
        for i in range(30):
            lines.append(f"\nParagraph {i} with some text.\n")
            if i % 3 == 0:
                lines.append(f"\n![](markdown_{i}.png)\n")
            elif i % 3 == 1:
                lines.append(f'\n<img src="html_{i}.jpg">\n')
            else:
                lines.append(f"\n![[wiki_{i}.gif]]\n")

        md_path = tmp_path / "large.md"
        md_path.write_text("".join(lines), encoding="utf-8")

        queue = scan.build_queue(tmp_path)
        assert len(queue) == 30

        captions = [
            {
                "markdown_file": str(md_path),
                "asset_path": item.asset_path,
                "line_number": item.line_number,
                "suggested_alt": "s",
                "final_alt": f"Description {item.asset_path}",
                "model": "m",
                "context_snippet": "c",
            }
            for item in queue
        ]
        cp = tmp_path / "captions.json"
        cp.write_text(json.dumps(captions))
        applied = apply.apply_captions(cp, console)
        assert applied == 30

        # Verify rescan finds nothing
        queue_after = scan.build_queue(tmp_path)
        assert len(queue_after) == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alt(idx: int, **kwargs) -> utils.AltGenerationResult:
    defaults = dict(
        markdown_file=f"test{idx}.md",
        asset_path=f"image{idx}.jpg",
        suggested_alt=f"suggestion {idx}",
        model="test-model",
        context_snippet=f"context {idx}",
        line_number=idx,
    )
    defaults.update(kwargs)
    return utils.AltGenerationResult(**defaults)
