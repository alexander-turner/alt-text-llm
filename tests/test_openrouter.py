"""Tests for the OpenRouter API client."""

import base64
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from alt_text_llm import openrouter


@pytest.fixture(autouse=True)
def _clear_models_cache():
    """Reset the process-wide model catalogue cache between tests."""
    openrouter.fetch_models.cache_clear()
    yield
    openrouter.fetch_models.cache_clear()


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


def test_get_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    assert openrouter.get_api_key() == "sk-or-test"


def test_get_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(openrouter.API_KEY_ENV_VAR, raising=False)
    with pytest.raises(openrouter.OpenRouterError, match=openrouter.API_KEY_ENV_VAR):
        openrouter.get_api_key()


def test_openrouter_error_is_alt_generation_error() -> None:
    """Per-request failures must be skippable by the generation loop."""
    from alt_text_llm import utils

    assert issubclass(openrouter.OpenRouterError, utils.AltGenerationError)


# ---------------------------------------------------------------------------
# Multimodal content parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected_mime",
    [
        ("photo.png", "image/png"),
        ("photo.jpg", "image/jpeg"),
        ("photo.jpeg", "image/jpeg"),
        ("photo.webp", "image/webp"),
        ("photo.gif", "image/gif"),
    ],
)
def test_content_part_image(
    temp_dir: Path, filename: str, expected_mime: str
) -> None:
    asset = temp_dir / filename
    asset.write_bytes(b"binarydata")

    part = openrouter._content_part(asset)

    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    expected_b64 = base64.b64encode(b"binarydata").decode("ascii")
    assert url == f"data:{expected_mime};base64,{expected_b64}"


@pytest.mark.parametrize(
    "filename, expected_mime",
    [
        ("clip.mp4", "video/mp4"),
        ("clip.webm", "video/webm"),
        ("clip.mov", "video/quicktime"),
        ("clip.mpeg", "video/mpeg"),
        # Unknown/uncommon video extensions fall back to mp4.
        ("clip.mkv", "video/mp4"),
        ("clip.avi", "video/mp4"),
    ],
)
def test_content_part_video(
    temp_dir: Path, filename: str, expected_mime: str
) -> None:
    asset = temp_dir / filename
    asset.write_bytes(b"videobytes")

    part = openrouter._content_part(asset)

    assert part["type"] == "video_url"
    url = part["video_url"]["url"]
    assert url.startswith(f"data:{expected_mime};base64,")


def test_content_part_unknown_image_extension_defaults_png(
    temp_dir: Path,
) -> None:
    asset = temp_dir / "mystery.bin"
    asset.write_bytes(b"data")

    part = openrouter._content_part(asset)

    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# generate_caption
# ---------------------------------------------------------------------------


def _mock_post_response(
    status_code: int = 200, json_data: dict[str, Any] | None = None, text: str = ""
) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.text = text
    response.json.return_value = json_data or {}
    return response


def test_generate_caption_success(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"pngdata")

    payload = {
        "choices": [{"message": {"content": "  A red square.  "}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "cost": 0.000123,
        },
    }
    captured: dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _mock_post_response(json_data=payload)

    monkeypatch.setattr(requests, "post", fake_post)

    caption, usage = openrouter.generate_caption(
        asset, "Describe this", "google/gemini-2.5-flash", 60
    )

    assert caption == "A red square."  # whitespace stripped
    assert usage["cost"] == 0.000123

    # Verify the request was constructed correctly.
    assert captured["url"] == f"{openrouter.OPENROUTER_BASE_URL}/chat/completions"
    assert captured["timeout"] == 60
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["Content-Type"] == "application/json"
    body = captured["json"]
    assert body["model"] == "google/gemini-2.5-flash"
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_generate_caption_sends_video_part(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "clip.mp4"
    asset.write_bytes(b"mp4data")

    captured: dict[str, Any] = {}

    def fake_post(url: str, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _mock_post_response(
            json_data={"choices": [{"message": {"content": "A clip."}}]}
        )

    monkeypatch.setattr(requests, "post", fake_post)

    caption, usage = openrouter.generate_caption(
        asset, "Describe", "google/gemini-2.5-flash", 60
    )

    assert caption == "A clip."
    assert usage == {}  # no usage block in response
    content_part = captured["json"]["messages"][0]["content"][1]
    assert content_part["type"] == "video_url"
    assert content_part["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_generate_caption_http_error(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _mock_post_response(
            status_code=401, text="Unauthorized"
        ),
    )

    with pytest.raises(openrouter.OpenRouterError, match="HTTP 401"):
        openrouter.generate_caption(asset, "p", "m", 60)


def test_generate_caption_error_object(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200 response carrying an error object must raise."""
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _mock_post_response(
            json_data={"error": {"message": "Unknown model: bogus"}}
        ),
    )

    with pytest.raises(openrouter.OpenRouterError, match="Unknown model: bogus"):
        openrouter.generate_caption(asset, "p", "bogus", 60)


def test_generate_caption_malformed_response(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _mock_post_response(json_data={"choices": []}),
    )

    with pytest.raises(openrouter.OpenRouterError, match="Unexpected"):
        openrouter.generate_caption(asset, "p", "m", 60)


def test_generate_caption_empty_caption(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _mock_post_response(
            json_data={"choices": [{"message": {"content": "   "}}]}
        ),
    )

    with pytest.raises(openrouter.OpenRouterError, match="empty caption"):
        openrouter.generate_caption(asset, "p", "m", 60)


def test_generate_caption_network_error(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(openrouter.API_KEY_ENV_VAR, "sk-or-test")
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    def raise_timeout(*a, **k):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", raise_timeout)

    with pytest.raises(openrouter.OpenRouterError, match="request failed"):
        openrouter.generate_caption(asset, "p", "m", 60)


def test_generate_caption_missing_key_raises(
    temp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(openrouter.API_KEY_ENV_VAR, raising=False)
    asset = temp_dir / "img.png"
    asset.write_bytes(b"data")

    with pytest.raises(openrouter.OpenRouterError, match="Missing"):
        openrouter.generate_caption(asset, "p", "m", 60)


# ---------------------------------------------------------------------------
# Models catalogue, pricing, completion
# ---------------------------------------------------------------------------


_SAMPLE_MODELS = {
    "data": [
        {
            "id": "google/gemini-2.5-flash",
            "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {
            # Malformed pricing entry — must not crash get_pricing.
            "id": "broken/model",
            "pricing": {"prompt": "not-a-number", "completion": "0.0"},
        },
        "this-is-not-a-dict",  # must be filtered out
    ]
}


def _mock_get(json_data: dict[str, Any]) -> Any:
    def fake_get(url: str, timeout=None):
        response = Mock()
        response.json.return_value = json_data
        response.raise_for_status = Mock()
        return response

    return fake_get


def test_fetch_models_filters_non_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(requests, "get", _mock_get(_SAMPLE_MODELS))
    models = openrouter.fetch_models()
    assert all(isinstance(entry, dict) for entry in models)
    assert len(models) == 3


def test_fetch_models_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def counting_get(url: str, timeout=None):
        calls["n"] += 1
        response = Mock()
        response.json.return_value = _SAMPLE_MODELS
        response.raise_for_status = Mock()
        return response

    monkeypatch.setattr(requests, "get", counting_get)
    openrouter.fetch_models()
    openrouter.fetch_models()
    assert calls["n"] == 1  # second call served from cache


def test_fetch_models_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_err(url: str, timeout=None):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", raise_err)
    with pytest.raises(openrouter.OpenRouterError, match="Failed to fetch"):
        openrouter.fetch_models()


def test_get_pricing_known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", _mock_get(_SAMPLE_MODELS))
    pricing = openrouter.get_pricing("google/gemini-2.5-flash")
    assert pricing == {"input": 0.0000003, "output": 0.0000025}


def test_get_pricing_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", _mock_get(_SAMPLE_MODELS))
    assert openrouter.get_pricing("does/not-exist") is None


def test_get_pricing_malformed_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", _mock_get(_SAMPLE_MODELS))
    assert openrouter.get_pricing("broken/model") is None


def test_get_pricing_network_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(url: str, timeout=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", raise_err)
    assert openrouter.get_pricing("google/gemini-2.5-flash") is None


def test_list_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", _mock_get(_SAMPLE_MODELS))
    ids = openrouter.list_model_ids()
    assert "google/gemini-2.5-flash" in ids
    assert "anthropic/claude-sonnet-4.5" in ids


def test_list_model_ids_network_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_err(url: str, timeout=None):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", raise_err)
    assert openrouter.list_model_ids() == []
