"""Client for the OpenRouter API.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint plus a public
model catalogue with live per-token pricing.  This module is the single place
that talks to OpenRouter: it builds multimodal requests (text + image/video),
fetches pricing for cost estimation, and lists model ids for shell completion.
"""

import base64
import functools
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from alt_text_llm import utils

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

# Optional attribution headers used by OpenRouter's public leaderboards. They
# have no functional effect on responses.
_REFERER = "https://github.com/alexander-turner/alt-text-llm"
_TITLE = "alt-text-llm"

_MODELS_TIMEOUT = 20

# OpenRouter accepts video through a dedicated ``video_url`` content part for
# models whose input modalities include video (currently Google Gemini).  Map
# the extensions we may hand it to their MIME types; unknown video extensions
# fall back to ``video/mp4``.
_VIDEO_MIME_BY_SUFFIX: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}


class OpenRouterError(utils.AltGenerationError):
    """Raised when an OpenRouter API request fails.

    Subclasses :class:`utils.AltGenerationError` so that per-asset request
    failures are skipped by the generation loop just like other caption errors.
    """


def get_api_key() -> str:
    """Return the OpenRouter API key, or raise a helpful error if it is unset."""
    key = os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise OpenRouterError(
            f"Missing {API_KEY_ENV_VAR}. Create a key at "
            f"https://openrouter.ai/keys and export it, e.g. "
            f"`export {API_KEY_ENV_VAR}=sk-or-...`."
        )
    return key


def _request_headers() -> dict[str, str]:
    """Build authenticated request headers for chat completions."""
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        "HTTP-Referer": _REFERER,
        "X-Title": _TITLE,
    }


def _content_part(attachment: Path) -> dict[str, Any]:
    """Build the multimodal content part (image or video) for *attachment*."""
    raw = attachment.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    suffix = attachment.suffix.lower()

    if utils.is_video_asset(str(attachment)) or suffix in _VIDEO_MIME_BY_SUFFIX:
        mime = _VIDEO_MIME_BY_SUFFIX.get(suffix, "video/mp4")
        return {
            "type": "video_url",
            "video_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    # Default to PNG when the extension isn't a recognised image type, so we
    # always send a valid ``image/*`` MIME prefix.
    guessed = mimetypes.guess_type(attachment.name)[0]
    image_mime = guessed if guessed and guessed.startswith("image/") else "image/png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image_mime};base64,{encoded}"},
    }


def generate_caption(
    attachment: Path,
    prompt: str,
    model: str,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    """Generate a caption for *attachment* via OpenRouter.

    Returns the caption text and the response's ``usage`` object (which includes
    OpenRouter's authoritative ``cost`` field in USD).

    Raises:
        OpenRouterError: If the request fails or the response is malformed.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    _content_part(attachment),
                ],
            }
        ],
    }

    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=_request_headers(),
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as err:
        raise OpenRouterError(
            f"OpenRouter request failed for {attachment}: {err}"
        ) from err

    if response.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter returned HTTP {response.status_code} for "
            f"{attachment}: {response.text.strip()}"
        )

    data = response.json()
    # OpenRouter signals model/provider problems via an ``error`` object even on
    # a 200 response.
    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = (
            error.get("message") if isinstance(error, dict) else error
        )
        raise OpenRouterError(
            f"OpenRouter error for {attachment}: {message}"
        )

    try:
        caption = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as err:
        raise OpenRouterError(
            f"Unexpected OpenRouter response for {attachment}: {data}"
        ) from err

    caption = (caption or "").strip()
    if not caption:
        raise OpenRouterError("OpenRouter returned an empty caption")

    usage = data.get("usage") if isinstance(data, dict) else None
    return caption, usage if isinstance(usage, dict) else {}


@functools.lru_cache(maxsize=1)
def fetch_models() -> tuple[dict[str, Any], ...]:
    """Fetch and cache OpenRouter's public model catalogue.

    Cached for the lifetime of the process. Raises :class:`OpenRouterError` on
    network failure so callers can decide whether to degrade gracefully.
    """
    try:
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models", timeout=_MODELS_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as err:
        raise OpenRouterError(
            f"Failed to fetch OpenRouter models: {err}"
        ) from err

    data = response.json().get("data", [])
    return tuple(entry for entry in data if isinstance(entry, dict))


def get_pricing(model: str) -> dict[str, float] | None:
    """Return live per-token USD pricing for *model*.

    Returns a dict with ``input`` and ``output`` prices (USD per token), or
    ``None`` if the model is unknown or pricing cannot be fetched/parsed.
    """
    try:
        models = fetch_models()
    except OpenRouterError:
        return None

    for entry in models:
        if entry.get("id") == model:
            pricing = entry.get("pricing") or {}
            try:
                return {
                    "input": float(pricing["prompt"]),
                    "output": float(pricing["completion"]),
                }
            except (KeyError, TypeError, ValueError):
                return None
    return None


def list_model_ids() -> list[str]:
    """Return all OpenRouter model ids, or ``[]`` if the catalogue is unreachable.

    Used by shell completion, so it never raises.
    """
    try:
        return [entry["id"] for entry in fetch_models() if "id" in entry]
    except OpenRouterError:
        return []
