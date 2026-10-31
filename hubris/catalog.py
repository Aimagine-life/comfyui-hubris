"""Model catalog for node dropdowns: fetched once at startup, cached on disk."""

from __future__ import annotations

import json
import logging
from typing import Any

from . import client

log = logging.getLogger("comfyui-hubris")
CACHE_FILE = client.PACKAGE_DIR / "catalog_cache.json"

# Preferred defaults, first one present in the catalog wins.
DEFAULTS = {
    "text": ["google/gemini-3.8-flash", "openai/gpt-6-luna", "deepseek/deepseek-chat"],
    "image": ["google/gemini-3.1-flash-image", "bytedance-seed/seedream-4.5", "black-forest-labs/flux.2-pro"],
    "video": ["kwaivgi/kling-v3.0-std", "google/veo-3.1-lite", "alibaba/wan-2.6"],
    "speech": ["google/gemini-3.8-flash-tts", "elevenlabs/eleven-flash-v2.5", "microsoft/mai-voice-2-flash"],
}

_models: list[dict[str, Any]] | None = None


def _load() -> list[dict[str, Any]]:
    global _models
    if _models is not None:
        return _models
    try:
        body = client.request("GET", "/models", params={"output_modalities": "all"}, auth=False, timeout=15)
        _models = body["data"]
        try:
            CACHE_FILE.write_text(json.dumps(_models), encoding="utf-8")
        except OSError:
            pass
    except Exception as exc:  # startup must not fail because the network is down
        log.warning("[Hubris] catalog fetch failed (%s), using cache", exc)
        try:
            _models = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _models = []
    return _models


def matches(model: dict[str, Any], kind: str) -> bool:
    out = model.get("output_modalities") or ["text"]
    if kind == "text":
        return "text" in out and "speech" not in out and "embeddings" not in out
    if kind == "image" and is_vector(model):
        return False  # SVG output cannot become a ComfyUI IMAGE
    return kind in out


def is_vector(model: dict[str, Any]) -> bool:
    return "svg" in (model.get("description") or "").lower()


def model_ids(kind: str) -> list[str]:
    ids = sorted(m["id"] for m in _load() if matches(m, kind))
    preferred = next((d for d in DEFAULTS[kind] if d in ids), None)
    if preferred:
        ids.remove(preferred)
        ids.insert(0, preferred)
    return ids or DEFAULTS[kind][:1]


def get(model_id: str) -> dict[str, Any] | None:
    return next((m for m in _load() if m["id"] == model_id), None)


def is_chat_image_model(model_id: str) -> bool:
    """Image models that also output text are served by /chat/completions."""
    model = get(model_id)
    return bool(model) and "text" in (model.get("output_modalities") or [])


def supports(model_id: str, parameter: str) -> bool:
    model = get(model_id)
    return bool(model) and parameter in (model.get("supported_parameters") or [])


def voices(model_id: str) -> list[str]:
    model = get(model_id)
    return list((model or {}).get("supported_voices") or [])


def video_model(model_id: str) -> dict[str, Any] | None:
    """Capabilities from /videos/models (durations, resolutions, frame images)."""
    body = client.request("GET", "/videos/models", timeout=30)
    return next((m for m in body.get("data", []) if m["id"] == model_id), None)
