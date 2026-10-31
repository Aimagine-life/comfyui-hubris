"""HTTP client for the Hubris API: key lookup, requests, readable errors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

BASE_URL = os.environ.get("HUBRIS_BASE_URL", "https://api.hubris.pw/v1").rstrip("/")
PACKAGE_DIR = Path(__file__).resolve().parent.parent
KEY_FILE = PACKAGE_DIR / "hubris_api_key.txt"
USER_AGENT = "comfyui-hubris/1.0.0"

KEY_HELP = (
    "Hubris API key not found. Create a key at https://hubris.pw/keys and paste it into "
    f"{KEY_FILE} (one line, works right away) or set the HUBRIS_API_KEY environment "
    "variable and restart ComfyUI."
)


class HubrisError(Exception):
    """Error shown to the user in the ComfyUI error dialog."""


def get_api_key() -> str:
    key = os.environ.get("HUBRIS_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise HubrisError(KEY_HELP)
    if not key.startswith("sk-gw-"):
        raise HubrisError("Hubris API key must start with sk-gw-. " + KEY_HELP)
    return key


def _headers(extra: dict[str, str] | None = None, auth: bool = True) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT, "X-Title": "ComfyUI"}
    if auth:
        headers["Authorization"] = f"Bearer {get_api_key()}"
    if extra:
        headers.update(extra)
    return headers


def error_message(response: requests.Response) -> str:
    """OpenAI-style error body -> one line for the user."""
    try:
        err = response.json().get("error") or {}
        message = err.get("message") or response.text[:300]
        code = err.get("code")
    except ValueError:
        message, code = response.text[:300] or response.reason, None
    suffix = f" ({code})" if code else ""
    return f"Hubris API {response.status_code}{suffix}: {message}"


def request(
    method: str,
    path: str,
    *,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    auth: bool = True,
    timeout: float = 600,
    raw: bool = False,
) -> Any:
    url = f"{BASE_URL}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=json,
            params=params,
            headers=_headers(headers, auth),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise HubrisError(f"Cannot reach Hubris API: {exc.__class__.__name__}") from exc
    if response.status_code >= 400:
        raise HubrisError(error_message(response))
    if raw:
        return response
    try:
        return response.json()
    except ValueError as exc:
        raise HubrisError("Hubris API returned a non-JSON response") from exc
