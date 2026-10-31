"""HTTP client for the Hubris API: key lookup, requests, readable errors."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import requests

BASE_URL = os.environ.get("HUBRIS_BASE_URL", "https://api.hubris.pw/v1").rstrip("/")
PACKAGE_DIR = Path(__file__).resolve().parent.parent
KEY_NAME = "hubris_api_key.txt"
KEY_FILE = PACKAGE_DIR / KEY_NAME  # older installs; lost when the node is updated
USER_AGENT = "comfyui-hubris/1.0.1"


def _user_key_file() -> Path | None:
    """ComfyUI/user/hubris_api_key.txt: survives node updates."""
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / KEY_NAME
    except (ImportError, AttributeError):
        return None


def _key_help() -> str:
    place = _user_key_file() or KEY_FILE
    return (
        "Hubris API key not found. Create a key at https://hubris.pw/keys and paste it into "
        f"{place} (one line, works right away) or set the HUBRIS_API_KEY environment "
        "variable and restart ComfyUI."
    )


class HubrisError(Exception):
    """Error shown to the user in the ComfyUI error dialog."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP status; None for network errors and local checks


def get_api_key() -> str:
    key = os.environ.get("HUBRIS_API_KEY", "").strip()
    for path in (_user_key_file(), KEY_FILE):
        if key:
            break
        if path and path.is_file():
            key = path.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise HubrisError(_key_help())
    if not key.startswith("sk-gw-"):
        raise HubrisError("Hubris API key must start with sk-gw-. " + _key_help())
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
    timeout: float | tuple[float, float] = 600,
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
    except requests.Timeout as exc:
        raise HubrisError(
            "Hubris API did not answer in time. The request may still have run and been billed; "
            "check https://hubris.pw/logs before retrying"
        ) from exc
    except requests.RequestException as exc:
        raise HubrisError(f"Cannot reach Hubris API: {exc.__class__.__name__}") from exc
    if response.status_code >= 400:
        raise HubrisError(error_message(response), response.status_code)
    if raw:
        return response
    try:
        return response.json()
    except ValueError as exc:
        raise HubrisError("Hubris API returned a non-JSON response") from exc


def is_transient(error: HubrisError) -> bool:
    """Network failure, rate limit or server error: worth retrying a read-only request."""
    return error.status is None or error.status == 429 or error.status >= 500


def retry_until(deadline: float, call: Callable[[], Any], on_wait: Callable[[], None] = lambda: None) -> Any:
    """Repeat a read-only call on transient errors until the monotonic deadline."""
    while True:
        try:
            return call()
        except HubrisError as error:
            if not is_transient(error) or time.monotonic() > deadline:
                raise
        for _ in range(20):
            on_wait()
            time.sleep(0.25)
