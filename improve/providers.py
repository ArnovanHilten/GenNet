"""OpenAI-compatible chat client. Secrets come from the environment only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


def settings_from_env() -> dict[str, str]:
    return {
        "api_key": os.environ.get("IMPROVE_API_KEY", ""),
        "base_url": os.environ.get("IMPROVE_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        "model": os.environ.get("IMPROVE_MODEL", "gpt-4o-mini"),
    }


def configured() -> bool:
    return bool(settings_from_env()["api_key"])


def chat(system: str, user: str, timeout: int = 120) -> str:
    cfg = settings_from_env()
    if not cfg["api_key"]:
        raise LLMError("IMPROVE_API_KEY is not set")
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        cfg["base_url"] + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"],
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise LLMError(f"LLM HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected LLM response: {body!r}") from exc
