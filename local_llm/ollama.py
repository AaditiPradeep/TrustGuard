"""Minimal Ollama chat client with no extra Python dependency."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "qwen3:1.7b"


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int,
    temperature: float = 0.2,
    top_p: float = 0.8,
) -> str:
    """Send one non-streaming chat request to a local Ollama server."""
    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
    }).encode("utf-8")
    request = Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama could not run {model}: {detail}. "
            f"Run `ollama pull {model}` and try again."
        ) from error
    except URLError as error:
        raise RuntimeError(
            "Cannot reach Ollama at " + base_url + ". Start the Ollama app, "
            f"then run `ollama pull {model}`."
        ) from error

    content = data.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Ollama returned an empty response for {model}.")
    return content.strip()
