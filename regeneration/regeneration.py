"""Call the generator LLM (local Ollama) to produce a revised answer."""

from __future__ import annotations

import re
from typing import Callable, Optional

SYSTEM_PROMPT = (
    "You are a careful medical-information assistant who corrects a draft answer "
    "using only the evidence supplied by the user. Return only the revised answer."
)


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"^\s*(revised answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def regenerate_answer(
    prompt: str,
    generator: Optional[Callable[[str], str]] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 350,
) -> str:
    """Return the revised answer; raise if the model returns nothing usable."""
    if generator is None:
        from local_llm.ollama import DEFAULT_MODEL, chat

        generator = lambda p: chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p}],
            model=model_name or DEFAULT_MODEL,
            max_tokens=max_tokens,
        )
    revised = _clean(generator(prompt))
    if not revised:
        raise ValueError("Generator returned an empty revision.")
    return revised