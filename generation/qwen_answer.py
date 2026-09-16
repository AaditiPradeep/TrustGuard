"""Generate a response to a user question through local Ollama."""

from __future__ import annotations

from local_llm.ollama import DEFAULT_MODEL, chat


MODEL_NAME = DEFAULT_MODEL

SYSTEM_PROMPT = """You are a careful medical-information assistant. Answer the
user's question clearly and concisely. State uncertainty when the available
information may depend on an individual's condition. Do not invent sources or
citation markers. Return only the answer, without analysis or Markdown headings."""


def generate_answer(
    question: str,
    *,
    model_name: str = MODEL_NAME,
    max_tokens: int = 250,
) -> str:
    """Generate one answer for ``question`` with a local Ollama model."""
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty.")
    return chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model_name,
        max_tokens=max_tokens,
    )
