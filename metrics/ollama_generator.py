"""Local Ollama response sampler used by the Answer Stability metric."""

from __future__ import annotations

from local_llm.ollama import DEFAULT_MODEL, chat


def generator_llm(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Sample one answer locally; temperature must stay above zero for stability."""
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0 for Answer Stability.")
    return chat(
        [{"role": "user", "content": question}],
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )
