"""Regeneration loop: verify -> fix flagged claims -> re-verify (max 2 attempts).

Every revision is sent back through pipeline.run_trustguard, which re-extracts
claims and re-verifies them against the evidence from scratch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from regeneration.feedback_builder import build_regeneration_prompt, extract_feedback
from regeneration.regeneration import regenerate_answer

MAX_ATTEMPTS = 2
MIN_IMPROVEMENT = 1.0   # composite score is 0-100; stop if it gains less than this


def _to_dict(result: Any) -> dict:
    return result if isinstance(result, dict) else result.to_dict()


def composite_score(result: dict) -> Optional[float]:
    return (result.get("trust_aggregation") or {}).get("composite_score")


def is_accepted(result: dict) -> bool:
    """Uses the team's decision_engine verdict (ACCEPT vs REGENERATE)."""
    dec = result.get("decision") or {}
    return dec.get("decision") == "ACCEPT" and not dec.get("regeneration_required", False)


@dataclass
class Attempt:
    attempt: int                  # 0 = original answer, 1..2 = regenerations
    answer: str
    composite_score: Optional[float]
    decision: str
    flagged: dict                 # counts: contradicted / unverifiable / supported
    prompt: Optional[str] = None


@dataclass
class RegenerationOutcome:
    question: str
    original_answer: str
    final_answer: str
    status: str                   # accepted | accepted_after_regeneration | human_review
    stop_reason: str
    needs_human_review: bool
    safety_gate_triggered: bool
    attempts: list[Attempt] = field(default_factory=list)
    final_result: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_with_regeneration(
    question: str,
    answer: Optional[str] = None,
    *,
    trust_runner: Optional[Callable[..., Any]] = None,
    regenerator: Optional[Callable[[str], str]] = None,
    max_attempts: int = MAX_ATTEMPTS,
    **trust_kwargs,
) -> RegenerationOutcome:
    """Evaluate an answer; if it is not trusted, regenerate up to two times.

    ``trust_kwargs`` go straight to ``run_trustguard`` (top_k=..., retriever=...).
    """
    max_attempts = min(max_attempts, MAX_ATTEMPTS)  # hard cap at two
    if trust_runner is None:
        from pipeline import build_retriever, run_trustguard

        trust_runner = run_trustguard
        trust_kwargs.setdefault("retriever", build_retriever())  # build once

    result = _to_dict(trust_runner(question, answer, **trust_kwargs))
    current = original = result["answer"]
    fb = extract_feedback(result)
    attempts = [Attempt(0, current, composite_score(result),
                        (result.get("decision") or {}).get("decision", ""), fb.summary())]

    def finish(status: str, reason: str, human: bool) -> RegenerationOutcome:
        return RegenerationOutcome(
            question=question, original_answer=original, final_answer=current,
            status=status, stop_reason=reason, needs_human_review=human,
            safety_gate_triggered=bool((result.get("decision") or {}).get("safety_gate_triggered")),
            attempts=attempts, final_result=result,
        )

    if is_accepted(result):
        return finish("accepted", "original answer already trusted", False)

    for n in range(1, max_attempts + 1):
        if not fb.needs_fix:
            return finish("human_review",
                          "score below threshold but no flagged claims to fix", True)

        prompt = build_regeneration_prompt(question, current, fb)
        try:
            revised = regenerate_answer(prompt, generator=regenerator)
        except Exception as exc:  # Ollama down, empty output, ...
            return finish("human_review", f"regeneration failed: {exc}", True)

        if revised.strip() == current.strip():
            return finish("human_review", "revision identical to previous answer", True)

        prev_score = composite_score(result)
        result = _to_dict(trust_runner(question, revised, **trust_kwargs))  # re-verify
        current = result["answer"]
        fb = extract_feedback(result)
        attempts.append(Attempt(n, current, composite_score(result),
                                (result.get("decision") or {}).get("decision", ""),
                                fb.summary(), prompt))

        if is_accepted(result):
            return finish("accepted_after_regeneration",
                          f"trusted after {n} regeneration(s)", False)

        new_score = composite_score(result)
        if (n < max_attempts and prev_score is not None and new_score is not None
                and new_score - prev_score < MIN_IMPROVEMENT):
            return finish("human_review", "trust score stopped improving", True)

    return finish("human_review", f"still not trusted after {max_attempts} attempts", True)