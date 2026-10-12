"""Plain-language, structured reports for TrustGuard results."""

from __future__ import annotations

from typing import Any, Mapping


METRIC_NAMES = {
    "evidence_support": "Evidence support",
    "source_reliability": "Source reliability",
    "answer_stability": "Answer consistency",
    "citation_quality": "Citation quality",
}


def _percent(value: Any) -> str:
    return "not available" if value is None else f"{float(value):.1f}/100"


def build_trust_report(
    question: str, answer: str, aggregation: Mapping[str, Any], decision: Mapping[str, Any],
) -> dict:
    """Turn technical scoring output into a report for a non-specialist."""
    metrics = []
    for item in aggregation.get("metric_contributions", []):
        score = item.get("score")
        metrics.append({
            "name": METRIC_NAMES.get(item.get("metric"), item.get("metric", "Metric")),
            "score": round(score * 100, 2) if score is not None else None,
            "weight": round(item.get("applied_weight", 0) * 100, 1) if item.get("applied_weight") is not None else None,
            "coverage": round(item["coverage"] * 100, 1) if item.get("coverage") is not None else None,
            "explanation": (
                "This score was not available, so the available checks were weighted fairly."
                if score is None else _metric_explanation(item.get("metric"), score)
            ),
        })
    claims = []
    for item in aggregation.get("claim_contributions", []):
        claims.append({
            "claim": item.get("claim_text", ""),
            "status": item.get("status", "UNVERIFIED").title(),
            "metric_scores": {
                "evidence_support": item.get("evidence_support_score"),
                "source_reliability": (
                    round(item["source_reliability_score"] * 100, 2)
                    if item.get("source_reliability_score") is not None else None
                ),
                "citation_quality": item.get("citation_quality_score"),
                "answer_consistency": item.get("answer_stability_score"),
            },
            "trust_score": item.get("trust_score"),
            "category": item.get("category", "UNAVAILABLE").title(),
            "explanation": item.get("explanation", ""),
        })
    score = aggregation.get("composite_score")
    summary = (
        f"TrustGuard's overall confidence is {_percent(score)} ({aggregation.get('category', 'UNAVAILABLE').lower()} trust). "
        f"Recommendation: {decision.get('decision', 'REVIEW').lower()}."
    )
    return {
        "plain_language_summary": summary,
        "question": question,
        "answer_reviewed": answer,
        "overall_trust_score": score,
        "trust_category": aggregation.get("category"),
        "recommendation": decision.get("decision"),
        "recommendation_reason": decision.get("reason"),
        "why": decision.get("flags", []),
        "metric_scores": metrics,
        "claims": claims,
        "missing_information": aggregation.get("missing_metrics", []),
        "source_reliability_coverage": aggregation.get("source_reliability_coverage", 0) * 100,
    }


def _metric_explanation(metric: str, score: float) -> str:
    level = "strong" if score >= .80 else "some" if score >= .60 else "limited"
    descriptions = {
        "evidence_support": "how well the retrieved material backs up the answer's claims",
        "source_reliability": "whether authoritative retrieved sources support or contradict the answer",
        "answer_stability": "whether repeated answers to the same question stay consistent",
        "citation_quality": "how directly the retrieved source material relates to the claims",
    }
    return f"{level.capitalize()} confidence in {descriptions.get(metric, 'this check')}."
