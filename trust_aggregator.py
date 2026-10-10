"""Combine existing TrustGuard metric outputs without recalculating them.

This module deliberately treats the four metric modules as the source of
truth.  It only normalises their already-produced response dictionaries and
explains how much each available metric contributed to the composite score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


DEFAULT_WEIGHTS = {
    "evidence_support": 0.40,
    "source_reliability": 0.25,
    "answer_stability": 0.20,
    "citation_quality": 0.15,
}


def _number(value: Any) -> Optional[float]:
    """Return a finite score in [0, 1], or None when it is unavailable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value))


def _metric_values(
    evidence_support: Mapping[str, Any],
    source_reliability: Mapping[str, Any],
    answer_stability: Optional[Mapping[str, Any]],
    citation_quality: Optional[Mapping[str, Any]],
) -> dict[str, Optional[float]]:
    citation = None
    if citation_quality and citation_quality.get("available") is not False:
        citation = _number(citation_quality.get("citation_quality_score"))
    return {
        "evidence_support": _number(evidence_support.get("evidence_support_score", 0) / 100),
        "source_reliability": _number(source_reliability.get("rave_score")),
        "answer_stability": _number(
            answer_stability.get("average_stability") if answer_stability else None
        ),
        "citation_quality": citation,
    }


def trust_category(score: Optional[float]) -> str:
    if score is None:
        return "UNAVAILABLE"
    if score >= 0.80:
        return "HIGH"
    if score >= 0.60:
        return "MODERATE"
    return "LOW"


@dataclass
class MetricContribution:
    metric: str
    score: Optional[float]
    configured_weight: float
    applied_weight: float
    contribution: Optional[float]
    available: bool
    coverage: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("score", "applied_weight", "contribution", "coverage"):
            if data[key] is not None:
                data[key] = round(data[key], 4)
        return data


@dataclass
class ClaimTrustContribution:
    claim_id: str
    claim_text: str
    evidence_support_score: Optional[float]
    source_reliability_score: Optional[float]
    citation_quality_score: Optional[float]
    answer_stability_score: Optional[float]
    trust_score: Optional[float]
    category: str
    status: str
    explanation: str

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "evidence_support_score",
            "citation_quality_score", "trust_score",
            "answer_stability_score",
        ):
            if data[key] is not None:
                data[key] = round(data[key], 2)
        return data


@dataclass
class TrustAggregation:
    composite_score: Optional[float]
    category: str
    metric_contributions: list[MetricContribution] = field(default_factory=list)
    claim_contributions: list[ClaimTrustContribution] = field(default_factory=list)
    missing_metrics: list[str] = field(default_factory=list)
    source_reliability_coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 2) if self.composite_score is not None else None,
            "category": self.category,
            "metric_contributions": [item.to_dict() for item in self.metric_contributions],
            "claim_contributions": [item.to_dict() for item in self.claim_contributions],
            "missing_metrics": self.missing_metrics,
            "source_reliability_coverage": round(self.source_reliability_coverage, 4),
        }


def _claim_explanation(status: str, has_rave: bool, has_citation: bool) -> str:
    base = {
        "SUPPORTED": "The retrieved evidence supports this claim.",
        "CONTRADICTED": "The retrieved evidence conflicts with this claim.",
        "MIXED": "The retrieved evidence both supports and conflicts with this claim.",
        "UNVERIFIED": "The retrieved evidence did not verify this claim.",
    }.get(status, "There was not enough information to verify this claim.")
    if not has_rave:
        base += " The source-agreement score was not available for this claim."
    if not has_citation:
        base += " No citation-quality score was available for this claim."
    return base


def aggregate_trust(
    evidence_support: Mapping[str, Any],
    source_reliability: Mapping[str, Any],
    answer_stability: Optional[Mapping[str, Any]],
    citation_quality: Optional[Mapping[str, Any]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> TrustAggregation:
    """Produce an availability-aware weighted composite and claim summaries.

    Missing metrics are explicitly reported and their configured weight is
    redistributed over the scores that are available; missing data therefore
    never silently becomes a zero.  No underlying metric value is altered.
    """
    configured = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if set(configured) != set(DEFAULT_WEIGHTS) or any(value < 0 for value in configured.values()):
        raise ValueError("weights must contain all four non-negative metric weights.")
    if sum(configured.values()) <= 0:
        raise ValueError("at least one metric weight must be positive.")

    values = _metric_values(evidence_support, source_reliability, answer_stability, citation_quality)
    available = {name: score for name, score in values.items() if score is not None}
    total_weight = sum(configured[name] for name in available)
    contributions = []
    for name in DEFAULT_WEIGHTS:
        score = values[name]
        applied_weight = configured[name] / total_weight if score is not None and total_weight else None
        contributions.append(MetricContribution(
            metric=name,
            score=score,
            configured_weight=configured[name],
            applied_weight=applied_weight,
            contribution=(score * applied_weight) if score is not None and applied_weight is not None else None,
            available=score is not None,
            coverage=(
                _number(source_reliability.get("coverage"))
                if name == "source_reliability" else None
            ),
        ))
    composite = sum(item.contribution for item in contributions if item.contribution is not None)
    composite = round(composite * 100, 2) if available else None

    evidence_by_claim = {
        item["claim_id"]: item for item in evidence_support.get("claim_scores", [])
    }
    rave_by_claim = {
        item["claim_id"]: item for item in source_reliability.get("claim_results", [])
    }
    citation_by_claim: dict[str, list[float]] = {}
    if citation_quality:
        for item in citation_quality.get("judgements", []):
            score = _number(item.get("score"))
            if score is not None:
                citation_by_claim.setdefault(item.get("claim_id", ""), []).append(score)

    claims = []
    for claim_id, evidence in evidence_by_claim.items():
        rave = rave_by_claim.get(claim_id, {})
        citation_scores = citation_by_claim.get(claim_id, [])
        claim_values = {
            "evidence_support": _number(evidence.get("score", 0) / 100),
            "source_reliability": _number(rave.get("rave_score")),
            "answer_stability": values["answer_stability"],
            "citation_quality": (sum(citation_scores) / len(citation_scores)) if citation_scores else None,
        }
        claim_weight = sum(configured[name] for name, value in claim_values.items() if value is not None)
        claim_score = (
            sum(value * configured[name] / claim_weight for name, value in claim_values.items() if value is not None) * 100
            if claim_weight else None
        )
        status = evidence.get("status", "UNVERIFIED")
        claims.append(ClaimTrustContribution(
            claim_id=claim_id,
            claim_text=evidence.get("claim_text", ""),
            evidence_support_score=evidence.get("score"),
            source_reliability_score=rave.get("rave_score"),
            citation_quality_score=(sum(citation_scores) / len(citation_scores) * 100) if citation_scores else None,
            answer_stability_score=(values["answer_stability"] * 100) if values["answer_stability"] is not None else None,
            trust_score=claim_score,
            category=trust_category((claim_score / 100) if claim_score is not None else None),
            status=status,
            explanation=_claim_explanation(status, claim_values["source_reliability"] is not None, bool(citation_scores)),
        ))

    return TrustAggregation(
        composite_score=composite,
        category=trust_category((composite / 100) if composite is not None else None),
        metric_contributions=contributions,
        claim_contributions=claims,
        missing_metrics=[name for name, score in values.items() if score is None],
        source_reliability_coverage=_number(source_reliability.get("coverage")) or 0.0,
    )
