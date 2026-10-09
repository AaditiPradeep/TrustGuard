"""Net NLI Evidence Support scoring.

The score is weighted entailment minus weighted contradiction. This avoids the
old ratio's saturation problem, where any claim with support and no retrieved
contradiction was automatically assigned 100/100.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verification.evidence_support import EvidenceSupportResult


@dataclass
class ClaimEvidenceScore:
    claim_id: str
    claim_text: str
    score: float
    status: str
    support_weight: float
    contradiction_weight: float
    supporting_passages: int
    contradicting_passages: int
    irrelevant_passages: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class ResponseEvidenceScore:
    evidence_support_score: float
    total_claims: int
    supported_claims: int
    contradicted_claims: int
    mixed_claims: int
    unverified_claims: int
    claim_scores: list[ClaimEvidenceScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "evidence_support_score": self.evidence_support_score,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "contradicted_claims": self.contradicted_claims,
            "mixed_claims": self.mixed_claims,
            "unverified_claims": self.unverified_claims,
            "claim_scores": [item.to_dict() for item in self.claim_scores],
        }


def _weights(result: EvidenceSupportResult) -> list[float]:
    values = [max(0.0, item.retrieval_weight) for item in result.passage_judgments]
    if sum(values) == 0.0:
        values = [max(0.0, item.retrieval_score) for item in result.passage_judgments]
    total = sum(values)
    if total:
        return [value / total for value in values]
    return [1.0 / len(values)] * len(values) if values else []


def calculate_claim_evidence_score(result: EvidenceSupportResult) -> ClaimEvidenceScore:
    """Compute net NLI evidence strength using hybrid retrieval weights.

    Every passage contributes its full entailment and contradiction
    probabilities. Neutral passages normally contribute little to either
    value, instead of being able to inflate a support-only ratio.
    """
    support = contradiction = 0.0
    supporting = contradicting = irrelevant = 0
    for judgment, weight in zip(result.passage_judgments, _weights(result)):
        support += weight * max(0.0, judgment.entailment_probability)
        contradiction += weight * max(0.0, judgment.contradiction_probability)

        if judgment.verdict == "SUPPORTED":
            supporting += 1
        elif judgment.verdict == "CONTRADICTED":
            contradicting += 1
        else:
            irrelevant += 1

    score = round(max(0.0, support - contradiction) * 100.0, 2)
    status = (
        "MIXED" if supporting and contradicting else
        "SUPPORTED" if supporting else
        "CONTRADICTED" if contradicting else "UNVERIFIED"
    )
    return ClaimEvidenceScore(
        claim_id=result.claim_id,
        claim_text=result.claim_text,
        score=score,
        status=status,
        support_weight=round(support, 4),
        contradiction_weight=round(contradiction, 4),
        supporting_passages=supporting,
        contradicting_passages=contradicting,
        irrelevant_passages=irrelevant,
    )


def calculate_response_evidence_score(results: list[EvidenceSupportResult]) -> ResponseEvidenceScore:
    """Average claim-level NLI Evidence Support scores into the final 0-100 value."""
    scores = [calculate_claim_evidence_score(item) for item in results]
    score = round(sum(item.score for item in scores) / len(scores), 2) if scores else 0.0
    return ResponseEvidenceScore(
        evidence_support_score=score,
        total_claims=len(scores),
        supported_claims=sum(item.status == "SUPPORTED" for item in scores),
        contradicted_claims=sum(item.status == "CONTRADICTED" for item in scores),
        mixed_claims=sum(item.status == "MIXED" for item in scores),
        unverified_claims=sum(item.status == "UNVERIFIED" for item in scores),
        claim_scores=scores,
    )
