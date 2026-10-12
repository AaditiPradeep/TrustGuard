"""Banded NLI Evidence Support scoring.

Evidence Support uses relevance-weighted DeBERTa entailment and contradiction
probabilities. Clear net support maps to 70-100, clear net contradiction maps
to 0-30, and mixed or inconclusive evidence remains in the 30-70 middle band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verification.evidence_support import EvidenceSupportResult


NET_MARGIN_THRESHOLD = 0.20


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


def _banded_evidence_score(
    support: float,
    contradiction: float,
    supporting_passages: int,
    contradicting_passages: int,
) -> tuple[float, str]:
    """Map the net NLI margin into support, mixed, and contradiction bands.

    A margin of +0.20 is the point where support enters the 70-100 band;
    -0.20 is the equivalent point for the 0-30 contradiction band. The middle
    band retains uncertainty instead of treating any positive signal as proof.
    """
    margin = support - contradiction
    if margin >= NET_MARGIN_THRESHOLD:
        strength = min(1.0, (margin - NET_MARGIN_THRESHOLD) / (1.0 - NET_MARGIN_THRESHOLD))
        return 70.0 + 30.0 * strength, "SUPPORTED"
    if margin <= -NET_MARGIN_THRESHOLD:
        strength = min(1.0, (-margin - NET_MARGIN_THRESHOLD) / (1.0 - NET_MARGIN_THRESHOLD))
        return 30.0 - 30.0 * strength, "CONTRADICTED"
    if supporting_passages == 0 and contradicting_passages == 0:
        return 50.0, "UNVERIFIED"
    return max(30.0, min(70.0, 50.0 + 100.0 * margin)), "MIXED"


def calculate_claim_evidence_score(result: EvidenceSupportResult) -> ClaimEvidenceScore:
    """Compute banded, relevance-weighted NLI support for one claim."""
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

    score, status = _banded_evidence_score(
        support, contradiction, supporting, contradicting
    )
    return ClaimEvidenceScore(
        claim_id=result.claim_id,
        claim_text=result.claim_text,
        score=round(score, 2),
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
