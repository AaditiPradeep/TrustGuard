"""
TrustGuard - Evidence Support Score

Calculates a deterministic Evidence Support score from
passage-level evidence judgments.

For each claim:

    ES_claim =
        sum(relevance of SUPPORTED passages)
        ------------------------------------
        sum(relevance of SUPPORTED passages)
        +
        sum(relevance of CONTRADICTED passages)

IRRELEVANT passages are excluded because they provide
neither supporting nor contradicting evidence.

The score is converted to a 0-100 scale.

If there is no supporting or contradicting evidence,
the claim is marked UNVERIFIED and receives a score of 0.

The response-level Evidence Support score is the mean
of all claim-level Evidence Support scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verification.evidence_support import EvidenceSupportResult


# ============================================================
# RESULT DATA CLASSES
# ============================================================

@dataclass
class ClaimEvidenceScore:
    """
    Numerical Evidence Support result for one claim.
    """

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
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "score": self.score,
            "status": self.status,
            "support_weight": self.support_weight,
            "contradiction_weight": self.contradiction_weight,
            "supporting_passages": self.supporting_passages,
            "contradicting_passages": self.contradicting_passages,
            "irrelevant_passages": self.irrelevant_passages,
        }


@dataclass
class ResponseEvidenceScore:
    """
    Evidence Support result for the complete LLM response.
    """

    evidence_support_score: float

    total_claims: int
    supported_claims: int
    contradicted_claims: int
    mixed_claims: int
    unverified_claims: int

    claim_scores: list[ClaimEvidenceScore] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return {
            "evidence_support_score":
                self.evidence_support_score,

            "total_claims":
                self.total_claims,

            "supported_claims":
                self.supported_claims,

            "contradicted_claims":
                self.contradicted_claims,

            "mixed_claims":
                self.mixed_claims,

            "unverified_claims":
                self.unverified_claims,

            "claim_scores": [
                result.to_dict()
                for result in self.claim_scores
            ],
        }


# ============================================================
# CLAIM-LEVEL SCORE
# ============================================================

def calculate_claim_evidence_score(
    result: EvidenceSupportResult,
) -> ClaimEvidenceScore:
    """
    Calculate Evidence Support for a single claim.

    Formula:

        ES_claim =
            support_weight
            ------------------------------
            support_weight + contradiction_weight

    where:

        support_weight =
            sum retrieval relevance scores of
            SUPPORTED passages

        contradiction_weight =
            sum retrieval relevance scores of
            CONTRADICTED passages

    IRRELEVANT passages do not participate in the
    numerator or denominator.
    """

    support_weight = 0.0
    contradiction_weight = 0.0

    supporting_count = 0
    contradicting_count = 0
    irrelevant_count = 0

    for judgment in result.passage_judgments:

        verdict = judgment.verdict.upper()

        relevance = max(
            0.0,
            float(judgment.retrieval_score),
        )

        if verdict == "SUPPORTED":

            support_weight += relevance
            supporting_count += 1

        elif verdict == "CONTRADICTED":

            contradiction_weight += relevance
            contradicting_count += 1

        else:

            irrelevant_count += 1

    # --------------------------------------------------------
    # Determine claim status
    # --------------------------------------------------------

    if (
        supporting_count > 0
        and contradicting_count > 0
    ):
        status = "MIXED"

    elif supporting_count > 0:
        status = "SUPPORTED"

    elif contradicting_count > 0:
        status = "CONTRADICTED"

    else:
        status = "UNVERIFIED"

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    usable_weight = (
        support_weight
        + contradiction_weight
    )

    if usable_weight == 0:

        score = 0.0

    else:

        score = (
            support_weight
            / usable_weight
        ) * 100.0

    score = round(
        score,
        2,
    )

    return ClaimEvidenceScore(
        claim_id=result.claim_id,
        claim_text=result.claim_text,
        score=score,
        status=status,
        support_weight=round(
            support_weight,
            4,
        ),
        contradiction_weight=round(
            contradiction_weight,
            4,
        ),
        supporting_passages=supporting_count,
        contradicting_passages=contradicting_count,
        irrelevant_passages=irrelevant_count,
    )


# ============================================================
# RESPONSE-LEVEL SCORE
# ============================================================

def calculate_response_evidence_score(
    results: list[EvidenceSupportResult],
) -> ResponseEvidenceScore:
    """
    Calculate Evidence Support for an entire LLM response.

    Response score:

        ES_response =
            average of all claim Evidence Support scores

    Every extracted factual claim contributes equally.
    """

    if not results:

        return ResponseEvidenceScore(
            evidence_support_score=0.0,
            total_claims=0,
            supported_claims=0,
            contradicted_claims=0,
            mixed_claims=0,
            unverified_claims=0,
            claim_scores=[],
        )

    claim_scores = [
        calculate_claim_evidence_score(result)
        for result in results
    ]

    response_score = sum(
        result.score
        for result in claim_scores
    ) / len(claim_scores)

    response_score = round(
        response_score,
        2,
    )

    supported = sum(
        result.status == "SUPPORTED"
        for result in claim_scores
    )

    contradicted = sum(
        result.status == "CONTRADICTED"
        for result in claim_scores
    )

    mixed = sum(
        result.status == "MIXED"
        for result in claim_scores
    )

    unverified = sum(
        result.status == "UNVERIFIED"
        for result in claim_scores
    )

    return ResponseEvidenceScore(
        evidence_support_score=response_score,
        total_claims=len(claim_scores),
        supported_claims=supported,
        contradicted_claims=contradicted,
        mixed_claims=mixed,
        unverified_claims=unverified,
        claim_scores=claim_scores,
    )


# ============================================================
# DISPLAY
# ============================================================

def print_evidence_support_score(
    result: ResponseEvidenceScore,
) -> None:
    """
    Print Evidence Support results in a readable format.
    """

    print("\n")
    print("=" * 70)
    print("EVIDENCE SUPPORT SCORE")
    print("=" * 70)

    for i, claim in enumerate(
        result.claim_scores,
        start=1,
    ):

        print(
            f"\nClaim {i}: "
            f"{claim.claim_text}"
        )

        print(
            f"Status               : "
            f"{claim.status}"
        )

        print(
            f"Evidence Support Score: "
            f"{claim.score:.2f}"
        )

        print(
            f"Support Weight       : "
            f"{claim.support_weight:.4f}"
        )

        print(
            f"Contradiction Weight : "
            f"{claim.contradiction_weight:.4f}"
        )

        print(
            f"Supporting Passages  : "
            f"{claim.supporting_passages}"
        )

        print(
            f"Contradicting Passages: "
            f"{claim.contradicting_passages}"
        )

        print(
            f"Irrelevant Passages  : "
            f"{claim.irrelevant_passages}"
        )

    print("\n")
    print("-" * 70)

    print(
        f"Total Claims        : "
        f"{result.total_claims}"
    )

    print(
        f"Supported Claims    : "
        f"{result.supported_claims}"
    )

    print(
        f"Contradicted Claims : "
        f"{result.contradicted_claims}"
    )

    print(
        f"Mixed Claims        : "
        f"{result.mixed_claims}"
    )

    print(
        f"Unverified Claims   : "
        f"{result.unverified_claims}"
    )

    print("-" * 70)

    print(
        f"\nFINAL EVIDENCE SUPPORT SCORE: "
        f"{result.evidence_support_score:.2f}/100"
    )

    print("=" * 70)