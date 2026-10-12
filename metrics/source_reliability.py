"""Authority-weighted, answer-aligned RAVE scoring.

RAVE measures whether retrieved evidence from sources of different authority
tiers supports or contradicts a generated claim. Evidence Support aggregates
every retrieved passage; RAVE first deduplicates them into independent sources.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from data.schema import EvidencePassage
from data.source_registry import baseline_authority
from verification.evidence_support import EvidenceSupportResult, PassageJudgment


NLI_CONFIDENCE_THRESHOLD = 0.55

# Values are human-readable percentages. RAVE stores the final score as a
# fraction in [0, 1] so it remains compatible with the trust aggregator.
_BANDS = {
    "high": {"support": (85.0, 100.0), "contradiction": (0.0, 15.0)},
    "medium": {"support": (75.0, 90.0), "contradiction": (10.0, 25.0)},
    "lower": {"support": (65.0, 80.0), "contradiction": (20.0, 35.0)},
}

_SOURCE_ALIASES = {
    "who": "who", "world health organization": "who",
    "cdc": "cdc", "centers for disease control and prevention": "cdc",
    "openfda": "openfda", "fda": "openfda",
    "food and drug administration": "openfda",
    "medlineplus": "medlineplus", "nih / medlineplus": "medlineplus",
    "nih/medlineplus": "medlineplus", "nih medlineplus": "medlineplus",
    "pubmed": "pubmed", "pubmedqa-context": "pubmedqa_context",
}


def normalize_source_name(source_name: str) -> Optional[str]:
    """Map corpus labels to the source-authority registry key."""
    return _SOURCE_ALIASES.get(" ".join(source_name.lower().split()))


def _passage_source_key(passage: EvidencePassage) -> Optional[str]:
    """Return the source key, excluding unverified FAERS reports."""
    source_key = normalize_source_name(passage.source_name)
    if source_key != "openfda":
        return source_key
    fingerprint = f"{passage.title} {passage.url}".lower()
    return None if "faers" in fingerprint or "drug/event" in fingerprint else source_key


def _base_identifier(identifier: str, passage_id: str) -> str:
    """Collapse dense-retrieval chunks back to their parent document."""
    base = (identifier or passage_id).split(":chunk:", 1)[0].strip()
    return base or passage_id


def source_identity(passage: EvidencePassage) -> Optional[str]:
    """Return the unit treated as independent for RAVE aggregation."""
    source_key = _passage_source_key(passage)
    if source_key is None:
        return None
    if source_key in {"pubmed", "pubmedqa_context"}:
        return f"{source_key}:{_base_identifier(passage.identifier, passage.passage_id)}"
    return source_key


def _direction(judgment: PassageJudgment) -> tuple[str, float]:
    """Classify one passage using the documented 0.55 NLI threshold."""
    entailment = max(0.0, judgment.entailment_probability)
    contradiction = max(0.0, judgment.contradiction_probability)
    if entailment >= NLI_CONFIDENCE_THRESHOLD and entailment > contradiction:
        return "SUPPORTS", entailment
    if contradiction >= NLI_CONFIDENCE_THRESHOLD and contradiction > entailment:
        return "CONTRADICTS", contradiction
    return "INCONCLUSIVE", max(0.0, judgment.neutral_probability)


def _judgment_strength(judgment: PassageJudgment) -> float:
    """Choose the strongest duplicate chunk without counting it twice."""
    status, confidence = _direction(judgment)
    if status != "INCONCLUSIVE":
        return confidence
    return max(confidence, judgment.entailment_probability, judgment.contradiction_probability)


def _tier(baseline: float) -> str:
    if baseline >= 0.90:
        return "high"
    if baseline >= 0.80:
        return "medium"
    return "lower"


def _banded_score(status: str, confidence: float, baseline: float) -> float:
    """Return the RAVE fraction for one source according to its scenario band."""
    if status in {"INCONCLUSIVE", "INTERNALLY_CONFLICTED"}:
        return 0.50

    strength = max(0.0, min(1.0, (confidence - NLI_CONFIDENCE_THRESHOLD) / (1.0 - NLI_CONFIDENCE_THRESHOLD)))
    band = _BANDS[_tier(baseline)]
    if status == "SUPPORTS":
        lower, upper = band["support"]
        percent = lower + (upper - lower) * strength
    elif status == "CONTRADICTS":
        lower, upper = band["contradiction"]
        percent = upper - (upper - lower) * strength
    else:
        raise ValueError(f"Unknown RAVE source status: {status}")
    return percent / 100.0


@dataclass
class SourceRAVEResult:
    claim_id: str
    source_id: str
    source_key: str
    source_name: str
    stance: Optional[str]
    nli_confidence: float
    baseline_authority: Optional[float]
    corroboration: Optional[float]
    rave_score: float
    n_independent_sources: int
    status: str
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClaimRAVEResult:
    claim_id: str
    claim_text: str
    rave_score: Optional[float]
    status: str
    source_results: list[SourceRAVEResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "rave_score": self.rave_score,
            "status": self.status,
            "source_results": [item.to_dict() for item in self.source_results],
        }


@dataclass
class ResponseRAVEResult:
    rave_score: Optional[float]
    total_claims: int
    claims_with_rave: int
    coverage: float
    claim_results: list[ClaimRAVEResult] = field(default_factory=list)
    note: str = (
        "RAVE combines source authority with whether retrieved evidence supports "
        "or contradicts the generated answer."
    )

    def to_dict(self) -> dict:
        return {
            "rave_score": self.rave_score,
            "rave_percent": self.rave_score * 100.0 if self.rave_score is not None else None,
            "total_claims": self.total_claims,
            "claims_with_rave": self.claims_with_rave,
            "coverage": self.coverage,
            "coverage_percent": self.coverage * 100.0,
            "claim_results": [item.to_dict() for item in self.claim_results],
            "note": self.note,
        }


@dataclass
class _SourceCandidate:
    source_id: str
    source_key: str
    source_name: str
    baseline: float
    status: str
    confidence: float
    retrieval_weight: float
    evidence_ids: list[str]


def _document_candidates(result: EvidenceSupportResult) -> Iterable[tuple[EvidencePassage, PassageJudgment]]:
    """Use only the strongest NLI judgement from duplicate document chunks."""
    by_document: dict[tuple[str, str], tuple[EvidencePassage, PassageJudgment, float]] = {}
    judgment_by_id = {item.evidence_id: item for item in result.passage_judgments}
    for evidence_id, passage in result.evidence_passages.items():
        judgment = judgment_by_id.get(evidence_id)
        source_key = _passage_source_key(passage)
        if judgment is None or source_key is None:
            continue
        document_key = (source_key, _base_identifier(passage.identifier, passage.passage_id))
        strength = _judgment_strength(judgment)
        current = by_document.get(document_key)
        if current is None or strength > current[2]:
            by_document[document_key] = (passage, judgment, strength)
    return [(passage, judgment) for passage, judgment, _ in by_document.values()]


def _source_candidates(result: EvidenceSupportResult) -> list[_SourceCandidate]:
    grouped: dict[str, list[tuple[EvidencePassage, PassageJudgment]]] = {}
    for passage, judgment in _document_candidates(result):
        identity = source_identity(passage)
        if identity is not None:
            grouped.setdefault(identity, []).append((passage, judgment))

    candidates: list[_SourceCandidate] = []
    for identity, records in grouped.items():
        passage = records[0][0]
        source_key = _passage_source_key(passage)
        if source_key is None:
            continue
        directions = [(item, *_direction(judgment), judgment) for item, judgment in records]
        strong_statuses = {status for _, status, _, _ in directions if status in {"SUPPORTS", "CONTRADICTS"}}
        selected = max(directions, key=lambda item: _judgment_strength(item[3]))
        if len(strong_statuses) > 1:
            status, confidence = "INTERNALLY_CONFLICTED", 0.0
        else:
            _, status, confidence, _ = selected
        selected_judgment = selected[3]
        retrieval_weight = max(0.0, selected_judgment.retrieval_weight)
        if retrieval_weight == 0.0:
            retrieval_weight = max(0.0, selected_judgment.retrieval_score)
        candidates.append(_SourceCandidate(
            source_id=identity,
            source_key=source_key,
            source_name=passage.source_name,
            baseline=baseline_authority(source_key),
            status=status,
            confidence=confidence,
            retrieval_weight=retrieval_weight,
            evidence_ids=[judgment.evidence_id for _, judgment in records],
        ))
    return candidates


def _corroboration(candidate: _SourceCandidate, candidates: list[_SourceCandidate]) -> Optional[float]:
    """Keep peer agreement as a diagnostic; it no longer changes RAVE."""
    if candidate.status not in {"SUPPORTS", "CONTRADICTS"}:
        return None
    peers = [
        item for item in candidates
        if item.source_id != candidate.source_id and item.status in {"SUPPORTS", "CONTRADICTS"}
    ]
    if not peers:
        return None
    total = sum(item.retrieval_weight for item in peers)
    agreeing = sum(item.retrieval_weight for item in peers if item.status == candidate.status)
    return agreeing / total if total else None


def calculate_claim_rave(result: EvidenceSupportResult) -> ClaimRAVEResult:
    """Calculate the hybrid-retrieval-weighted RAVE score for one claim."""
    candidates = _source_candidates(result)
    source_results: list[SourceRAVEResult] = []
    for candidate in candidates:
        peers = [item for item in candidates if item.source_id != candidate.source_id]
        agreement = _corroboration(candidate, candidates)
        source_results.append(SourceRAVEResult(
            claim_id=result.claim_id,
            source_id=candidate.source_id,
            source_key=candidate.source_key,
            source_name=candidate.source_name,
            stance=candidate.status if candidate.status in {"SUPPORTS", "CONTRADICTS"} else None,
            nli_confidence=round(candidate.confidence, 4),
            baseline_authority=candidate.baseline,
            corroboration=round(agreement, 4) if agreement is not None else None,
            rave_score=round(_banded_score(candidate.status, candidate.confidence, candidate.baseline), 4),
            n_independent_sources=len(peers),
            status=candidate.status,
            evidence_ids=candidate.evidence_ids,
        ))

    if not source_results:
        return ClaimRAVEResult(
            result.claim_id,
            result.claim_text,
            None,
            "NO_RETRIEVED_SOURCE_EVIDENCE",
            [],
        )

    weights = [max(0.0, item.retrieval_weight) for item in candidates]
    total_weight = sum(weights)
    if total_weight == 0.0:
        weights = [1.0 / len(source_results)] * len(source_results)
    else:
        weights = [weight / total_weight for weight in weights]
    claim_score = sum(item.rave_score * weight for item, weight in zip(source_results, weights))
    statuses = {item.status for item in source_results}
    claim_status = (
        "INTERNALLY_CONFLICTED" if "INTERNALLY_CONFLICTED" in statuses else
        "MIXED" if {"SUPPORTS", "CONTRADICTS"}.issubset(statuses) else
        "SUPPORTS" if "SUPPORTS" in statuses else
        "CONTRADICTS" if "CONTRADICTS" in statuses else
        "INCONCLUSIVE"
    )
    return ClaimRAVEResult(
        result.claim_id, result.claim_text, round(claim_score, 4), claim_status, source_results
    )


def calculate_response_rave(evidence_results: list[EvidenceSupportResult]) -> ResponseRAVEResult:
    """Calculate answer-level RAVE as the mean of its claim scores."""
    claims = [calculate_claim_rave(result) for result in evidence_results]
    scored = [item.rave_score for item in claims if item.rave_score is not None]
    score = round(sum(scored) / len(scored), 4) if scored else None
    total = len(claims)
    return ResponseRAVEResult(
        rave_score=score,
        total_claims=total,
        claims_with_rave=len(scored),
        coverage=(len(scored) / total) if total else 0.0,
        claim_results=claims,
    )
