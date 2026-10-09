"""Claim-level RAVE source reliability.

RAVE measures the authority and independent corroboration of the retrieved
evidence.  It is intentionally separate from Evidence Support: reliable
sources can consistently contradict an answer, which should lower Evidence
Support but not make those sources unreliable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from data.schema import EvidencePassage
from data.source_registry import baseline_authority
from verification.evidence_support import EvidenceSupportResult, PassageJudgment


DEFAULT_ALPHA = 0.7

# Sensitivity scenarios test whether RAVE conclusions change under reasonable
# alternative authority-tier assumptions. They do not replace the registry's
# current configuration used by the integrated pipeline.
DEFAULT_SENSITIVITY_SCENARIOS = {
    "conservative": {
        "who": 0.90, "openfda": 0.90, "cdc": 0.90,
        "medlineplus": 0.80,
        "pubmed": 0.65, "pubmedqa_context": 0.65,
    },
    "current": {
        "who": 0.95, "openfda": 0.95, "cdc": 0.95,
        "medlineplus": 0.85,
        "pubmed": 0.75, "pubmedqa_context": 0.70,
    },
    "literature_favouring": {
        "who": 0.95, "openfda": 0.95, "cdc": 0.95,
        "medlineplus": 0.85,
        "pubmed": 0.85, "pubmedqa_context": 0.85,
    },
}

_SOURCE_ALIASES = {
    "who": "who",
    "world health organization": "who",
    "cdc": "cdc",
    "centers for disease control and prevention": "cdc",
    "openfda": "openfda",
    "fda": "openfda",
    "food and drug administration": "openfda",
    "medlineplus": "medlineplus",
    "nih / medlineplus": "medlineplus",
    "nih/medlineplus": "medlineplus",
    "nih medlineplus": "medlineplus",
    "pubmed": "pubmed",
    "pubmedqa-context": "pubmedqa_context",
}


def normalize_source_name(source_name: str) -> Optional[str]:
    """Map corpus labels to the registry key used for baseline authority."""
    cleaned = " ".join(source_name.lower().split())
    return _SOURCE_ALIASES.get(cleaned)


def _passage_source_key(passage: EvidencePassage) -> Optional[str]:
    """Return the rated source key, excluding unverified FAERS reports.

    The present corpus only pulls FDA structured labels.  If a future corpus
    adds openFDA drug-event/FAERS records, they must receive their own policy
    and cannot silently inherit the regulatory-label authority score.
    """
    source_key = normalize_source_name(passage.source_name)
    if source_key != "openfda":
        return source_key
    fingerprint = f"{passage.title} {passage.url}".lower()
    if "faers" in fingerprint or "drug/event" in fingerprint:
        return None
    return source_key


def _base_identifier(identifier: str, passage_id: str) -> str:
    """Collapse dense-retrieval chunks back to their parent document."""
    base = (identifier or passage_id).split(":chunk:", 1)[0].strip()
    return base or passage_id


def source_identity(passage: EvidencePassage) -> Optional[str]:
    """Return the unit treated as independent for corroboration.

    Individual PubMed PMIDs are independent.  For the first version, WHO,
    CDC, MedlinePlus, and FDA label evidence are treated as organisation-level
    groups because their documents can share editorial guidance.
    """
    source_key = _passage_source_key(passage)
    if source_key is None:
        return None
    if source_key in {"pubmed", "pubmedqa_context"}:
        return f"{source_key}:{_base_identifier(passage.identifier, passage.passage_id)}"
    return source_key


def _stance_confidence(judgment: PassageJudgment) -> tuple[Optional[str], float]:
    verdict = judgment.verdict.upper()
    if verdict == "SUPPORTED":
        return "SUPPORTED", max(0.0, judgment.entailment_probability)
    if verdict == "CONTRADICTED":
        return "CONTRADICTED", max(0.0, judgment.contradiction_probability)
    return None, 0.0


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
    rave_score: Optional[float]
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
        "RAVE measures the reliability and agreement of retrieved evidence; "
        "it does not determine whether the generated answer is supported."
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
    stance: Optional[str]
    confidence: float
    evidence_ids: list[str]
    internally_conflicted: bool = False


def _document_candidates(
    result: EvidenceSupportResult,
) -> Iterable[tuple[EvidencePassage, PassageJudgment]]:
    """Use only the strongest NLI judgement from duplicate document chunks."""
    by_document: dict[tuple[str, str], tuple[EvidencePassage, PassageJudgment, float]] = {}
    judgment_by_id = {item.evidence_id: item for item in result.passage_judgments}
    for evidence_id, passage in result.evidence_passages.items():
        judgment = judgment_by_id.get(evidence_id)
        if judgment is None:
            continue
        source_key = _passage_source_key(passage)
        if source_key is None:
            continue
        document_key = (source_key, _base_identifier(passage.identifier, passage.passage_id))
        _, confidence = _stance_confidence(judgment)
        current = by_document.get(document_key)
        if current is None or confidence > current[2]:
            by_document[document_key] = (passage, judgment, confidence)
    return [(passage, judgment) for passage, judgment, _ in by_document.values()]


def _baseline(source_key: str, overrides: Optional[dict[str, float]]) -> float:
    """Use an analysis override when supplied, otherwise the registry value."""
    if overrides is not None and source_key in overrides:
        value = float(overrides[source_key])
        if not 0.0 <= value <= 1.0:
            raise ValueError("baseline overrides must be between 0 and 1.")
        return value
    return baseline_authority(source_key)


def _source_candidates(
    result: EvidenceSupportResult,
    baseline_overrides: Optional[dict[str, float]] = None,
) -> list[_SourceCandidate]:
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
        non_neutral = [
            (item, *_stance_confidence(judgment))
            for item, judgment in records
            if _stance_confidence(judgment)[0] is not None
        ]
        stances = {stance for _, stance, _ in non_neutral}
        evidence_ids = [judgment.evidence_id for _, judgment in records]
        if len(stances) > 1:
            candidates.append(_SourceCandidate(
                source_id=identity,
                source_key=source_key,
                source_name=passage.source_name,
                baseline=_baseline(source_key, baseline_overrides),
                stance=None,
                confidence=0.0,
                evidence_ids=evidence_ids,
                internally_conflicted=True,
            ))
            continue
        if not non_neutral:
            candidates.append(_SourceCandidate(
                source_id=identity,
                source_key=source_key,
                source_name=passage.source_name,
                baseline=_baseline(source_key, baseline_overrides),
                stance=None,
                confidence=0.0,
                evidence_ids=evidence_ids,
            ))
            continue
        _, stance, confidence = max(non_neutral, key=lambda item: item[2])
        candidates.append(_SourceCandidate(
            source_id=identity,
            source_key=source_key,
            source_name=passage.source_name,
            baseline=_baseline(source_key, baseline_overrides),
            stance=stance,
            confidence=confidence,
            evidence_ids=evidence_ids,
        ))
    return candidates


def calculate_claim_rave(
    result: EvidenceSupportResult,
    alpha: float = DEFAULT_ALPHA,
    baseline_overrides: Optional[dict[str, float]] = None,
) -> ClaimRAVEResult:
    """Calculate source RAVE scores and their weighted claim-level mean."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    candidates = _source_candidates(result, baseline_overrides=baseline_overrides)
    valid = [item for item in candidates if item.stance is not None]
    source_results: list[SourceRAVEResult] = []

    for candidate in candidates:
        if candidate.internally_conflicted:
            source_results.append(SourceRAVEResult(
                claim_id=result.claim_id,
                source_id=candidate.source_id,
                source_key=candidate.source_key,
                source_name=candidate.source_name,
                stance=None,
                nli_confidence=0.0,
                baseline_authority=candidate.baseline,
                corroboration=None,
                rave_score=None,
                n_independent_sources=0,
                status="INTERNALLY_CONFLICTED",
                evidence_ids=candidate.evidence_ids,
            ))
            continue
        if candidate.stance is None:
            source_results.append(SourceRAVEResult(
                claim_id=result.claim_id,
                source_id=candidate.source_id,
                source_key=candidate.source_key,
                source_name=candidate.source_name,
                stance=None,
                nli_confidence=0.0,
                baseline_authority=candidate.baseline,
                corroboration=None,
                rave_score=None,
                n_independent_sources=0,
                status="NEUTRAL",
                evidence_ids=candidate.evidence_ids,
            ))
            continue

        peers = [item for item in valid if item.source_id != candidate.source_id]
        if not peers:
            corroboration = None
            # Missing independent evidence is not disagreement. Keep the
            # authority visible, but do not present it as a full RAVE score.
            score = None
            status = "UNCORROBORATED"
        else:
            peer_weight = sum(item.baseline * item.confidence for item in peers)
            agreement = sum(
                item.baseline * item.confidence
                for item in peers
                if item.stance == candidate.stance
            )
            corroboration = agreement / peer_weight if peer_weight else None
            score = alpha * candidate.baseline + (1.0 - alpha) * (corroboration or 0.0)
            status = "CORROBORATED" if corroboration >= 0.5 else "DISAGREED"
        source_results.append(SourceRAVEResult(
            claim_id=result.claim_id,
            source_id=candidate.source_id,
            source_key=candidate.source_key,
            source_name=candidate.source_name,
            stance=candidate.stance,
            nli_confidence=round(candidate.confidence, 4),
            baseline_authority=candidate.baseline,
            corroboration=round(corroboration, 4) if corroboration is not None else None,
            rave_score=round(score, 4) if score is not None else None,
            n_independent_sources=len(peers),
            status=status,
            evidence_ids=candidate.evidence_ids,
        ))

    scored = [item for item in source_results if item.rave_score is not None]
    if not scored:
        return ClaimRAVEResult(
            claim_id=result.claim_id,
            claim_text=result.claim_text,
            rave_score=None,
            status="NO_USABLE_SOURCE_EVIDENCE",
            source_results=source_results,
        )
    total_confidence = sum(item.nli_confidence for item in scored)
    claim_score = (
        sum(item.rave_score * item.nli_confidence for item in scored) / total_confidence
        if total_confidence else None
    )
    return ClaimRAVEResult(
        claim_id=result.claim_id,
        claim_text=result.claim_text,
        rave_score=round(claim_score, 4) if claim_score is not None else None,
        status="COMPUTED",
        source_results=source_results,
    )


def calculate_response_rave(
    evidence_results: list[EvidenceSupportResult],
    alpha: float = DEFAULT_ALPHA,
    baseline_overrides: Optional[dict[str, float]] = None,
) -> ResponseRAVEResult:
    """Calculate RAVE across claims, excluding claims with no usable evidence."""
    claims = [
        calculate_claim_rave(
            result, alpha=alpha, baseline_overrides=baseline_overrides
        )
        for result in evidence_results
    ]
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


def run_rave_sensitivity_analysis(
    evidence_results: list[EvidenceSupportResult],
    alpha: float = DEFAULT_ALPHA,
    scenarios: Optional[dict[str, dict[str, float]]] = None,
) -> dict:
    """Recalculate RAVE under documented baseline-authority scenarios.

    This supports final-review analysis: stable results across these scenarios
    are evidence that the conclusion is not dependent on one exact baseline.
    """
    active_scenarios = scenarios or DEFAULT_SENSITIVITY_SCENARIOS
    reports = {
        name: calculate_response_rave(
            evidence_results,
            alpha=alpha,
            baseline_overrides=overrides,
        ).to_dict()
        for name, overrides in active_scenarios.items()
    }
    scores = [report["rave_score"] for report in reports.values() if report["rave_score"] is not None]
    return {
        "alpha": alpha,
        "scenarios": reports,
        "rave_score_range": (
            {"minimum": min(scores), "maximum": max(scores)} if scores else None
        ),
    }
