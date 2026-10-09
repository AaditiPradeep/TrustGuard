"""Integrated citation-quality flow for TrustGuard.

This module connects Person 3's citation extractor/resolver with the MiniLM +
NLI citation judge. It intentionally keeps citation *coverage* separate from
citation *quality*: a response with no citations should not be mistaken for a
response whose citations were all judged irrelevant.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from collections import Counter
from math import log, sqrt
import re
from typing import TYPE_CHECKING, Sequence

from citation.citation_extractor import (
    extract_citations,
    map_claims_to_citations,
    resolve_citations,
)
from data.schema import Claim, EvidencePassage

if TYPE_CHECKING:
    from verification.evidence_support import EvidenceSupportResult


def _as_dicts(judgements) -> list[dict]:
    """Convert either judge dataclasses or fallback dictionaries to dicts."""
    return [asdict(item) if is_dataclass(item) else dict(item) for item in judgements]


def _tfidf_fallback(pairs: Sequence[dict]) -> dict:
    """Score claim/source overlap without transformer or native DLL dependencies.

    This is deliberately a lexical relevance fallback.  It keeps the report
    usable when Windows policy blocks the MiniLM/DeBERTa dependencies, but it
    does not claim to be an NLI entailment judgement.
    """
    judgements = []
    for pair in pairs:
        claim_tokens = re.findall(r"[a-z0-9]+", pair["claim"].lower())
        source_tokens = re.findall(r"[a-z0-9]+", pair["source_text"].lower())
        claim_counts, source_counts = Counter(claim_tokens), Counter(source_tokens)
        vocabulary = set(claim_counts) | set(source_counts)
        # Two-document TF-IDF cosine, implemented here to remain available
        # when the environment blocks compiled ML packages.
        weights = {
            token: log(3 / (1 + int(token in claim_counts) + int(token in source_counts))) + 1
            for token in vocabulary
        }
        dot = sum(
            claim_counts[token] * source_counts[token] * weights[token] ** 2
            for token in vocabulary
        )
        claim_norm = sqrt(sum((count * weights[token]) ** 2 for token, count in claim_counts.items()))
        source_norm = sqrt(sum((count * weights[token]) ** 2 for token, count in source_counts.items()))
        score = dot / (claim_norm * source_norm) if claim_norm and source_norm else 0.0

        label = "DIRECT" if score >= 0.20 else "INDIRECT" if score >= 0.05 else "IRRELEVANT"
        judgements.append({
            "claim_id": pair["claim_id"],
            "claim_text": pair["claim"],
            "citation_id": pair["citation_id"],
            "source_name": pair["source_name"],
            "source_text": pair["source_text"],
            "label": label,
            "score": score,
            "claim_source_similarity": score,
            "entailment_probability": None,
            "contradiction_probability": None,
            "neutral_probability": None,
            "rationale": "TF-IDF lexical fallback; MiniLM/DeBERTa NLI was unavailable.",
        })

    quality = sum(item["score"] for item in judgements) / len(judgements) if judgements else 0.0
    return {"judgements": judgements, "citation_quality_score": quality}


def _judge_with_fallback(pairs: Sequence[dict], judge=None) -> tuple[dict, str]:
    """Use the semantic/NLI judge, falling back only for blocked dependencies."""
    if not pairs:
        return {"judgements": [], "citation_quality_score": 0.0}, "not_applicable"
    try:
        if judge is None:
            from citation.minilm_judge import MiniLMCitationJudge
            judge = MiniLMCitationJudge()
        return judge.judge_claims(pairs), "minilm_deberta_nli"
    except (ImportError, OSError, RuntimeError):
        return _tfidf_fallback(pairs), "tfidf_lexical_fallback"


def evaluate_citations(
    response_text: str,
    claims: Sequence[Claim],
    sources: Sequence[EvidencePassage],
    question: str = "",
    judge=None,
) -> dict:
    """Extract, resolve, map, and score citations in one call.

    Parameters
    ----------
    response_text:
        The LLM answer containing citation markers such as ``[1]`` or URLs.
    claims:
        Atomic claims extracted from the same answer.
    sources:
        The ordered source list that the answer's numbered references point to.
    question:
        Original user question. Retained in the mapped records for reporting.
    judge:
        Optional pre-created ``MiniLMCitationJudge``. It is created lazily when
        omitted, so importing the rest of TrustGuard does not load large models.

    Returns
    -------
    dict
        Citation quality, citation coverage, counts, and per-pair judgements.
    """

    extracted = extract_citations(response_text)
    resolved = resolve_citations(extracted, sources)
    pairs = map_claims_to_citations(
        response_text=response_text,
        claims=claims,
        citations=resolved,
        question=question,
    )

    cited_claim_ids = {pair.claim_id for pair in pairs}
    coverage = (len(cited_claim_ids) / len(claims)) if claims else 0.0

    # No citation markers is different from poor citation quality.
    if not extracted:
        return {
            "available": False,
            "citation_quality_score": None,
            "citation_quality_percent": None,
            "citation_coverage": 0.0,
            "citation_coverage_percent": 0.0,
            "citation_count": 0,
            "mapped_pair_count": 0,
            "counts": {
                "DIRECT": 0,
                "INDIRECT": 0,
                "IRRELEVANT": 0,
                "MISATTRIBUTED": 0,
            },
            "judgements": [],
        }

    resolved_pairs = [pair for pair in pairs if pair.source_text]
    unresolved_pairs = [pair for pair in pairs if not pair.source_text]

    judge_input = [
        {
            "claim_id": pair.claim_id,
            "claim": pair.claim_text,
            "citation_id": pair.citation_id,
            "source_name": pair.source_name,
            "source_text": pair.source_text,
        }
        for pair in resolved_pairs
    ]

    judged_report, method = _judge_with_fallback(judge_input, judge)
    judgement_dicts = _as_dicts(judged_report["judgements"])

    # A citation that cannot be resolved to the source list is an attribution
    # failure. Keep it explicit instead of feeding an empty source to MiniLM.
    for pair in unresolved_pairs:
        judgement_dicts.append(
            {
                "claim_id": pair.claim_id,
                "claim_text": pair.claim_text,
                "citation_id": pair.citation_id,
                "source_name": pair.source_name,
                "source_text": "",
                "label": "MISATTRIBUTED",
                "score": 0.0,
                "claim_source_similarity": 0.0,
                "entailment_probability": 0.0,
                "contradiction_probability": 0.0,
                "neutral_probability": 0.0,
                "rationale": "The citation could not be resolved to the supplied source list.",
            }
        )

    scores = [float(j["score"]) for j in judgement_dicts]
    quality = sum(scores) / len(scores) if scores else 0.0

    counts = {
        "DIRECT": sum(j["label"] == "DIRECT" for j in judgement_dicts),
        "INDIRECT": sum(j["label"] == "INDIRECT" for j in judgement_dicts),
        "IRRELEVANT": sum(j["label"] == "IRRELEVANT" for j in judgement_dicts),
        "MISATTRIBUTED": sum(j["label"] == "MISATTRIBUTED" for j in judgement_dicts),
    }

    return {
        "available": True,
        "method": method,
        "citation_quality_score": quality,
        "citation_quality_percent": quality * 100.0,
        "citation_coverage": coverage,
        "citation_coverage_percent": coverage * 100.0,
        "citation_count": len(extracted),
        "mapped_pair_count": len(pairs),
        "counts": counts,
        "judgements": judgement_dicts,
    }


def evaluate_retrieved_citations(
    evidence_results: Sequence["EvidenceSupportResult"],
    question: str = "",
    judge=None,
) -> dict:
    """Score every passage retrieved during Evidence Support as a citation.

    This is the automatic bridge between the two metrics.  A claim is paired
    with each of its retrieved evidence passages, so citation quality is
    calculated for the same evidence records used by Evidence Support.  No
    hand-written ``citation_sources`` list is required.
    """
    pairs = []
    evidence_verdicts: dict[tuple[str, str], str] = {}

    for evidence_result in evidence_results:
        verdict_by_id = {
            passage.evidence_id: passage.verdict
            for passage in evidence_result.passage_judgments
        }
        for evidence_id, source in evidence_result.evidence_passages.items():
            pairs.append({
                "claim_id": evidence_result.claim_id,
                "claim": evidence_result.claim_text,
                "citation_id": f"retrieved-{evidence_id}",
                "source_name": source.source_name,
                "source_text": source.text,
            })
            evidence_verdicts[(evidence_result.claim_id, evidence_id)] = (
                verdict_by_id.get(evidence_id, "UNJUDGED")
            )

    judged_report, method = _judge_with_fallback(pairs, judge)
    judgements = _as_dicts(judged_report["judgements"])

    for item in judgements:
        evidence_id = item["citation_id"].removeprefix("retrieved-")
        item["evidence_id"] = evidence_id
        item["evidence_support_verdict"] = evidence_verdicts.get(
            (item["claim_id"], evidence_id), "UNJUDGED"
        )

    counts = {
        label: sum(item["label"] == label for item in judgements)
        for label in ("DIRECT", "INDIRECT", "IRRELEVANT", "MISATTRIBUTED")
    }
    cited_claim_ids = {item["claim_id"] for item in pairs}
    coverage = (
        len(cited_claim_ids) / len(evidence_results)
        if evidence_results else 0.0
    )
    return {
        "available": bool(pairs),
        "mode": "retrieved_evidence",
        "method": method,
        "citation_quality_score": judged_report["citation_quality_score"],
        "citation_quality_percent": judged_report["citation_quality_score"] * 100.0,
        "citation_count": len(pairs),
        "mapped_pair_count": len(pairs),
        "citation_coverage": coverage,
        "citation_coverage_percent": coverage * 100.0,
        "counts": counts,
        "judgements": judgements,
    }
