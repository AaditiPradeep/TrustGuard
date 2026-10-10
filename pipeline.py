"""TrustGuard integrated pipeline.

Connects the three parallel work streams:

Person 1
    claim extraction -> evidence verification -> numerical Evidence Support

Person 2
    hybrid TF-IDF + dense retrieval -> Answer Stability

Person 3
    citation extraction/resolution -> citation-quality judgement

Source Reliability / RAVE scores authority and independent corroboration of
the retrieved evidence. It remains separate from Evidence Support, which
measures whether that evidence supports the generated answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional, Sequence

from data.schema import EvidencePassage
from metrics.answer_stability import answer_stability, compute_answer_stability
from metrics.evidence_support_score import calculate_response_evidence_score
from metrics.source_reliability import (
    calculate_response_rave,
    run_rave_sensitivity_analysis,
)
from retrieval.hybrid import HybridRetriever
from verification.claim_extractor import extract_claims
from verification.evidence_support import verify_claims
from trust_aggregator import aggregate_trust
from decision_engine import decide_response
from trust_report import build_trust_report


@dataclass
class TrustGuardResult:
    question: str
    answer: str
    answer_origin: str
    claims: list[dict]
    evidence_support: dict
    source_reliability: dict
    answer_stability: Optional[dict]
    citation_quality: Optional[dict]
    trust_aggregation: Optional[dict] = None
    decision: Optional[dict] = None
    trust_report: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def build_retriever(
    corpus_path: str = "cache/evidence_corpus.jsonl",
    dense_index_dir: str = "cache/dense_evidence_index",
) -> HybridRetriever:
    """Load Person 2's hybrid TF-IDF + dense retriever."""
    return HybridRetriever.from_jsonl(
        corpus_path,
        dense_index_dir=dense_index_dir,
    )


def run_trustguard(
    question: str,
    answer: Optional[str] = None,
    *,
    response_id: str = "response_001",
    retriever: Optional[HybridRetriever] = None,
    corpus_path: str = "cache/evidence_corpus.jsonl",
    dense_index_dir: str = "cache/dense_evidence_index",
    top_k: int = 5,
    stability_responses: Optional[Sequence[str]] = None,
    citation_sources: Optional[Sequence[EvidencePassage]] = None,
    citation_judge=None,
    answer_generator: Optional[Callable[[str], str]] = None,
    answer_model_name: str = "qwen3:1.7b",
    stability_generator: Optional[Callable[[str], str]] = None,
    stability_model_name: str = "qwen3:1.7b",
    stability_samples: int = 3,
) -> TrustGuardResult:
    """Run every currently implemented TrustGuard component together.

    ``answer`` is the response being evaluated. When omitted, the configured
    Qwen instruction model generates it from ``question`` before verification.

    ``stability_responses`` should contain >=2 independently sampled answers
    to the same question. If omitted, local Ollama samples ``stability_samples``
    varied answers using ``stability_model_name``.

    ``citation_sources`` can be supplied for citations already present in the
    answer. If it is omitted, Citation Quality automatically evaluates every
    passage retrieved by Evidence Support, using the exact source records from
    that verification step.
    """

    if answer is None:
        if answer_generator is None:
            from generation.qwen_answer import generate_answer
            answer_generator = lambda prompt: generate_answer(
                prompt, model_name=answer_model_name
            )
        print(f"Generating answer locally with Ollama ({answer_model_name})...")
        answer = answer_generator(question)
        answer_origin = answer_model_name
    else:
        answer = answer.strip()
        answer_origin = "provided"

    if not answer:
        raise ValueError("answer must not be empty.")

    if retriever is None:
        retriever = build_retriever(
            corpus_path=corpus_path,
            dense_index_dir=dense_index_dir,
        )

    # Person 1: claim extraction.
    claims = extract_claims(
        question=question,
        answer=answer,
        response_id=response_id,
    )

    # Person 1 + Person 2: evidence verification over hybrid retrieval.
    evidence_results = verify_claims(
        claims=claims,
        retriever=retriever,
        top_k=top_k,
    )
    evidence_score = calculate_response_evidence_score(evidence_results)

    evidence_payload = evidence_score.to_dict()
    evidence_payload["claim_verification"] = [r.to_dict() for r in evidence_results]
    rave_payload = calculate_response_rave(evidence_results).to_dict()
    rave_payload["sensitivity_analysis"] = run_rave_sensitivity_analysis(
        evidence_results
    )

    # Person 2: response-level stability.
    stability_payload = None
    if stability_responses is not None:
        responses = list(stability_responses)
        if len(responses) < 2:
            raise ValueError("stability_responses must contain at least 2 responses.")
        stability_payload = compute_answer_stability(responses).to_dict()
        stability_payload["stability_percent"] = (
            stability_payload["average_stability"] * 100.0
        )
    else:
        if stability_samples < 2:
            raise ValueError("stability_samples must be at least 2.")
        if stability_generator is None:
            from metrics.ollama_generator import generator_llm
            stability_generator = lambda prompt: generator_llm(
                prompt, model_name=stability_model_name
            )
        print(
            f"Measuring Answer Stability with {stability_model_name} "
            f"({stability_samples} samples)..."
        )
        stability_payload = answer_stability(
            stability_generator, question, n=stability_samples
        ).to_dict()
        stability_payload["stability_percent"] = (
            stability_payload["average_stability"] * 100.0
        )
        stability_payload["model"] = stability_model_name

    # Person 3: citation quality. Lazily imported because its NLI models are
    # heavy and should only load when citation scoring is requested.
    from citation.pipeline import evaluate_retrieved_citations

    citation_payload = evaluate_retrieved_citations(
        evidence_results=evidence_results,
        question=question,
        judge=citation_judge,
    )
    if citation_sources is not None:
        from citation.pipeline import evaluate_citations

        citation_payload = evaluate_citations(
            response_text=answer,
            claims=claims,
            sources=list(citation_sources),
            question=question,
            judge=citation_judge,
        )

    aggregation = aggregate_trust(
        evidence_payload, rave_payload, stability_payload, citation_payload
    ).to_dict()
    decision = decide_response(aggregation).to_dict()
    report = build_trust_report(question, answer, aggregation, decision)

    return TrustGuardResult(
        question=question,
        answer=answer,
        answer_origin=answer_origin,
        claims=[claim.to_dict() for claim in claims],
        evidence_support=evidence_payload,
        source_reliability=rave_payload,
        answer_stability=stability_payload,
        citation_quality=citation_payload,
        trust_aggregation=aggregation,
        decision=decision,
        trust_report=report,
    )
