"""Offline integration tests for the interfaces shared by all three people."""

from data.schema import Claim
from citation.citation_extractor import extract_citations, map_claims_to_citations, resolve_citations
from metrics.answer_stability import compute_answer_stability
from metrics.evidence_support_score import calculate_response_evidence_score
from retrieval.hybrid import HybridRetriever
from retrieval.index import EvidenceIndex
from tests.mock_corpus import MOCK_PASSAGES
from verification.evidence_support import EvidenceSupportResult, PassageJudgment
from citation.minilm_judge import CitationJudgement
from citation.pipeline import evaluate_retrieved_citations


def test_hybrid_fallback_works_without_dense_index():
    retriever = HybridRetriever(EvidenceIndex(MOCK_PASSAGES), dense_index=None)
    results = retriever.query("warfarin NSAID bleeding risk", top_k=3)
    assert results
    assert results[0].passage.source_name == "WHO"


def test_stability_interface():
    result = compute_answer_stability([
        "Warfarin can increase bleeding risk.",
        "Warfarin may increase the risk of bleeding.",
        "Warfarin is associated with bleeding risk.",
    ])
    assert 0.0 <= result.average_stability <= 1.0
    assert len(result.pairwise) == 3


def test_evidence_support_score_interface():
    result = EvidenceSupportResult(
        claim_id="c1",
        claim_text="Warfarin and NSAIDs increase bleeding risk.",
        verdict="MIXED",
        passage_judgments=[
            PassageJudgment("e1", "WHO", "A", 0.92, "SUPPORTED", "supports"),
            PassageJudgment("e2", "PubMed", "B", 0.88, "SUPPORTED", "supports"),
            PassageJudgment("e3", "PubMed", "C", 0.71, "CONTRADICTED", "conflicts"),
            PassageJudgment("e4", "CDC", "D", 0.65, "IRRELEVANT", "not enough"),
        ],
    )
    score = calculate_response_evidence_score([result])
    assert score.total_claims == 1
    assert score.mixed_claims == 1
    assert abs(score.evidence_support_score - 71.71) < 0.01


def test_citation_extraction_resolution_mapping_interface():
    answer = "Warfarin with NSAIDs can increase bleeding risk [1]."
    claim = Claim(
        text="Warfarin with NSAIDs can increase bleeding risk.",
        source_response_id="r1",
    )
    citations = resolve_citations(extract_citations(answer), [MOCK_PASSAGES[2]])
    pairs = map_claims_to_citations(answer, [claim], citations, question="What about warfarin?")
    assert len(pairs) == 1
    assert pairs[0].source_name == "WHO"


class _DeterministicCitationJudge:
    """Keeps the integration test offline and independent of ML downloads."""

    def judge_claims(self, pairs):
        judgements = [
            CitationJudgement(
                claim_id=pair["claim_id"],
                claim_text=pair["claim"],
                citation_id=pair["citation_id"],
                source_name=pair["source_name"],
                source_text=pair["source_text"],
                label="DIRECT",
                score=0.9,
                claim_source_similarity=0.9,
                entailment_probability=0.9,
                contradiction_probability=0.0,
                neutral_probability=0.1,
                rationale="Deterministic test judgement.",
            )
            for pair in pairs
        ]
        return {
            "judgements": judgements,
            "citation_quality_score": 0.9 if judgements else 0.0,
        }


def test_retrieved_evidence_is_automatically_scored_as_citations():
    passage = MOCK_PASSAGES[2]
    evidence = EvidenceSupportResult(
        claim_id="c1",
        claim_text="Warfarin with NSAIDs can increase bleeding risk.",
        verdict="SUPPORTED",
        passage_judgments=[
            PassageJudgment(
                passage.passage_id, "WHO", passage.title, 0.92,
                "SUPPORTED", "supports",
            )
        ],
        evidence_passages={passage.passage_id: passage},
    )

    report = evaluate_retrieved_citations(
        [evidence],
        question="What happens when warfarin is taken with NSAIDs?",
        judge=_DeterministicCitationJudge(),
    )

    assert report["mode"] == "retrieved_evidence"
    assert report["citation_count"] == 1
    assert report["judgements"][0]["evidence_id"] == passage.passage_id
    assert report["judgements"][0]["evidence_support_verdict"] == "SUPPORTED"
