"""Offline integration tests for the interfaces shared by all three people."""

from data.schema import Claim, EvidencePassage
from citation.citation_extractor import extract_citations, map_claims_to_citations, resolve_citations
from metrics.answer_stability import compute_answer_stability
from metrics.evidence_support_score import calculate_response_evidence_score
from metrics.source_reliability import (
    calculate_claim_rave,
    calculate_response_rave,
    normalize_source_name,
    run_rave_sensitivity_analysis,
)
from data.source_registry import authority_rubric
from retrieval.hybrid import HybridRetriever
from retrieval.index import EvidenceIndex
from tests.mock_corpus import MOCK_PASSAGES
from verification.evidence_support import (
    EvidenceSupportResult,
    PassageJudgment,
    judge_evidence_passages,
)
from retrieval.hybrid import HybridResult
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
            PassageJudgment("e1", "WHO", "A", 0.92, "SUPPORTED", "supports", 0.90, 0.05, 0.05),
            PassageJudgment("e2", "PubMed", "B", 0.88, "SUPPORTED", "supports", 0.80, 0.10, 0.10),
            PassageJudgment("e3", "PubMed", "C", 0.71, "CONTRADICTED", "conflicts", 0.05, 0.95, 0.00),
            PassageJudgment("e4", "CDC", "D", 0.65, "IRRELEVANT", "not enough"),
        ],
    )
    score = calculate_response_evidence_score([result])
    assert score.total_claims == 1
    assert score.mixed_claims == 1
    assert abs(score.evidence_support_score - 24.02) < 0.01


class _DeterministicNLIPredictor:
    def __init__(self, scores):
        self.scores = scores

    def score(self, premise, hypothesis):
        return self.scores[premise]


def test_nli_maps_entailment_contradiction_and_neutral_to_verdicts():
    claim = Claim(text="Warfarin increases bleeding risk.", source_response_id="r1")
    passages = MOCK_PASSAGES[:3]
    evidence = [
        HybridResult(passage, score=score)
        for passage, score in zip(passages, (0.50, 0.30, 0.20))
    ]
    predictor = _DeterministicNLIPredictor({
        passages[0].text: {"entailment": 0.90, "contradiction": 0.05, "neutral": 0.05},
        passages[1].text: {"entailment": 0.05, "contradiction": 0.90, "neutral": 0.05},
        passages[2].text: {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.90},
    })

    judgments = judge_evidence_passages(claim, evidence, nli_predictor=predictor)

    assert [item.verdict for item in judgments] == [
        "SUPPORTED", "CONTRADICTED", "IRRELEVANT"
    ]
    assert [item.retrieval_weight for item in judgments] == [0.5, 0.3, 0.2]


def test_neutral_evidence_does_not_inflate_nli_support_score():
    result = EvidenceSupportResult(
        claim_id="c1",
        claim_text="A test claim.",
        verdict="MIXED",
        passage_judgments=[
            PassageJudgment("support", "S", "", 0.5, "SUPPORTED", "", 0.80, 0.10, 0.10),
            PassageJudgment("contradiction", "S", "", 0.3, "CONTRADICTED", "", 0.05, 0.90, 0.05),
            PassageJudgment("neutral", "S", "", 0.2, "IRRELEVANT", "", 0.49, 0.01, 0.50),
        ],
    )
    score = calculate_response_evidence_score([result])

    # Net evidence uses every NLI probability:
    # entailment = .5*.8 + .3*.05 + .2*.49 = .513
    # contradiction = .5*.1 + .3*.9 + .2*.01 = .322
    assert score.evidence_support_score == 19.1


def test_support_without_contradiction_is_not_automatically_perfect():
    result = EvidenceSupportResult(
        claim_id="c1",
        claim_text="A weakly supported claim.",
        verdict="SUPPORTED",
        passage_judgments=[
            PassageJudgment("e1", "WHO", "", 1.0, "SUPPORTED", "", 0.62, 0.08, 0.30),
        ],
    )

    score = calculate_response_evidence_score([result])

    assert score.evidence_support_score == 54.0


def _rave_result(claim_id, entries):
    """Build deterministic claim evidence for RAVE tests without model inference."""
    passages = {}
    judgments = []
    for index, (source, identifier, verdict, confidence) in enumerate(entries, start=1):
        evidence_id = f"e{index}"
        passages[evidence_id] = EvidencePassage(
            text=f"Evidence {index}",
            source_name=source,
            source_tier="test",
            identifier=identifier,
            passage_id=evidence_id,
        )
        judgments.append(PassageJudgment(
            evidence_id, source, "", 1.0, verdict, "",
            entailment_probability=confidence if verdict == "SUPPORTED" else 0.0,
            contradiction_probability=confidence if verdict == "CONTRADICTED" else 0.0,
            neutral_probability=confidence if verdict == "IRRELEVANT" else 0.0,
        ))
    return EvidenceSupportResult(
        claim_id=claim_id,
        claim_text=f"Claim {claim_id}",
        verdict="MIXED",
        passage_judgments=judgments,
        evidence_passages=passages,
    )


def test_rave_marks_single_source_as_uncorroborated():
    report = calculate_claim_rave(_rave_result("c1", [
        ("WHO", "who-page", "SUPPORTED", 0.9),
    ]))

    assert report.rave_score is None
    assert report.source_results[0].status == "UNCORROBORATED"
    assert report.source_results[0].n_independent_sources == 0
    assert report.source_results[0].baseline_authority == 0.95


def test_rave_deduplicates_chunks_and_treats_pubmed_pmids_independently():
    report = calculate_claim_rave(_rave_result("c1", [
        ("PubMed", "100:chunk:0", "SUPPORTED", 0.8),
        ("PubMed", "100:chunk:1", "SUPPORTED", 0.9),
        ("PubMed", "200", "SUPPORTED", 0.7),
    ]))

    assert len(report.source_results) == 2
    assert {item.source_id for item in report.source_results} == {"pubmed:100", "pubmed:200"}
    assert all(item.n_independent_sources == 1 for item in report.source_results)
    assert all(item.status == "CORROBORATED" for item in report.source_results)


def test_rave_excludes_internally_conflicted_sources_and_reports_coverage():
    conflicted = _rave_result("c1", [
        ("WHO", "who-a", "SUPPORTED", 0.9),
        ("WHO", "who-b", "CONTRADICTED", 0.8),
        ("PubMed", "101", "SUPPORTED", 0.7),
    ])
    neutral = _rave_result("c2", [
        ("CDC", "cdc-a", "IRRELEVANT", 0.9),
    ])
    corroborated = _rave_result("c3", [
        ("WHO", "who-c", "SUPPORTED", 0.9),
        ("PubMed", "102", "SUPPORTED", 0.8),
    ])

    claim = calculate_claim_rave(conflicted)
    response = calculate_response_rave([corroborated, neutral])

    who = next(item for item in claim.source_results if item.source_key == "who")
    pubmed = next(item for item in claim.source_results if item.source_key == "pubmed")
    assert who.status == "INTERNALLY_CONFLICTED"
    assert who.rave_score is None
    assert pubmed.status == "UNCORROBORATED"
    assert pubmed.rave_score is None
    assert claim.rave_score is None
    assert response.claims_with_rave == 1
    assert response.coverage == 0.5


def test_rave_marks_opposing_independent_sources_as_disagreed():
    report = calculate_claim_rave(_rave_result("c1", [
        ("NIH / MedlinePlus", "topic-a", "SUPPORTED", 0.9),
        ("PubMed", "303", "CONTRADICTED", 0.8),
    ]))

    assert normalize_source_name("NIH / MedlinePlus") == "medlineplus"
    assert {item.status for item in report.source_results} == {"DISAGREED"}
    assert all(item.corroboration == 0.0 for item in report.source_results)


def test_source_authority_rubric_and_sensitivity_analysis_are_explicit():
    who = authority_rubric("who")
    assert who["baseline_authority"] == 0.95
    assert who["weighted_score_out_of_5"] == 5.0

    evidence = _rave_result("c1", [
        ("WHO", "who-a", "SUPPORTED", 0.9),
        ("PubMed", "404", "SUPPORTED", 0.8),
    ])
    analysis = run_rave_sensitivity_analysis([evidence])

    assert set(analysis["scenarios"]) == {
        "conservative", "current", "literature_favouring"
    }
    assert analysis["rave_score_range"]["minimum"] <= analysis["rave_score_range"]["maximum"]


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
