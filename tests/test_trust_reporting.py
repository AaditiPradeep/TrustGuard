"""Tests for aggregation, decision rules, and the layperson report."""

from decision_engine import decide_response
from trust_aggregator import aggregate_trust
from trust_report import build_trust_report


def _aggregation(status="SUPPORTED", claim_text="A treatment may help."):
    return aggregate_trust(
        {"evidence_support_score": 80, "claim_scores": [{
            "claim_id": "c1", "claim_text": claim_text, "score": 80,
            "status": status,
        }]},
        {"rave_score": 0.9, "claim_results": [{"claim_id": "c1", "rave_score": 0.9}]},
        {"average_stability": 0.8},
        {"available": True, "citation_quality_score": 0.7,
         "judgements": [{"claim_id": "c1", "score": 0.7}]},
    )


def test_aggregation_only_combines_existing_metric_outputs():
    report = _aggregation()

    # .4*.8 + .25*.9 + .2*.8 + .15*.7 = .81; no metric formula is recalculated.
    assert report.composite_score == 81.0
    assert report.category == "HIGH"
    assert report.claim_contributions[0].evidence_support_score == 80


def test_missing_metric_is_reported_and_available_weights_are_redistributed():
    report = aggregate_trust(
        {"evidence_support_score": 80, "claim_scores": []},
        {"rave_score": None, "claim_results": []},
        {"average_stability": 0.6},
        {"available": False, "citation_quality_score": None, "judgements": []},
    )

    assert report.missing_metrics == ["source_reliability", "citation_quality"]
    assert report.composite_score == 73.33


def test_contradicted_high_risk_claim_requires_regeneration():
    decision = decide_response(_aggregation("CONTRADICTED", "This dose prevents bleeding.").to_dict())

    assert decision.decision == "REGENERATE"
    assert decision.safety_gate_triggered is True


def test_plain_language_report_lists_each_claims_metric_scores():
    aggregation = _aggregation().to_dict()
    decision = decide_response(aggregation).to_dict()
    report = build_trust_report("Question?", "Answer.", aggregation, decision)

    assert report["recommendation"] == "ACCEPT"
    assert report["claims"][0]["metric_scores"]["evidence_support"] == 80
    assert "overall confidence" in report["plain_language_summary"]
