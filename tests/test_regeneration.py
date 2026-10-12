"""Offline tests for Person 3. Run: pytest -q tests/test_regeneration.py"""

from regeneration.feedback_builder import build_regeneration_prompt, extract_feedback
from regeneration.regeneration_controller import run_with_regeneration


def fake_result(answer, verdicts, score, decision):
    claims = []
    for i, v in enumerate(verdicts):
        pid = f"p{i}"
        claims.append({
            "claim_id": f"c{i}", "claim_text": f"claim {i}", "verdict": v,
            "passage_judgments": [{
                "evidence_id": pid, "source_name": "WHO", "retrieval_weight": 1.0,
                "verdict": "CONTRADICTED" if v in ("CONTRADICTED", "MIXED") else "IRRELEVANT",
                "contradiction_probability": 0.9,
            }],
            "evidence_passages": {pid: {"text": f"evidence text {i}", "source_name": "WHO"}},
        })
    return {
        "answer": answer,
        "decision": {"decision": decision, "regeneration_required": decision != "ACCEPT",
                     "safety_gate_triggered": False},
        "trust_aggregation": {"composite_score": score},
        "evidence_support": {"claim_verification": claims},
    }


def test_feedback_split_and_prompt():
    fb = extract_feedback(fake_result(
        "a", ["SUPPORTED", "CONTRADICTED", "IRRELEVANT", "MIXED"], 30, "REGENERATE"))
    assert fb.summary() == {"contradicted": 2, "unverifiable": 1, "supported": 1}
    prompt = build_regeneration_prompt("q", "a", fb)
    assert "keep these" in prompt and "evidence text 1" in prompt
    assert fb.needs_fix


def test_accept_after_one_regeneration():
    seq = iter([fake_result("bad", ["SUPPORTED", "CONTRADICTED"], 30, "REGENERATE"),
                fake_result("good", ["SUPPORTED", "SUPPORTED"], 90, "ACCEPT")])
    out = run_with_regeneration("q", "bad", trust_runner=lambda q, a, **k: next(seq),
                                regenerator=lambda p: "good")
    assert out.status == "accepted_after_regeneration"
    assert not out.needs_human_review and len(out.attempts) == 2


def test_already_accepted_makes_no_llm_call():
    def boom(p):
        raise AssertionError("should not regenerate")
    out = run_with_regeneration(
        "q", "ok", trust_runner=lambda q, a, **k: fake_result("ok", ["SUPPORTED"], 95, "ACCEPT"),
        regenerator=boom)
    assert out.status == "accepted"


def test_max_two_attempts_then_human_review():
    seq = iter([fake_result(f"a{i}", ["CONTRADICTED"], 20 + i * 10, "REGENERATE")
                for i in range(5)])
    texts = iter(["a1", "a2", "a3", "a4"])
    out = run_with_regeneration("q", "a0", trust_runner=lambda q, a, **k: next(seq),
                                regenerator=lambda p: next(texts))
    assert out.needs_human_review and len(out.attempts) == 3  # original + 2 regens


def test_no_improvement_stops_early():
    seq = iter([fake_result("a0", ["CONTRADICTED"], 30, "REGENERATE"),
                fake_result("a1", ["CONTRADICTED"], 30, "REGENERATE"),
                fake_result("a2", ["CONTRADICTED"], 40, "REGENERATE")])
    out = run_with_regeneration("q", "a0", trust_runner=lambda q, a, **k: next(seq),
                                regenerator=lambda p: "a1")
    assert out.needs_human_review and "stopped improving" in out.stop_reason
    assert len(out.attempts) == 2


def test_generator_failure_falls_back_to_human():
    def fail(p):
        raise RuntimeError("ollama down")
    out = run_with_regeneration(
        "q", "x", trust_runner=lambda q, a, **k: fake_result("x", ["CONTRADICTED"], 20, "REGENERATE"),
        regenerator=fail)
    assert out.needs_human_review and "regeneration failed" in out.stop_reason


def test_identical_revision_stops():
    out = run_with_regeneration(
        "q", "x", trust_runner=lambda q, a, **k: fake_result("x", ["CONTRADICTED"], 20, "REGENERATE"),
        regenerator=lambda p: "x")
    assert out.needs_human_review and "identical" in out.stop_reason
