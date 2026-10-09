"""NLI-based evidence-support verification for TrustGuard.

Each evidence passage is compared with an extracted claim using the local
DeBERTa NLI model.  No generative LLM is used for classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from data.schema import Claim, EvidencePassage


NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MODEL_NAME = NLI_MODEL_NAME  # Compatibility alias for existing callers.


@dataclass
class PassageJudgment:
    """NLI judgement for one retrieved passage and one claim."""

    evidence_id: str
    source_name: str
    title: str
    retrieval_score: float
    verdict: str
    explanation: str
    entailment_probability: float = 0.0
    contradiction_probability: float = 0.0
    neutral_probability: float = 0.0
    retrieval_weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_name": self.source_name,
            "title": self.title,
            "retrieval_score": self.retrieval_score,
            "retrieval_weight": self.retrieval_weight,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "entailment_probability": self.entailment_probability,
            "contradiction_probability": self.contradiction_probability,
            "neutral_probability": self.neutral_probability,
        }


@dataclass
class EvidenceSupportResult:
    """Passage-level and aggregate evidence assessment for one claim."""

    claim_id: str
    claim_text: str
    verdict: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    irrelevant_evidence_ids: list[str] = field(default_factory=list)
    passage_judgments: list[PassageJudgment] = field(default_factory=list)
    evidence_passages: dict[str, EvidencePassage] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "contradicting_evidence_ids": self.contradicting_evidence_ids,
            "irrelevant_evidence_ids": self.irrelevant_evidence_ids,
            "passage_judgments": [item.to_dict() for item in self.passage_judgments],
            "evidence_passages": {
                evidence_id: passage.to_dict()
                for evidence_id, passage in self.evidence_passages.items()
            },
            "explanation": self.explanation,
        }


class NLIPredictor(Protocol):
    """Small interface used to keep NLI verification easy to unit test."""

    def score(self, premise: str, hypothesis: str) -> dict[str, float]: ...


class DebertaNLIPredictor:
    """Run the already-used DeBERTa NLI model from the local HF cache."""

    def __init__(self, model_name: str = NLI_MODEL_NAME) -> None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as error:
            raise ImportError(
                "Evidence Support NLI requires transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError as error:
            raise RuntimeError(
                f"The local NLI model {model_name!r} is unavailable. "
                "Download it once before running Evidence Support."
            ) from error
        self._classifier = pipeline(
            "text-classification", model=model, tokenizer=tokenizer
        )
        self._id2label = getattr(model.config, "id2label", {})

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        result = self._classifier(
            premise, text_pair=hypothesis, truncation=True, top_k=None
        )
        if isinstance(result, dict):
            result = [result]
        elif result and isinstance(result[0], list):
            result = result[0]
        probabilities = {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        }
        for item in result:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).lower()
            if label.startswith("label_"):
                try:
                    label = str(self._id2label[int(label.rsplit("_", 1)[1])]).lower()
                except (KeyError, TypeError, ValueError):
                    continue
            if "entail" in label:
                probabilities["entailment"] = float(item.get("score", 0.0))
            elif "contrad" in label:
                probabilities["contradiction"] = float(item.get("score", 0.0))
            elif "neutral" in label:
                probabilities["neutral"] = float(item.get("score", 0.0))
        return probabilities


def retrieve_evidence(claim: Claim, retriever, top_k: int = 5):
    return retriever.query(claim.text, top_k=top_k)


def _retrieval_relevance(result) -> float:
    """Use hybrid retrieval's final fused score for NLI weighting."""
    return max(0.0, float(getattr(result, "score", 0.0)))


def _normalised_weights(evidence_results) -> list[float]:
    relevance = [_retrieval_relevance(item) for item in evidence_results]
    total = sum(relevance)
    if total > 0.0:
        return [value / total for value in relevance]
    return [1.0 / len(evidence_results)] * len(evidence_results)


def _label(probabilities: dict[str, float]) -> str:
    label, _ = max(
        ("SUPPORTED", probabilities["entailment"]),
        ("CONTRADICTED", probabilities["contradiction"]),
        ("IRRELEVANT", probabilities["neutral"]),
        key=lambda item: item[1],
    )
    return label


def judge_evidence_passages(
    claim: Claim,
    evidence_results,
    model_name: str = NLI_MODEL_NAME,
    nli_predictor: NLIPredictor | None = None,
) -> list[PassageJudgment]:
    """Classify each retrieved passage with local DeBERTa NLI."""
    if not evidence_results:
        return []
    predictor = nli_predictor or DebertaNLIPredictor(model_name)
    judgments: list[PassageJudgment] = []
    for result, weight in zip(evidence_results, _normalised_weights(evidence_results)):
        passage = result.passage
        probabilities = predictor.score(passage.text, claim.text)
        entailment = max(0.0, float(probabilities.get("entailment", 0.0)))
        contradiction = max(0.0, float(probabilities.get("contradiction", 0.0)))
        neutral = max(0.0, float(probabilities.get("neutral", 0.0)))
        verdict = _label({
            "entailment": entailment,
            "contradiction": contradiction,
            "neutral": neutral,
        })
        judgments.append(PassageJudgment(
            evidence_id=passage.passage_id,
            source_name=passage.source_name,
            title=passage.title,
            retrieval_score=_retrieval_relevance(result),
            retrieval_weight=weight,
            verdict=verdict,
            explanation=(
                "DeBERTa NLI probabilities: "
                f"entailment={entailment:.3f}, contradiction={contradiction:.3f}, "
                f"neutral={neutral:.3f}."
            ),
            entailment_probability=entailment,
            contradiction_probability=contradiction,
            neutral_probability=neutral,
        ))
    return judgments


def aggregate_passage_judgments(
    claim: Claim, judgments: list[PassageJudgment]
) -> EvidenceSupportResult:
    """Keep conflicting evidence visible instead of resolving it by voting."""
    supporting = [item.evidence_id for item in judgments if item.verdict == "SUPPORTED"]
    contradicting = [item.evidence_id for item in judgments if item.verdict == "CONTRADICTED"]
    irrelevant = [item.evidence_id for item in judgments if item.verdict == "IRRELEVANT"]
    if supporting and contradicting:
        verdict, explanation = "MIXED", "Retrieved evidence contains support and contradiction."
    elif supporting:
        verdict, explanation = "SUPPORTED", "At least one passage entails the claim."
    elif contradicting:
        verdict, explanation = "CONTRADICTED", "At least one passage contradicts the claim."
    else:
        verdict, explanation = "IRRELEVANT", "All retrieved passages are neutral to the claim."
    return EvidenceSupportResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradicting,
        irrelevant_evidence_ids=irrelevant,
        passage_judgments=judgments,
        explanation=explanation,
    )


def judge_claim(
    claim: Claim,
    evidence_results,
    model_name: str = NLI_MODEL_NAME,
    nli_predictor: NLIPredictor | None = None,
) -> EvidenceSupportResult:
    if not evidence_results:
        return EvidenceSupportResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            verdict="IRRELEVANT",
            explanation="No evidence passages were retrieved for this claim.",
        )
    result = aggregate_passage_judgments(
        claim,
        judge_evidence_passages(
            claim, evidence_results, model_name=model_name, nli_predictor=nli_predictor
        ),
    )
    result.evidence_passages = {
        item.passage.passage_id: item.passage for item in evidence_results
    }
    return result


def verify_claims(
    claims: list[Claim],
    retriever,
    top_k: int = 5,
    nli_predictor: NLIPredictor | None = None,
) -> list[EvidenceSupportResult]:
    """Retrieve and verify every claim, loading DeBERTa only once."""
    predictor = nli_predictor
    results: list[EvidenceSupportResult] = []
    for index, claim in enumerate(claims, start=1):
        print(f"\n{'=' * 70}")
        print(f"Verifying claim {index}/{len(claims)}")
        print(f"Claim: {claim.text}")
        evidence = retrieve_evidence(claim, retriever, top_k=top_k)
        print(f"Retrieved {len(evidence)} evidence passages.")
        if predictor is None and evidence:
            predictor = DebertaNLIPredictor()
        results.append(judge_claim(claim, evidence, nli_predictor=predictor))
    return results
