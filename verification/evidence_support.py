"""
TrustGuard - Evidence Support Verification

For every extracted medical claim:

1. Retrieve the most relevant evidence passages.
2. Judge EACH retrieved passage independently as:
      SUPPORTED
      CONTRADICTED
      IRRELEVANT
3. Aggregate the passage-level judgments into a claim-level result.

Claim-level result:
    SUPPORTED     -> at least one supporting passage and no contradiction
    CONTRADICTED  -> at least one contradicting passage and no support
    IRRELEVANT    -> no supporting or contradicting passage
    MIXED         -> both supporting and contradicting passages exist

MIXED preserves conflicting evidence instead of allowing the LLM
to silently choose one side.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from data.schema import Claim, EvidencePassage
from local_llm.ollama import DEFAULT_MODEL, chat
from retrieval.dense_index import DenseEvidenceIndex
from retrieval.hybrid import HybridRetriever
from verification.claim_extractor import extract_claims


MODEL_NAME = DEFAULT_MODEL


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PassageJudgment:
    """
    Judgment for one evidence passage against one claim.
    """

    evidence_id: str
    source_name: str
    title: str
    retrieval_score: float
    verdict: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_name": self.source_name,
            "title": self.title,
            "retrieval_score": self.retrieval_score,
            "verdict": self.verdict,
            "explanation": self.explanation,
        }


@dataclass
class EvidenceSupportResult:
    """
    Final evidence-support result for one claim.
    """

    claim_id: str
    claim_text: str

    # Claim-level aggregated state
    verdict: str

    supporting_evidence_ids: list[str] = field(
        default_factory=list
    )

    contradicting_evidence_ids: list[str] = field(
        default_factory=list
    )

    irrelevant_evidence_ids: list[str] = field(
        default_factory=list
    )

    passage_judgments: list[PassageJudgment] = field(
        default_factory=list
    )

    # The exact records retrieved for this claim.  Keeping these records with
    # the verification result lets downstream citation-quality code score the
    # evidence that was actually retrieved, rather than requiring a second,
    # manually maintained source list.
    evidence_passages: dict[str, EvidencePassage] = field(default_factory=dict)

    explanation: str = ""

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "verdict": self.verdict,
            "supporting_evidence_ids":
                self.supporting_evidence_ids,
            "contradicting_evidence_ids":
                self.contradicting_evidence_ids,
            "irrelevant_evidence_ids":
                self.irrelevant_evidence_ids,
            "passage_judgments": [
                judgment.to_dict()
                for judgment in self.passage_judgments
            ],
            "evidence_passages": {
                evidence_id: passage.to_dict()
                for evidence_id, passage in self.evidence_passages.items()
            },
            "explanation": self.explanation,
        }


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
/no_think

You are an evidence-based medical claim verification system.

You will receive:

1. ONE medical claim.
2. Several retrieved evidence passages.

Your task is to judge EACH evidence passage independently.

For every evidence passage, classify its relationship to the
claim as exactly ONE of:

SUPPORTED
CONTRADICTED
IRRELEVANT


IMPORTANT:

Use ONLY the supplied evidence passage when judging that passage.

Do NOT use your own medical knowledge.

Do NOT use another evidence passage to decide the verdict for the
current passage.


SUPPORTED

Choose SUPPORTED only when the passage directly supports or
clearly entails the important factual content of the claim.


CONTRADICTED

Choose CONTRADICTED only when the passage directly conflicts
with the claim or an important part of the claim.


IRRELEVANT

Choose IRRELEVANT when the passage does not contain enough
information to support or contradict the claim.

This includes:

- same topic but different relationship
- missing important qualifier
- evidence that is too general
- evidence that discusses the drug/disease but not the assertion
- missing information


QUALIFIER RULE

Pay special attention to qualifiers such as:

- may
- can
- always
- never
- common
- commonly
- rare
- rarely
- usually
- sometimes
- mild
- severe
- serious
- dosage
- frequency
- duration
- patient population
- disease stage


Example:

CLAIM:
"Drug X rarely causes condition Y."

EVIDENCE:
"Drug X can cause condition Y."

VERDICT:
IRRELEVANT

Reason:
The evidence establishes occurrence but does not establish rarity.


Example:

CLAIM:
"Drug X never causes condition Y."

EVIDENCE:
"Drug X may cause condition Y."

VERDICT:
CONTRADICTED


ABSENCE IS NOT CONTRADICTION.

Example:

CLAIM:
"Metformin may cause nausea."

EVIDENCE:
"Metformin can cause lactic acidosis."

VERDICT:
IRRELEVANT


OUTPUT FORMAT

Return ONLY valid JSON.

Use exactly this structure:

{
  "judgments": [
    {
      "evidence_id": "id1",
      "verdict": "SUPPORTED",
      "explanation": "Short explanation."
    },
    {
      "evidence_id": "id2",
      "verdict": "CONTRADICTED",
      "explanation": "Short explanation."
    },
    {
      "evidence_id": "id3",
      "verdict": "IRRELEVANT",
      "explanation": "Short explanation."
    }
  ]
}

Return one judgment for EVERY supplied evidence passage.

Do not invent evidence IDs.

Do not omit evidence passages.

Do not return Markdown.

Do not return text outside the JSON object.
""".strip()


# ============================================================
# ============================================================
# JSON PARSING
# ============================================================

def _extract_json(text: str) -> dict[str, Any]:

    if not text:
        raise ValueError(
            "Cannot parse JSON from an empty model response."
        )

    text = text.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Could not find JSON in model response:\n\n"
            f"{text}"
        )

    try:
        return json.loads(
            match.group(0)
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            "Could not parse JSON from model response:\n\n"
            f"{text}"
        ) from error


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_evidence(
    claim: Claim,
    retriever,
    top_k: int = 5,
):
    """
    Retrieve top-k evidence chunks for a claim.
    """

    return retriever.query(
        claim.text,
        top_k=top_k,
    )


def _retrieval_relevance(result) -> float:
    """Return a 0-1-ish relevance value for Evidence Support weighting.

    Hybrid retrieval uses an RRF fusion score for ranking. That score is not
    a semantic similarity, so when a HybridResult is supplied we prefer its
    dense cosine similarity; if dense is unavailable we fall back to TF-IDF
    cosine similarity, and finally to the generic ``score`` attribute.
    """

    dense_score = getattr(result, "dense_score", None)
    if dense_score is not None:
        return max(0.0, float(dense_score))

    tfidf_score = getattr(result, "tfidf_score", None)
    if tfidf_score is not None:
        return max(0.0, float(tfidf_score))

    return max(0.0, float(getattr(result, "score", 0.0)))


def _format_evidence(results) -> str:
    """Format retrieved evidence for Qwen."""

    blocks = []

    for i, result in enumerate(results, start=1):
        passage = result.passage
        relevance = _retrieval_relevance(result)

        extra_scores = []
        dense_score = getattr(result, "dense_score", None)
        tfidf_score = getattr(result, "tfidf_score", None)
        fused_score = getattr(result, "score", None)

        if dense_score is not None:
            extra_scores.append(f"Dense similarity: {float(dense_score):.4f}")
        if tfidf_score is not None:
            extra_scores.append(f"TF-IDF similarity: {float(tfidf_score):.4f}")
        if dense_score is not None or tfidf_score is not None:
            extra_scores.append(f"Hybrid fusion score: {float(fused_score):.4f}")

        score_details = "\n".join(extra_scores)
        if score_details:
            score_details = "\n" + score_details

        block = f"""
EVIDENCE {i}

Evidence ID: {passage.passage_id}

Source: {passage.source_name}

Title: {passage.title}

Retrieval Relevance: {relevance:.4f}{score_details}

Text:
{passage.text}
""".strip()

        blocks.append(block)

    return "\n\n".join(blocks)


# ============================================================
# MODEL CALL
# ============================================================

def _call_model(
    model_name: str,
    messages: list[dict],
) -> str | None:
    return chat(messages, model=model_name, max_tokens=1000, temperature=0.1)


# ============================================================
# PASSAGE-LEVEL VERIFICATION
# ============================================================

def judge_evidence_passages(
    claim: Claim,
    evidence_results,
    model_name: str = MODEL_NAME,
) -> list[PassageJudgment]:
    """
    Classify every retrieved passage independently.
    """

    if not evidence_results:
        return []

    evidence_text = _format_evidence(
        evidence_results
    )

    prompt = f"""
/no_think

CLAIM:

{claim.text}


RETRIEVED EVIDENCE:

{evidence_text}


Judge EACH evidence passage independently.

For each passage choose exactly one:

SUPPORTED
CONTRADICTED
IRRELEVANT


Remember:

- semantic similarity alone does not mean support
- missing information is IRRELEVANT, not contradiction
- qualifiers must be established
- do not use information from one evidence passage to judge another
- return one judgment for every supplied evidence ID


Return ONLY JSON:

{{
  "judgments": [
    {{
      "evidence_id": "evidence_id",
      "verdict": "SUPPORTED",
      "explanation": "Brief explanation."
    }}
  ]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    content = _call_model(
        model_name=model_name,
        messages=messages,
    )

    # Retry once
    if not content:

        print(
            "Qwen returned empty evidence judgment. "
            "Retrying..."
        )

        retry_messages = [
            {
                "role": "system",
                "content": """
/no_think

Judge every evidence passage independently.

Use only the supplied evidence.

Return only valid JSON.

Do not omit any evidence passage.
""".strip(),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        content = _call_model(
            model_name=model_name,
            messages=retry_messages,
        )

    if not content:
        raise RuntimeError(
            "Qwen returned empty evidence judgment "
            "after two attempts."
        )

    parsed = _extract_json(
        content
    )

    raw_judgments = parsed.get(
        "judgments",
        [],
    )

    if not isinstance(
        raw_judgments,
        list,
    ):
        raise ValueError(
            "Qwen did not return a valid judgments list."
        )

    # Map retrieved IDs to their actual objects.
    result_map = {
        result.passage.passage_id: result
        for result in evidence_results
    }

    allowed_verdicts = {
        "SUPPORTED",
        "CONTRADICTED",
        "IRRELEVANT",
    }

    # Store model responses by evidence ID.
    model_judgments = {}

    for item in raw_judgments:

        if not isinstance(
            item,
            dict,
        ):
            continue

        evidence_id = str(
            item.get(
                "evidence_id",
                "",
            )
        ).strip()

        # Prevent invented IDs.
        if evidence_id not in result_map:
            continue

        verdict = str(
            item.get(
                "verdict",
                "",
            )
        ).strip().upper()

        if verdict not in allowed_verdicts:
            continue

        explanation = str(
            item.get(
                "explanation",
                "",
            )
        ).strip()

        model_judgments[evidence_id] = {
            "verdict": verdict,
            "explanation": explanation,
        }

    # --------------------------------------------------------
    # Build final list.
    #
    # If Qwen accidentally omits a retrieved passage,
    # classify it conservatively as IRRELEVANT.
    # --------------------------------------------------------

    judgments = []

    for result in evidence_results:

        passage = result.passage

        model_result = model_judgments.get(
            passage.passage_id
        )

        if model_result is None:

            verdict = "IRRELEVANT"

            explanation = (
                "No valid judgment was returned for this "
                "evidence passage, so it was conservatively "
                "treated as irrelevant."
            )

        else:

            verdict = model_result[
                "verdict"
            ]

            explanation = model_result[
                "explanation"
            ]

            if not explanation:

                explanation = (
                    f"The passage was classified as "
                    f"{verdict} relative to the claim."
                )

        judgments.append(
            PassageJudgment(
                evidence_id=passage.passage_id,
                source_name=passage.source_name,
                title=passage.title,
                retrieval_score=_retrieval_relevance(result),
                verdict=verdict,
                explanation=explanation,
            )
        )

    return judgments


# ============================================================
# CLAIM-LEVEL AGGREGATION
# ============================================================

def aggregate_passage_judgments(
    claim: Claim,
    judgments: list[PassageJudgment],
) -> EvidenceSupportResult:
    """
    Aggregate passage-level judgments.

    IMPORTANT:
    We deliberately do NOT use majority voting.

    If both support and contradiction exist,
    the result becomes MIXED.

    Source Reliability / RAVE can later help resolve
    which evidence should carry greater weight.
    """

    supporting_ids = [
        judgment.evidence_id
        for judgment in judgments
        if judgment.verdict == "SUPPORTED"
    ]

    contradicting_ids = [
        judgment.evidence_id
        for judgment in judgments
        if judgment.verdict == "CONTRADICTED"
    ]

    irrelevant_ids = [
        judgment.evidence_id
        for judgment in judgments
        if judgment.verdict == "IRRELEVANT"
    ]

    # --------------------------------------------------------
    # Deterministic aggregation
    # --------------------------------------------------------

    if (
        supporting_ids
        and contradicting_ids
    ):

        verdict = "MIXED"

        explanation = (
            "The retrieved evidence contains both supporting "
            "and contradicting passages. The disagreement is "
            "preserved rather than resolved by majority vote."
        )

    elif supporting_ids:

        verdict = "SUPPORTED"

        explanation = (
            "At least one retrieved passage supports the claim "
            "and no retrieved passage directly contradicts it."
        )

    elif contradicting_ids:

        verdict = "CONTRADICTED"

        explanation = (
            "At least one retrieved passage contradicts the "
            "claim and no retrieved passage directly supports it."
        )

    else:

        verdict = "IRRELEVANT"

        explanation = (
            "None of the retrieved passages directly support "
            "or contradict the claim."
        )

    return EvidenceSupportResult(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        verdict=verdict,
        supporting_evidence_ids=supporting_ids,
        contradicting_evidence_ids=contradicting_ids,
        irrelevant_evidence_ids=irrelevant_ids,
        passage_judgments=judgments,
        explanation=explanation,
    )


# ============================================================
# VERIFY ONE CLAIM
# ============================================================

def judge_claim(
    claim: Claim,
    evidence_results,
    model_name: str = MODEL_NAME,
) -> EvidenceSupportResult:
    """
    Judge all evidence passages and aggregate the result.
    """

    if not evidence_results:

        return EvidenceSupportResult(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            verdict="IRRELEVANT",
            supporting_evidence_ids=[],
            contradicting_evidence_ids=[],
            irrelevant_evidence_ids=[],
            passage_judgments=[],
            explanation=(
                "No evidence passages were retrieved "
                "for this claim."
            ),
        )

    judgments = judge_evidence_passages(
        claim=claim,
        evidence_results=evidence_results,
        model_name=model_name,
    )

    result = aggregate_passage_judgments(
        claim=claim,
        judgments=judgments,
    )
    result.evidence_passages = {
        evidence_result.passage.passage_id: evidence_result.passage
        for evidence_result in evidence_results
    }
    return result


# ============================================================
# VERIFY ALL CLAIMS
# ============================================================

def verify_claims(
    claims: list[Claim],
    retriever,
    top_k: int = 5,
) -> list[EvidenceSupportResult]:

    results = []

    for i, claim in enumerate(
        claims,
        start=1,
    ):

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Verifying claim "
            f"{i}/{len(claims)}"
        )

        print(
            f"Claim: {claim.text}"
        )

        evidence = retrieve_evidence(
            claim=claim,
            retriever=retriever,
            top_k=top_k,
        )

        print(
            f"Retrieved "
            f"{len(evidence)} evidence passages."
        )

        result = judge_claim(
            claim=claim,
            evidence_results=evidence,
        )

        results.append(
            result
        )

    return results

if __name__ == "__main__":

    test_question = (
    "What should a patient know about insulin and diabetes?"
)

    test_answer = """
    Insulin helps regulate blood glucose levels.
    Insulin is used in the treatment of diabetes.
    Insulin can cause hypoglycemia.
    Insulin can never cause low blood sugar.
    All patients with type 2 diabetes must always use insulin.
    Insulin commonly causes hearing loss.
    """

    print("\nExtracting claims...\n")

    extracted_claims = extract_claims(
        question=test_question,
        answer=test_answer,
        response_id="test_response_003",
    )

    print(
        f"\nExtracted {len(extracted_claims)} claims:\n"
    )

    for i, claim in enumerate(
        extracted_claims,
        start=1,
    ):
        print(f"Claim {i}")
        print(f"ID   : {claim.claim_id}")
        print(f"Text : {claim.text}")
        print()

    print("\nLoading hybrid evidence retriever...")

    retriever = HybridRetriever.from_jsonl(
        "cache/evidence_corpus.jsonl",
        dense_index_dir="cache/dense_evidence_index",
    )

    print("\nVerifying claims...")

    results = verify_claims(
        claims=extracted_claims,
        retriever=retriever,
        top_k=5,
    )

    from metrics.evidence_support_score import (
        calculate_response_evidence_score,
        print_evidence_support_score,
    )

    print("\n\nFINAL RESULTS")
    print("=" * 70)

    for result in results:

        print(f"\nClaim   : {result.claim_text}")
        print(f"Verdict : {result.verdict}")

        print(
            f"Supporting evidence   : "
            f"{result.supporting_evidence_ids}"
        )

        print(
            f"Contradicting evidence: "
            f"{result.contradicting_evidence_ids}"
        )

        print(
            f"Irrelevant evidence   : "
            f"{result.irrelevant_evidence_ids}"
        )

        print("\nPassage-level judgments:")

        for judgment in result.passage_judgments:

            print(
                f"\n  Evidence ID : {judgment.evidence_id}"
            )

            print(
                f"  Source      : {judgment.source_name}"
            )

            print(
                f"  Verdict     : {judgment.verdict}"
            )

            print(
                f"  Retrieval   : {judgment.retrieval_score:.4f}"
            )

            print(
                f"  Explanation : {judgment.explanation}"
            )

        print(
            f"\nOverall explanation: "
            f"{result.explanation}"
        )

        print("-" * 70)

    # Calculate response-level evidence score
    evidence_score = calculate_response_evidence_score(
        results
    )

    print_evidence_support_score(
        evidence_score
    )
