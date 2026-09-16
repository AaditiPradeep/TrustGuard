"""
TrustGuard — context-based citation quality evaluation.

IMPORTANT DESIGN:
The four citation-quality labels are based on relevance to the USER'S
QUESTION, not merely whether the source agrees with the LLM claim.

    DIRECT          = cited information directly answers the question.
    INDIRECT        = cited information is related to the question but does
                      not directly answer it.
    IRRELEVANT      = cited information is unrelated to the question.
    MISATTRIBUTED   = the citation/source identity is incorrectly attributed.

Support/contradiction is a SEPARATE field:
    SUPPORTS | CONTRADICTS | NEUTRAL | UNCLEAR

Therefore a source can be:
    DIRECT + SUPPORTS
    DIRECT + CONTRADICTS
    DIRECT + NEUTRAL
    INDIRECT + SUPPORTS
    etc.

This prevents a citation that directly answers a question by contradicting
the LLM from being incorrectly labelled "irrelevant."
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from citation.citation_extractor import ClaimCitation


# Relevance score used for the response-level citation-quality metric.
# Misattribution is kept as a separate error condition and receives zero
# citation-quality credit by default.
CITATION_SCORES = {
    "direct": 1.0,
    "indirect": 0.5,
    "irrelevant": 0.0,
    "misattributed": 0.0,
}

VALID_LABELS = frozenset(CITATION_SCORES)
VALID_STANCES = frozenset({"supports", "contradicts", "neutral", "unclear"})


@dataclass(frozen=True)
class CitationJudgement:
    claim_id: str
    citation_id: str
    label: str
    score: float
    stance: str = "unclear"
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


JUDGE_SYSTEM_PROMPT = """You are evaluating the quality of a citation in a
question-answering system.

The classification MUST be based on the USER QUESTION and the information
contained in the CITED SOURCE.

Use exactly one relevance label:

1. direct
   The cited source contains information that directly answers the user's
   question. It can SUPPORT or CONTRADICT the LLM's claim. Agreement with
   the LLM is NOT required for a citation to be direct.

2. indirect
   The cited source contains information related to the user's question or
   topic, but it does not directly answer what the user asked.

3. irrelevant
   The cited source is completely unrelated to the user's question.

4. misattributed
   The citation is presented as pointing to a particular source, but the
   source identity/attribution is wrong or the cited source cannot reasonably
   be the source represented by that citation.

Separately classify the relationship between the cited information and the
LLM claim:

- supports: the source supports the claim
- contradicts: the source conflicts with the claim
- neutral: the source neither supports nor contradicts the claim
- unclear: insufficient information to determine

IMPORTANT:
- Do NOT call a citation irrelevant just because it contradicts the claim.
- First decide whether it answers the USER QUESTION.
- The question is the primary context for direct/indirect/irrelevant.
- A direct contradiction is still DIRECT.

Return JSON only:
{
  "label": "direct|indirect|irrelevant|misattributed",
  "stance": "supports|contradicts|neutral|unclear",
  "rationale": "brief explanation"
}
"""


def build_judge_prompt(pair: ClaimCitation) -> str:
    return (
        JUDGE_SYSTEM_PROMPT
        + "\n\nUSER QUESTION:\n"
        + pair.question
        + "\n\nLLM CLAIM:\n"
        + pair.claim_text
        + "\n\nCITATION AS SHOWN IN RESPONSE:\n"
        + pair.citation_text
        + "\n\nCITED SOURCE TITLE:\n"
        + pair.source_title
        + "\n\nCITED SOURCE NAME:\n"
        + pair.source_name
        + "\n\nCITED SOURCE IDENTIFIER:\n"
        + pair.source_identifier
        + "\n\nCITED SOURCE URL:\n"
        + pair.source_url
        + "\n\nCITED SOURCE CONTENT:\n"
        + pair.source_text
        + "\n"
    )


def _parse_judge_output(output: Any) -> tuple[str, str, str]:
    if isinstance(output, Mapping):
        data = output
    else:
        text = str(output).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError("LLM judge did not return JSON.")
            data = json.loads(match.group(0))

    label = str(data.get("label", "")).strip().lower().replace(" ", "_")
    aliases = {
        "direct_support": "direct",
        "indirect_support": "indirect",
        "not_supported": "irrelevant",
        "misattributed_support": "misattributed",
    }
    label = aliases.get(label, label)

    stance = str(data.get("stance", "unclear")).strip().lower()
    stance = stance.replace(" ", "_")
    stance_aliases = {
        "support": "supports",
        "contradiction": "contradicts",
        "contradict": "contradicts",
        "none": "neutral",
        "not_applicable": "neutral",
    }
    stance = stance_aliases.get(stance, stance)

    if label not in VALID_LABELS:
        raise ValueError(f"Invalid citation-quality label: {label!r}")
    if stance not in VALID_STANCES:
        raise ValueError(f"Invalid citation stance: {stance!r}")

    return label, stance, str(data.get("rationale", "")).strip()


def judge_claim_citation(
    pair: ClaimCitation,
    judge: Callable[[str], Any],
) -> CitationJudgement:
    label, stance, rationale = _parse_judge_output(judge(build_judge_prompt(pair)))
    return CitationJudgement(
        claim_id=pair.claim_id,
        citation_id=pair.citation_id,
        label=label,
        score=CITATION_SCORES[label],
        stance=stance,
        rationale=rationale,
    )


def score_citation_pairs(
    pairs: Sequence[ClaimCitation],
    judge: Callable[[str], Any],
) -> list[CitationJudgement]:
    return [judge_claim_citation(pair, judge) for pair in pairs]


def aggregate_citation_quality(
    judgements: Sequence[CitationJudgement],
) -> float:
    """Average citation relevance/attribution quality in [0, 1]."""
    if not judgements:
        return 0.0
    return sum(j.score for j in judgements) / len(judgements)


def citation_quality_report(
    pairs: Sequence[ClaimCitation],
    judgements: Sequence[CitationJudgement],
) -> dict:
    counts = {label: 0 for label in CITATION_SCORES}
    stance_counts = {stance: 0 for stance in VALID_STANCES}

    for judgement in judgements:
        counts[judgement.label] += 1
        stance_counts[judgement.stance] += 1

    return {
        "citation_quality_score": aggregate_citation_quality(judgements),
        "pair_count": len(judgements),
        "label_counts": counts,
        "stance_counts": stance_counts,
        "judgements": [j.to_dict() for j in judgements],
    }
