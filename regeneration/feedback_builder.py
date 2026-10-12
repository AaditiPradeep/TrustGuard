"""Turn a TrustGuard result into targeted feedback and a regeneration prompt.

Reads ``result["evidence_support"]["claim_verification"]`` (a list of
``EvidenceSupportResult.to_dict()``) produced by verification/evidence_support.py.
Claim verdicts there are: SUPPORTED, CONTRADICTED, MIXED, IRRELEVANT.
(metrics/evidence_support_score.py calls IRRELEVANT "UNVERIFIED" - both are
handled here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED = {"SUPPORTED"}
CONTRADICTED = {"CONTRADICTED", "MIXED"}      # MIXED = conflicting evidence, fix it
UNVERIFIABLE = {"IRRELEVANT", "UNVERIFIED"}   # nothing in the corpus backs it

MAX_EVIDENCE_PER_CLAIM = 2
MAX_EVIDENCE_CHARS = 450


@dataclass
class ClaimFeedback:
    claim_id: str
    text: str
    verdict: str
    # (source_name, passage text) pairs used to correct the claim
    evidence: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Feedback:
    contradicted: list[ClaimFeedback] = field(default_factory=list)
    unverifiable: list[ClaimFeedback] = field(default_factory=list)
    supported: list[ClaimFeedback] = field(default_factory=list)

    @property
    def needs_fix(self) -> bool:
        return bool(self.contradicted or self.unverifiable)

    def summary(self) -> dict:
        return {
            "contradicted": len(self.contradicted),
            "unverifiable": len(self.unverifiable),
            "supported": len(self.supported),
        }


def _passage_text(item: dict, evidence_id: str) -> tuple[str, str]:
    passage = (item.get("evidence_passages") or {}).get(evidence_id) or {}
    text = " ".join(str(passage.get("text", "")).split())[:MAX_EVIDENCE_CHARS]
    source = passage.get("source_name") or ""
    return source, text


def _pick_evidence(item: dict, verdict: str) -> list[tuple[str, str]]:
    """Choose the passages that best explain the correction for this claim."""
    judgments = item.get("passage_judgments") or []
    if verdict in CONTRADICTED:
        # Passages that contradict the claim hold the correct information.
        chosen = sorted(
            (j for j in judgments if j.get("verdict") == "CONTRADICTED"),
            key=lambda j: j.get("contradiction_probability", 0.0), reverse=True,
        )
    else:
        # Unverifiable: show the most relevant retrieved passages so the model
        # can see what the evidence actually says instead.
        chosen = sorted(
            judgments, key=lambda j: j.get("retrieval_weight", 0.0), reverse=True
        )
    out = []
    for j in chosen[:MAX_EVIDENCE_PER_CLAIM]:
        source, text = _passage_text(item, j.get("evidence_id", ""))
        if text:
            out.append((source or j.get("source_name", ""), text))
    return out


def extract_feedback(result: dict) -> Feedback:
    """Sort every verified claim into contradicted / unverifiable / supported."""
    items = (result.get("evidence_support") or {}).get("claim_verification") or []
    fb = Feedback()
    for item in items:
        verdict = str(item.get("verdict", "")).upper()
        cf = ClaimFeedback(
            claim_id=str(item.get("claim_id", "")),
            text=str(item.get("claim_text", "")).strip(),
            verdict=verdict,
        )
        if verdict in SUPPORTED:
            fb.supported.append(cf)
        elif verdict in CONTRADICTED:
            cf.evidence = _pick_evidence(item, verdict)
            fb.contradicted.append(cf)
        else:  # IRRELEVANT / UNVERIFIED / anything unexpected is never kept silently
            cf.evidence = _pick_evidence(item, verdict)
            fb.unverifiable.append(cf)
    return fb


def _evidence_lines(c: ClaimFeedback, label: str) -> list[str]:
    return [f"   {label} ({src or 'source'}): {txt}" for src, txt in c.evidence]


def build_regeneration_prompt(question: str, original_answer: str, fb: Feedback) -> str:
    """Evidence-grounded prompt that fixes only the flagged claims."""
    lines = [
        "Revise the DRAFT ANSWER below using ONLY the evidence provided.",
        "",
        f"QUESTION:\n{question}",
        "",
        f"DRAFT ANSWER:\n{original_answer}",
        "",
    ]
    if fb.supported:
        lines.append("VERIFIED-CORRECT CLAIMS (keep these; do not change their meaning):")
        lines += [f"- {c.text}" for c in fb.supported]
        lines.append("")
    if fb.contradicted:
        lines.append("CLAIMS CONTRADICTED BY EVIDENCE (correct each one, or remove it):")
        for n, c in enumerate(fb.contradicted, 1):
            lines.append(f"{n}. {c.text}")
            lines += _evidence_lines(c, "Evidence")
        lines.append("")
    if fb.unverifiable:
        lines.append(
            "CLAIMS NOT SUPPORTED BY ANY EVIDENCE (remove, or rewrite with "
            "hedging such as 'may'):"
        )
        for n, c in enumerate(fb.unverifiable, 1):
            lines.append(f"{n}. {c.text}")
            lines += _evidence_lines(c, "Related evidence")
        lines.append("")
    lines += [
        "RULES:",
        "1. Keep every verified-correct claim.",
        "2. Change only the flagged claims. Do not add facts that are not in the evidence.",
        "3. If the evidence is not enough to answer a point, say that instead of guessing.",
        "4. Keep it concise. Do not invent sources or citation markers.",
        "5. Return ONLY the revised answer, with no preamble and no explanation.",
    ]
    return "\n".join(lines)