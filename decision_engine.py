"""Decision rules for a TrustGuard aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


HIGH_RISK_TERMS = (
    "dose", "dosage", "overdose", "drug interaction", "bleeding",
    "pregnan", "insulin", "anticoagul", "chest pain", "suicid",
    "emergency", "anaphyl", "allerg",
)


@dataclass
class TrustDecision:
    decision: str
    reason: str
    regeneration_required: bool
    safety_gate_triggered: bool
    threshold: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decide_response(
    aggregation: Mapping[str, Any], *, threshold: float = 70.0,
    enable_clinical_safety_gate: bool = True,
) -> TrustDecision:
    """Accept only a sufficiently trusted answer with no contradicted claim."""
    score = aggregation.get("composite_score")
    claims = aggregation.get("claim_contributions", [])
    contradicted = [item for item in claims if item.get("status") == "CONTRADICTED"]
    high_risk = [
        item for item in contradicted
        if any(term in item.get("claim_text", "").lower() for term in HIGH_RISK_TERMS)
    ]
    flags = []
    if score is None:
        flags.append("The overall score could not be calculated because no metric was available.")
    elif score < threshold:
        flags.append(f"The overall trust score ({score:.2f}/100) is below the {threshold:.0f}/100 threshold.")
    if contradicted:
        flags.append(f"{len(contradicted)} claim(s) conflict with retrieved evidence.")
    safety_gate = bool(high_risk) and enable_clinical_safety_gate
    if safety_gate:
        flags.append("A contradicted claim concerns a potentially high-risk medical topic.")

    regenerate = score is None or score < threshold or bool(contradicted)
    if regenerate:
        reason = "Regenerate the response and address the flagged claim(s) before presenting it as reliable."
        return TrustDecision("REGENERATE", reason, True, safety_gate, threshold, flags)
    return TrustDecision(
        "ACCEPT",
        "The response meets the trust threshold and no claim was contradicted by the retrieved evidence.",
        False,
        False,
        threshold,
        ["No decision-blocking evidence conflict was found."],
    )
