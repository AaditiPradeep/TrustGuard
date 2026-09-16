"""
TrustGuard — trusted evidence source registry.

This is the *baseline authority* half of RAVE:  Ri = alpha * Bi + (1 - alpha) * Ci
Bi (baseline authority) comes from this static hierarchy.
Ci (corroboration) is computed later, per-claim, in the Source Reliability module,
by checking how often a source's claims agree with independently-corroborated ones.

Keeping Bi here as a single source of truth means the Evidence Retrieval,
Source Reliability, and Citation Quality modules all agree on what "tier" a
source belongs to.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDef:
    name: str
    tier: str            # "regulatory" | "clinical_guideline" | "primary_literature" | "reference"
    baseline_authority: float  # Bi, in [0, 1]
    base_url: str
    notes: str = ""


# Baseline authority (Bi) reflects editorial/regulatory rigor, not popularity.
# Tune these during Phase 4 (Source Reliability / RAVE) once corroboration
# data (Ci) is available to calibrate against.
SOURCE_REGISTRY: dict[str, SourceDef] = {
    "who": SourceDef(
        name="WHO",
        tier="regulatory",
        baseline_authority=0.95,
        base_url="https://www.who.int",
        notes="World Health Organization fact sheets — global public health authority.",
    ),
    "openfda": SourceDef(
        name="openFDA",
        tier="regulatory",
        baseline_authority=0.95,
        base_url="https://api.fda.gov",
        notes="FDA drug labeling / adverse event data — regulatory ground truth for US drug safety.",
    ),
    "cdc": SourceDef(
        name="CDC",
        tier="regulatory",
        baseline_authority=0.93,
        base_url="https://www.cdc.gov",
        notes="US Centers for Disease Control and Prevention.",
    ),
    "medlineplus": SourceDef(
        name="MedlinePlus",
        tier="reference",
        baseline_authority=0.85,
        base_url="https://medlineplus.gov",
        notes="NIH/NLM consumer health reference, curated from clinical literature.",
    ),
    "pubmed": SourceDef(
        name="PubMed",
        tier="primary_literature",
        baseline_authority=0.75,
        base_url="https://pubmed.ncbi.nlm.nih.gov",
        notes=(
            "Primary literature via NCBI E-utilities. Wide quality variance "
            "(preprints excluded, but study quality still varies) -> Bi set "
            "lower than regulatory/reference tiers; Ci should do more work here."
        ),
    ),
    "pubmedqa_context": SourceDef(
        name="PubMedQA-context",
        tier="primary_literature",
        baseline_authority=0.70,
        base_url="https://pubmedqa.github.io",
        notes="Abstract context bundled with the PubMedQA benchmark itself.",
    ),
}


def get_source(key: str) -> SourceDef:
    try:
        return SOURCE_REGISTRY[key]
    except KeyError as e:
        raise KeyError(
            f"Unknown source key '{key}'. Known sources: {list(SOURCE_REGISTRY)}"
        ) from e


def baseline_authority(key: str) -> float:
    return get_source(key).baseline_authority
