"""
TrustGuard — trusted evidence source registry.

This defines the baseline authority tiers used by RAVE. The Source Reliability
module combines each tier with NLI support or contradiction using banded ranges.

Keeping Bi here as a single source of truth means the Evidence Retrieval,
Source Reliability, and Citation Quality modules all agree on what "tier" a
source belongs to.
"""

from dataclasses import dataclass


AUTHORITY_CRITERIA_WEIGHTS = {
    "institutional_mandate": 0.30,
    "editorial_governance": 0.30,
    "update_version_control": 0.20,
    "evidence_traceability": 0.20,
}


@dataclass(frozen=True)
class AuthorityRubric:
    """Transparent 1-5 justification for a source authority tier.

    This rubric documents the policy behind a baseline; it does not claim
    that a baseline is a calibrated real-world probability.
    """

    institutional_mandate: int
    editorial_governance: int
    update_version_control: int
    evidence_traceability: int

    @property
    def weighted_score(self) -> float:
        return sum(
            getattr(self, criterion) * weight
            for criterion, weight in AUTHORITY_CRITERIA_WEIGHTS.items()
        )


@dataclass(frozen=True)
class SourceDef:
    name: str
    tier: str            # "regulatory" | "clinical_guideline" | "primary_literature" | "reference"
    baseline_authority: float  # Bi, in [0, 1]
    base_url: str
    rubric: AuthorityRubric
    notes: str = ""


# Baseline authority reflects editorial/regulatory rigor, not popularity.
SOURCE_REGISTRY: dict[str, SourceDef] = {
    "who": SourceDef(
        name="WHO",
        tier="regulatory",
        baseline_authority=0.95,
        base_url="https://www.who.int",
        rubric=AuthorityRubric(5, 5, 5, 5),
        notes="World Health Organization fact sheets — global public health authority.",
    ),
    "openfda": SourceDef(
        name="openFDA",
        tier="regulatory",
        baseline_authority=0.95,
        base_url="https://api.fda.gov",
        rubric=AuthorityRubric(5, 5, 5, 5),
        notes=(
            "FDA structured drug labels — regulatory source for approved "
            "drug warnings, interactions, dosage, and boxed warnings. "
            "This rating does not apply to FAERS adverse-event reports."
        ),
    ),
    "cdc": SourceDef(
        name="CDC",
        tier="regulatory",
        baseline_authority=0.95,
        base_url="https://www.cdc.gov",
        rubric=AuthorityRubric(5, 5, 5, 5),
        notes="US Centers for Disease Control and Prevention.",
    ),
    "medlineplus": SourceDef(
        name="MedlinePlus",
        tier="reference",
        baseline_authority=0.85,
        base_url="https://medlineplus.gov",
        rubric=AuthorityRubric(4, 4, 4, 4),
        notes="NIH/NLM consumer health reference, curated from clinical literature.",
    ),
    "pubmed": SourceDef(
        name="PubMed",
        tier="primary_literature",
        baseline_authority=0.75,
        base_url="https://pubmed.ncbi.nlm.nih.gov",
        rubric=AuthorityRubric(3, 3, 3, 3),
        notes=(
            "Primary literature via NCBI E-utilities. Wide quality variance "
            "(preprints excluded, but study quality still varies) -> Bi set "
            "lower than regulatory/reference tiers."
        ),
    ),
    "pubmedqa_context": SourceDef(
        name="PubMedQA-context",
        tier="primary_literature",
        baseline_authority=0.70,
        base_url="https://pubmedqa.github.io",
        rubric=AuthorityRubric(3, 3, 2, 3),
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


def authority_rubric(key: str) -> dict:
    """Return the documented rationale for a source's baseline authority."""
    source = get_source(key)
    return {
        "source": source.name,
        "baseline_authority": source.baseline_authority,
        "criteria": {
            criterion: getattr(source.rubric, criterion)
            for criterion in AUTHORITY_CRITERIA_WEIGHTS
        },
        "weights": AUTHORITY_CRITERIA_WEIGHTS.copy(),
        "weighted_score_out_of_5": source.rubric.weighted_score,
    }
