"""
TrustGuard — medical QA dataset loading.

Two tiers, matching the project's own feasibility note ("public medical QA
and hallucination-benchmark datasets... are available for evaluation"):

1. load_pubmedqa() / load_medmcqa() — pull the real benchmarks via the
   HuggingFace `datasets` library. Needs `pip install datasets` and internet
   on first call (cached locally after that).

2. SEED_QA_ITEMS — a small hand-written, offline set of Med-HALT-style
   questions (reasoning + a couple of adversarial "confident but wrong"
   traps) so claim-extraction / retrieval / scoring code can be developed
   and unit-tested with zero network access, e.g. inside this sandbox.
   Swap in the real benchmarks once you're running with internet.
"""

from __future__ import annotations

from data.schema import QAItem

# ---------------------------------------------------------------------------
# Tier 2: offline seed set
# ---------------------------------------------------------------------------
# A mix of: (a) straightforward factual questions, (b) a "confidently wrong"
# trap similar to what Med-HALT's memory-based tests probe, and (c) a
# citation-bearing style claim so Citation Quality has something to chew on
# from day one.

SEED_QA_ITEMS: list[QAItem] = [
    QAItem(
        question="Is it safe to take metformin with contrast dye for a CT scan?",
        answer=(
            "Metformin is typically held before and after iodinated contrast "
            "in patients with reduced kidney function, due to a small risk of "
            "contrast-induced nephropathy leading to metformin accumulation "
            "and lactic acidosis; guidance varies by renal function."
        ),
        dataset="seed_medhalt_style",
        metadata={"category": "drug_safety", "difficulty": "reasoning"},
    ),
    QAItem(
        question="What is the maximum recommended daily dose of acetaminophen for a healthy adult?",
        answer="Commonly cited maximum is 4000 mg/day for a healthy adult, though many clinicians now recommend a lower ceiling (e.g. 3000 mg/day) for routine use.",
        dataset="seed_medhalt_style",
        metadata={"category": "dosage", "difficulty": "factual_recall"},
    ),
    QAItem(
        question="Does the FDA classify insulin as an over-the-counter medication in the United States?",
        answer="No — insulin is a prescription medication in the US (with limited pharmacist-dispensing exceptions in some states); it is not classified as OTC.",
        dataset="seed_medhalt_style",
        metadata={"category": "regulatory_trap", "difficulty": "adversarial"},
    ),
    QAItem(
        question="Can a patient with a penicillin allergy safely take cephalexin?",
        answer=(
            "Cross-reactivity between penicillins and cephalosporins is lower "
            "than historically assumed (roughly 1-2%), but caution and "
            "clinical judgment are still advised, especially with a history "
            "of severe reaction."
        ),
        dataset="seed_medhalt_style",
        metadata={"category": "drug_safety", "difficulty": "nuanced"},
    ),
    QAItem(
        question="Is warfarin safe to combine with NSAIDs like ibuprofen for routine pain relief?",
        answer="No — combining warfarin with NSAIDs significantly increases bleeding risk and is generally avoided; acetaminophen is preferred for routine pain relief in patients on warfarin.",
        dataset="seed_medhalt_style",
        metadata={"category": "drug_interaction", "difficulty": "high_stakes"},
    ),
]


def load_seed_qa() -> list[QAItem]:
    return SEED_QA_ITEMS


# ---------------------------------------------------------------------------
# Tier 1: real benchmarks via HuggingFace `datasets` (requires internet)
# ---------------------------------------------------------------------------

def load_pubmedqa(config: str = "pqa_labeled", split: str = "train", limit: int | None = 200) -> list[QAItem]:
    """
    PubMedQA: yes/no/maybe biomedical questions, each paired with the PubMed
    abstract context that supports the answer — useful because it gives us
    claim + evidence pairs "for free" to bootstrap the Evidence Support module.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Run `pip install datasets` (needs internet) to load the real "
            "PubMedQA benchmark. Use load_seed_qa() for offline development."
        ) from e

    ds = load_dataset("qiaojin/PubMedQA", config, split=split)
    items = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        context_text = " ".join(row.get("context", {}).get("contexts", []))
        items.append(
            QAItem(
                question=row["question"],
                answer=row.get("long_answer", "") or row.get("final_decision", ""),
                context=context_text,
                dataset="pubmedqa",
                metadata={"final_decision": row.get("final_decision", "")},
            )
        )
    return items


def load_medmcqa(split: str = "validation", limit: int | None = 200) -> list[QAItem]:
    """
    MedMCQA: large-scale multiple-choice medical entrance-exam questions.
    Good breadth for exercising claim extraction across many topics quickly.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Run `pip install datasets` (needs internet) to load the real "
            "MedMCQA benchmark. Use load_seed_qa() for offline development."
        ) from e

    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    options_keys = ["opa", "opb", "opc", "opd"]
    items = []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        correct_idx = row.get("cop")  # 0-3 index of correct option
        answer = row.get(options_keys[correct_idx], "") if correct_idx is not None else ""
        items.append(
            QAItem(
                question=row["question"],
                answer=answer,
                dataset="medmcqa",
                metadata={"subject": row.get("subject_name", "")},
            )
        )
    return items


if __name__ == "__main__":
    for item in load_seed_qa():
        print(f"[{item.item_id}] ({item.metadata.get('category')}) {item.question}")
