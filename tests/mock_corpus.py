"""
A tiny hand-written evidence corpus so retrieval + downstream modules can be
unit-tested without network access. Mirrors the shape a real PubMed/
MedlinePlus/openFDA pull would produce (see data/fetch_*.py).
"""

from data.schema import EvidencePassage

MOCK_PASSAGES: list[EvidencePassage] = [
    EvidencePassage(
        text=(
            "Metformin should be temporarily discontinued at the time of, or "
            "prior to, iodinated contrast imaging procedures in patients with "
            "an eGFR between 30-60 mL/min/1.73m2, and restarted 48 hours "
            "after the procedure provided renal function is stable, due to "
            "the risk of contrast-induced acute kidney injury and subsequent "
            "metformin-associated lactic acidosis."
        ),
        source_name="openFDA",
        source_tier="regulatory",
        title="Metformin — warnings",
        url="https://labels.fda.gov/mock-metformin",
        identifier="mock:metformin:warnings",
        published_date="2023",
    ),
    EvidencePassage(
        text=(
            "Insulin products in the United States are prescription-only "
            "medications regulated by the FDA; no insulin formulation is "
            "approved for over-the-counter sale nationwide, though some "
            "states permit pharmacist-initiated dispensing without an "
            "individual prescription under collaborative practice laws."
        ),
        source_name="MedlinePlus",
        source_tier="reference",
        title="Insulin — regulatory status",
        url="https://medlineplus.gov/mock-insulin",
        identifier="mock:medlineplus:insulin",
        published_date="2024",
    ),
    EvidencePassage(
        text=(
            "Concomitant use of warfarin with nonsteroidal anti-inflammatory "
            "drugs (NSAIDs) is associated with a substantially increased risk "
            "of gastrointestinal and other bleeding events, both through "
            "pharmacodynamic (platelet inhibition, GI mucosal injury) and "
            "pharmacokinetic interactions; acetaminophen is generally "
            "preferred for analgesia in anticoagulated patients."
        ),
        source_name="WHO",
        source_tier="regulatory",
        title="Anticoagulant drug interactions",
        url="https://www.who.int/mock-warfarin-nsaid",
        identifier="mock:who:warfarin_nsaid",
        published_date="2022",
    ),
    EvidencePassage(
        text=(
            "Historical estimates of penicillin-cephalosporin cross-reactivity "
            "of 8-10% were based on studies confounded by shared manufacturing "
            "processes; contemporary estimates using second- and "
            "third-generation cephalosporins with dissimilar side chains put "
            "cross-reactivity closer to 1-2%."
        ),
        source_name="PubMed",
        source_tier="primary_literature",
        title="Beta-lactam cross-reactivity revisited",
        url="https://pubmed.ncbi.nlm.nih.gov/mock12345/",
        identifier="mock12345",
        published_date="2021",
    ),
    EvidencePassage(
        text=(
            "The maximum recommended daily dose of acetaminophen for a "
            "healthy adult is 4000 mg per 24-hour period from all combined "
            "sources; however, many hepatology and pharmacy references now "
            "recommend a more conservative ceiling of 3000 mg/day for "
            "routine, non-medically-supervised use."
        ),
        source_name="MedlinePlus",
        source_tier="reference",
        title="Acetaminophen — dosage",
        url="https://medlineplus.gov/mock-acetaminophen",
        identifier="mock:medlineplus:apap",
        published_date="2024",
    ),
]
