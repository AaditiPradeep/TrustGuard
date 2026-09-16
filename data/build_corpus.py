"""
TrustGuard — unified evidence corpus builder.

Combines the evidence collected from:

    - WHO
    - CDC
    - NIH / MedlinePlus
    - FDA / openFDA
    - Existing PubMed corpus

The final corpus is written to:
    cache/evidence_corpus.jsonl

Usage:
    python -m data.build_corpus --out cache/evidence_corpus.jsonl
"""

from __future__ import annotations

import argparse
import os

from data.schema import EvidencePassage, read_jsonl, write_jsonl


# ---------------------------------------------------------
# Evidence files
# ---------------------------------------------------------

DEFAULT_FILES = {
    "Existing corpus": "cache/evidence_corpus.jsonl",
    "WHO": "cache/who_evidence.jsonl",
    "CDC": "cache/cdc_evidence.jsonl",
    "NIH / MedlinePlus": "cache/medlineplus_evidence.jsonl",
    "FDA / openFDA": "cache/fda_evidence.jsonl",
}


# ---------------------------------------------------------
# Load evidence
# ---------------------------------------------------------

def load_evidence_file(path: str) -> list[EvidencePassage]:
    """Load EvidencePassage records from a JSONL file."""

    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return []

    try:
        records = read_jsonl(path)
    except Exception as e:
        print(f"  [ERROR] Could not read {path}: {e}")
        return []

    passages: list[EvidencePassage] = []

    for record in records:

        text = str(record.get("text", "")).strip()

        if not text:
            continue

        try:
            passage = EvidencePassage(
                text=text,
                source_name=record.get("source_name", ""),
                source_tier=record.get("source_tier", ""),
                title=record.get("title", ""),
                url=record.get("url", ""),
                identifier=record.get("identifier", ""),
                published_date=record.get("published_date", ""),
                retrieved_at=record.get("retrieved_at", ""),
                passage_id=record.get("passage_id", ""),
            )

            passages.append(passage)

        except Exception as e:
            print(f"  [WARNING] Invalid record skipped: {e}")

    return passages


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_passage(passage: EvidencePassage) -> bool:
    """
    Basic validation for evidence entering the TrustGuard corpus.
    """

    text = passage.text.strip()

    if not text:
        return False

    # Avoid tiny fragments that are unlikely to provide
    # useful factual evidence.
    if len(text) < 50:
        return False

    if not passage.source_name.strip():
        return False

    return True


# ---------------------------------------------------------
# Build corpus
# ---------------------------------------------------------

def build_corpus(
    evidence_files: dict[str, str]
) -> list[EvidencePassage]:

    all_passages: list[EvidencePassage] = []

    print("\n" + "=" * 65)
    print("LOADING TRUSTGUARD EVIDENCE SOURCES")
    print("=" * 65)

    for source, path in evidence_files.items():

        print(f"\n[{source}]")
        print(f"  File: {path}")

        passages = load_evidence_file(path)

        print(f"  Loaded: {len(passages)}")

        valid = [
            p for p in passages
            if validate_passage(p)
        ]

        print(f"  Valid:  {len(valid)}")

        all_passages.extend(valid)

    # -----------------------------------------------------
    # Deduplicate
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("DEDUPLICATING")
    print("=" * 65)

    seen_ids: set[str] = set()
    deduped: list[EvidencePassage] = []

    for passage in all_passages:

        if passage.passage_id in seen_ids:
            continue

        seen_ids.add(passage.passage_id)
        deduped.append(passage)

    print(f"Before deduplication: {len(all_passages)}")
    print(f"After deduplication:  {len(deduped)}")

    return deduped


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    ap = argparse.ArgumentParser(
        description="Build the unified TrustGuard evidence corpus."
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output merged JSONL path"
    )

    ap.add_argument(
        "--existing",
        default=DEFAULT_FILES["Existing corpus"],
        help="Existing corpus containing the original PubMed/FDA data"
    )

    ap.add_argument(
        "--who",
        default=DEFAULT_FILES["WHO"],
        help="WHO evidence JSONL"
    )

    ap.add_argument(
        "--cdc",
        default=DEFAULT_FILES["CDC"],
        help="CDC evidence JSONL"
    )

    ap.add_argument(
        "--medlineplus",
        default=DEFAULT_FILES["NIH / MedlinePlus"],
        help="NIH / MedlinePlus evidence JSONL"
    )

    ap.add_argument(
        "--fda",
        default=DEFAULT_FILES["FDA / openFDA"],
        help="FDA/openFDA evidence JSONL"
    )

    args = ap.parse_args()

    evidence_files = {
        "Existing corpus": args.existing,
        "WHO": args.who,
        "CDC": args.cdc,
        "NIH / MedlinePlus": args.medlineplus,
        "FDA / openFDA": args.fda,
    }

    # -----------------------------------------------------
    # Important safety check
    # -----------------------------------------------------
    #
    # If input and output are the same file, we first load
    # everything into memory before writing, so the existing
    # corpus is not lost.
    #

    passages = build_corpus(evidence_files)

    if not passages:
        print("\nERROR: No valid evidence found.")
        return

    # -----------------------------------------------------
    # Write unified corpus
    # -----------------------------------------------------

    write_jsonl(passages, args.out)

    # -----------------------------------------------------
    # Source statistics
    # -----------------------------------------------------

    by_source: dict[str, int] = {}

    for passage in passages:
        source = passage.source_name or "Unknown"
        by_source[source] = by_source.get(source, 0) + 1

    print("\n" + "=" * 65)
    print("FINAL TRUSTGUARD EVIDENCE CORPUS")
    print("=" * 65)

    print(f"\nTotal passages: {len(passages)}")
    print(f"Output: {args.out}\n")

    for source, count in sorted(by_source.items()):
        print(f"  {source:<25} {count}")

    print("\nCorpus build complete.")


if __name__ == "__main__":
    main()