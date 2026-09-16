"""
Manual spot-check: run several different claims through the real hybrid
retriever and print the fused results for each, so you can eyeball whether
fusion is behaving sensibly across topics (not just the one metformin query).

Run with:  python test_hybrid_manual.py
"""

from retrieval.hybrid import HybridRetriever

CLAIMS = [
    "is metformin safe with contrast dye",
    "warfarin should not be combined with NSAIDs due to bleeding risk",
    "penicillin and cephalosporins have a high rate of cross-reactivity",
    "insulin is available over the counter without a prescription",
    "the maximum safe daily dose of acetaminophen is 4000 mg",
]


def main():
    print("Loading indexes (TF-IDF + dense)... this may take a few seconds.\n")
    hr = HybridRetriever.from_jsonl(
        "cache/evidence_corpus.jsonl",
        dense_index_dir="cache/dense_evidence_index",
    )

    for claim in CLAIMS:
        print("=" * 100)
        print(f"CLAIM: {claim}")
        print("=" * 100)
        results = hr.query(claim, top_k=5, require_both=True)
        if not results:
            print("  -> no results")
            continue
        for r in results:
            both = "BOTH" if r.in_both else ("tfidf only" if r.tfidf_rank else "dense only")
            print(
                f"  fused={r.score:.4f}  [{both:10s}]  "
                f"tfidf_rank={r.tfidf_rank}  dense_rank={r.dense_rank}  "
                f"-> {r.passage.source_name}: {r.passage.title}"
            )
        print()


if __name__ == "__main__":
    main()