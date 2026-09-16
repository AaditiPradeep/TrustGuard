"""
Smoke test for the dataset + evidence corpus layer, runnable fully offline.

Run with:  python -m tests.test_pipeline_smoke
"""

from data.load_qa_dataset import load_seed_qa
from retrieval.index import EvidenceIndex
from tests.mock_corpus import MOCK_PASSAGES


def main():
    qa_items = load_seed_qa()
    index = EvidenceIndex(MOCK_PASSAGES)

    print(f"Loaded {len(qa_items)} seed QA items and {len(index)} mock evidence passages.\n")

    for item in qa_items:
        print("=" * 100)
        print(f"Q: {item.question}")
        print(f"   (dataset={item.dataset}, category={item.metadata.get('category')})")
        results = index.query(item.answer or item.question, top_k=2)
        if not results:
            print("   -> No evidence retrieved above similarity threshold.")
            continue
        for r in results:
            print(f"   -> [{r.score:.3f}] {r.passage.source_name} ({r.passage.source_tier}): {r.passage.title}")
            print(f"      {r.passage.text[:160]}...")
    print("=" * 100)


if __name__ == "__main__":
    main()
