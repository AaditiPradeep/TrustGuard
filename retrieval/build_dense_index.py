"""Build a persistent FAISS semantic index from one or more corpus JSONL files."""

from __future__ import annotations

import argparse

from retrieval.dense_index import DEFAULT_MODEL, DenseEvidenceIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", required=True, help="One or more evidence JSONL files")
    parser.add_argument("--out", required=True, help="Directory for evidence.faiss and metadata")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence-transformers model name")
    parser.add_argument("--chunk-size", type=int, default=900, help="Maximum characters per evidence chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap between chunks in characters")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    index = DenseEvidenceIndex.from_jsonl(
        *args.corpus,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    index.save(args.out)
    print(f"Indexed {len(index)} evidence chunks with {args.model} -> {args.out}")


if __name__ == "__main__":
    main()
