"""Score a TrustGuard clean/corrupted-pair JSONL without Answer Stability.

For fixed answers, Answer Stability is not a valid outcome variable: it
measures variation across separately generated answers to a question. This
runner sets ``compute_stability=False`` and lets the existing aggregator
renormalise the remaining metric weights.

Optional explicit citations use this format::

  "citations": [{"source_id": "<passage_id>", "sentence_index": 0}]

The runner adds numbered markers to the indicated answer sentences and passes
the matching corpus passages to Citation Quality. A citation-swap corruption
therefore changes only the cited source, while keeping the claim sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.schema import EvidencePassage
from pipeline import run_trustguard
from evaluation.generate_corruption_dataset import sentence_parts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_by_id(path: Path) -> dict[str, EvidencePassage]:
    records = {}
    for row in read_jsonl(path):
        passage = EvidencePassage(**row)
        records[passage.passage_id] = passage
    return records


def materialise_citations(
    text: str,
    citations: list[dict[str, Any]],
    corpus: dict[str, EvidencePassage],
) -> tuple[str, list[EvidencePassage]]:
    """Insert [1], [2], ... into cited sentences and resolve their sources."""
    if not citations:
        return text, []
    parts = sentence_parts(text)
    grouped: dict[int, list[str]] = {}
    sources: list[EvidencePassage] = []
    for number, citation in enumerate(citations, start=1):
        if not isinstance(citation, dict):
            raise ValueError("Each citation must be an object with source_id and sentence_index.")
        source_id = str(citation.get("source_id") or "")
        if source_id not in corpus:
            raise ValueError(f"Citation source_id is not in the corpus: {source_id!r}")
        try:
            sentence_index = int(citation["sentence_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Each citation requires an integer sentence_index.") from error
        if not 0 <= sentence_index < len(parts):
            raise ValueError(f"sentence_index {sentence_index} is outside this answer.")
        grouped.setdefault(sentence_index, []).append(f" [{number}]")
        sources.append(corpus[source_id])
    rendered = "".join(part.rstrip() + "".join(grouped.get(index, [])) + (" " if part.endswith(" ") else "")
                       for index, part in enumerate(parts))
    return rendered.strip(), sources


def score_row(
    row: dict[str, Any],
    corpus: dict[str, EvidencePassage],
    *,
    top_k: int,
) -> dict[str, Any]:
    answer, sources = materialise_citations(
        str(row["response_text"]), list(row.get("citations") or []), corpus
    )
    result = run_trustguard(
        question=str(row["question"]),
        answer=answer,
        response_id=(f"{row['pair_id']}__{row['version']}__"
                     f"{row.get('corruption_type') or 'clean'}"),
        top_k=top_k,
        citation_sources=sources or None,
        compute_stability=False,
    )
    citation = result.citation_quality or {}
    return {
        "pair_id": row["pair_id"],
        "version": row["version"],
        "corruption_type": row.get("corruption_type"),
        "question": row["question"],
        "response_text": row["response_text"],
        "corruption": row.get("corruption"),
        "citation_mode": "explicit" if sources else "retrieved_evidence",
        "evidence": result.evidence_support.get("evidence_support_score"),
        "source": result.source_reliability.get("rave_percent"),
        "citation": citation.get("citation_quality_percent"),
        "composite": result.trust_aggregation.get("composite_score") if result.trust_aggregation else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score TrustGuard corruption pairs.")
    parser.add_argument("input", type=Path, help="Output of generate_corruption_dataset.py")
    parser.add_argument("output", type=Path, help="Scored JSONL output")
    parser.add_argument("--corpus", type=Path, default=Path("cache/evidence_corpus.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    corpus = corpus_by_id(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            print(f"[{index}/{len(rows)}] {row['pair_id']} {row['version']} {row.get('corruption_type') or ''}")
            handle.write(json.dumps(score_row(row, corpus, top_k=args.top_k), ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
