"""Run the normal TrustGuard pipeline and save candidate clean responses.

This does not invent citations. It saves the answer, its original scores, and
the retrieved evidence records that a reviewer can use to attach a correct
citation after checking the response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# When this file is run as ``python evaluation/collect_clean_responses.py``,
# Python places ``evaluation/`` rather than the repository root on sys.path.
# Add the root explicitly so the normal pipeline imports work from PowerShell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import run_trustguard


def read_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("question"):
                raise ValueError(f"Question missing on line {line_number}.")
            questions.append(row)
    return questions


def citation_candidates(result) -> list[dict[str, str]]:
    """Return deduplicated evidence identifiers for human citation review."""
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in result.evidence_support.get("claim_verification", []):
        for passage_id, passage in claim.get("evidence_passages", {}).items():
            if passage_id in seen:
                continue
            seen.add(passage_id)
            candidates.append({
                "source_id": passage_id,
                "source_name": str(passage.get("source_name", "")),
                "title": str(passage.get("title", "")),
                "url": str(passage.get("url", "")),
            })
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Create pipeline-scored clean-response candidates.")
    parser.add_argument("questions", type=Path, help="JSONL with id and question fields")
    parser.add_argument("output", type=Path, help="JSONL candidate clean-response output")
    parser.add_argument("--top-k", type=int, default=3, help="Evidence passages retrieved per claim")
    parser.add_argument("--stability-samples", type=int, default=3, help="Answer samples for Stability")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    args = parser.parse_args()

    items = read_questions(args.questions)
    if args.limit is not None:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for index, item in enumerate(items, start=1):
            item_id = str(item.get("id") or f"q{index:03d}")
            print(f"[{index}/{len(items)}] {item_id}")
            result = run_trustguard(
                question=str(item["question"]),
                response_id=item_id,
                top_k=args.top_k,
                stability_samples=args.stability_samples,
            )
            row = {
                "id": item_id,
                "question": result.question,
                "response_text": result.answer,
                # Fill this after checking the answer and the candidates.
                "citations": [],
                "citation_candidates": citation_candidates(result),
                "original_scores": {
                    "evidence": result.evidence_support.get("evidence_support_score"),
                    "source": result.source_reliability.get("rave_percent"),
                    "citation": result.citation_quality.get("citation_quality_percent") if result.citation_quality else None,
                    "composite": result.trust_aggregation.get("composite_score") if result.trust_aggregation else None,
                },
                "metadata": {
                    "review_status": "needs_human_review",
                    "top_k": args.top_k,
                    "stability_samples": args.stability_samples,
                },
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    print(f"Wrote {len(items)} candidate clean responses to {args.output}")
    print("Review every answer before using it as clean data, then add its citations.")


if __name__ == "__main__":
    main()
