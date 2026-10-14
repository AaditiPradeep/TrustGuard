"""Attach manually reviewed citation candidates to clean evaluation responses.

The mappings deliberately cover only answers for which a retrieved passage
supports a specific sentence. Rows without a mapping remain uncited instead
of receiving topical but unsupported references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply reviewed citations to clean candidates.")
    parser.add_argument("input", type=Path, help="clean_candidates.jsonl")
    parser.add_argument("output", type=Path, help="clean_responses.jsonl")
    parser.add_argument(
        "--mappings",
        type=Path,
        default=Path("evaluation/curated_citation_mappings.json"),
        help="Reviewed citation mapping JSON",
    )
    args = parser.parse_args()

    mappings: dict[str, list[dict[str, int | str]]] = json.loads(
        args.mappings.read_text(encoding="utf-8")
    )
    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cited_rows = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            response_id = str(row["id"])
            candidates = {candidate["source_id"] for candidate in row.get("citation_candidates", [])}
            citations = mappings.get(response_id, [])
            unknown = [citation["source_id"] for citation in citations if citation["source_id"] not in candidates]
            if unknown:
                raise ValueError(f"{response_id} maps sources absent from its retrieved candidates: {unknown}")
            row["citations"] = citations
            row.setdefault("metadata", {})["review_status"] = (
                "citation_curated" if citations else "uncited_no_direct_candidate"
            )
            if citations:
                cited_rows += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {args.output}; {cited_rows} have reviewed citations.")
    print("Uncited rows were retained because their retrieved candidates did not directly support the answer.")


if __name__ == "__main__":
    main()
