"""Analyse separately scored corruption sets against saved base-response scores.

The original base scores contain Answer Stability, whereas the fixed-answer
corruption scorer disables Stability. For a comparable preliminary composite,
this script recomputes every base composite from its saved Evidence Support,
RAVE, and Citation Quality values using the scorer's re-normalised weights:

    0.50 * evidence + 0.3125 * source + 0.1875 * citation

It always pairs a corruption with the base response having the same question
ID. Thus, a 10-row corruption file is compared with its corresponding 10 of
the 30 base responses, not with the full base-set average.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any


METRICS = ("evidence", "source", "citation", "composite")
BASE_COMPOSITE_WEIGHTS = {
    "evidence": 0.50,
    "source": 0.3125,
    "citation": 0.1875,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def comparable_base_scores(row: dict[str, Any]) -> dict[str, float] | None:
    scores = row.get("original_scores") or {}
    try:
        values = {metric: float(scores[metric]) for metric in BASE_COMPOSITE_WEIGHTS}
    except (KeyError, TypeError, ValueError):
        return None

    values["composite"] = sum(
        BASE_COMPOSITE_WEIGHTS[metric] * values[metric]
        for metric in BASE_COMPOSITE_WEIGHTS
    )
    return values


def paired_rows(
    base_by_id: dict[str, dict[str, Any]],
    corruptions: list[dict[str, Any]],
    corruption_type: str,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for corrupted in corruptions:
        pair_id = str(corrupted.get("pair_id") or "")
        if corrupted.get("corruption_type") != corruption_type:
            continue
        base = base_by_id.get(pair_id)
        if base is None:
            raise ValueError(f"No saved base response found for {pair_id!r}.")
        base_scores = comparable_base_scores(base)
        if base_scores is None:
            raise ValueError(f"Saved base response {pair_id!r} has incomplete scores.")

        record: dict[str, Any] = {
            "pair_id": pair_id,
            "corruption_type": corruption_type,
            "question": corrupted.get("question", base.get("question", "")),
            "rule": (corrupted.get("corruption") or {}).get("rule", ""),
        }
        for metric in METRICS:
            corrupted_value = corrupted.get(metric)
            if corrupted_value is None:
                raise ValueError(f"{pair_id!r} has no corrupted {metric!r} score.")
            base_value = base_scores[metric]
            record[f"base_{metric}"] = base_value
            record[f"corrupted_{metric}"] = float(corrupted_value)
            record[f"drop_{metric}"] = base_value - float(corrupted_value)
        pairs.append(record)
    return pairs


def wilcoxon_pvalue(drops: list[float]) -> float | None:
    """Return a two-sided paired Wilcoxon p-value when SciPy is available."""
    if not drops or all(drop == 0 for drop in drops):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(drops, zero_method="wilcox").pvalue)
    except (ImportError, ValueError):
        return None


def summarise(pairs: list[dict[str, Any]], corruption_type: str) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError(f"No usable {corruption_type!r} rows found.")

    summary: list[dict[str, Any]] = []
    for metric in METRICS:
        base = [float(row[f"base_{metric}"]) for row in pairs]
        corrupted = [float(row[f"corrupted_{metric}"]) for row in pairs]
        drops = [float(row[f"drop_{metric}"]) for row in pairs]
        detection = sum(1 if drop > 0 else 0.5 if drop == 0 else 0 for drop in drops) / len(drops)
        spread = stdev(drops) if len(drops) > 1 else 0.0
        effect_size = mean(drops) / spread if spread > 0 else None

        summary.append({
            "corruption_type": corruption_type,
            "metric": metric,
            "n_pairs": len(pairs),
            "base_mean": round(mean(base), 2),
            "corrupted_mean": round(mean(corrupted), 2),
            "mean_drop": round(mean(drops), 2),
            "median_drop": round(median(drops), 2),
            "detection_rate_pct": round(detection * 100, 2),
            "effect_size": round(effect_size, 3) if effect_size is not None and math.isfinite(effect_size) else None,
            "wilcoxon_p_value": round(wilcoxon_pvalue(drops), 6) if wilcoxon_pvalue(drops) is not None else None,
        })
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved base scores with separately scored corruptions.")
    parser.add_argument("--base", type=Path, required=True, help="clean_responses.jsonl")
    parser.add_argument("--fact", type=Path, required=True, help="scored fact-change JSONL")
    parser.add_argument("--negation", type=Path, required=True, help="scored negation JSONL")
    parser.add_argument("--citation", type=Path, required=True, help="scored citation-swap JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/analysis"))
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    base_by_id = {str(row["id"]): row for row in base_rows}

    inputs = {
        "fact_change": read_jsonl(args.fact),
        "negation": read_jsonl(args.negation),
        "citation_swap": read_jsonl(args.citation),
    }

    all_pairs: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    for corruption_type, rows in inputs.items():
        pairs = paired_rows(base_by_id, rows, corruption_type)
        all_pairs.extend(pairs)
        all_summary.extend(summarise(pairs, corruption_type))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "partial_corruption_pair_results.csv", all_pairs)
    write_csv(args.output_dir / "partial_corruption_metric_summary.csv", all_summary)

    metadata = {
        "analysis_type": "preliminary paired corruption analysis",
        "base_response_count": len(base_rows),
        "notes": [
            "Each corruption is paired to the base response with the same ID.",
            "Base composite scores are reweighted without Answer Stability to match fixed-answer corruption scoring.",
            "Base citation scores were logged before explicit citation mappings, so citation and composite comparisons are preliminary.",
        ],
        "summary": all_summary,
    }
    with (args.output_dir / "partial_corruption_metric_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved:")
    print(args.output_dir / "partial_corruption_metric_summary.csv")
    print(args.output_dir / "partial_corruption_pair_results.csv")
    print(args.output_dir / "partial_corruption_metric_summary.json")
    print("\nSummary:")
    for row in all_summary:
        print(
            f"{row['corruption_type']:14} {row['metric']:10} "
            f"n={row['n_pairs']:2} mean_drop={row['mean_drop']:6.2f} "
            f"detection={row['detection_rate_pct']:6.2f}%"
        )


if __name__ == "__main__":
    main()
