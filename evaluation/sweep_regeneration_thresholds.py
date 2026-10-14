"""Calibrate a score threshold for TrustGuard answer regeneration.

This script needs a *fully controlled* scored dataset: each clean answer and
each of its corrupted versions must have been scored with the same pipeline
configuration. It tests rules of the form ``regenerate when score < threshold``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def thresholds(start: int, stop: int, step: int) -> list[int]:
    if step <= 0 or start > stop:
        raise ValueError("Require a positive step and start <= stop.")
    return list(range(start, stop + 1, step))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep TrustGuard regeneration thresholds.")
    parser.add_argument("input", type=Path, help="Fully scored clean/corrupted JSONL")
    parser.add_argument("output", type=Path, help="CSV summary of threshold results")
    parser.add_argument("--metric", default="composite", choices=["composite", "evidence", "source", "citation"])
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--stop", type=int, default=90)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    clean = [row for row in rows if row.get("version") == "clean"]
    corrupted = [row for row in rows if row.get("version") == "corrupted"]
    if not clean or not corrupted:
        raise ValueError("Input needs both clean and corrupted scored rows.")

    clean_scores = [float(row[args.metric]) for row in clean if row.get(args.metric) is not None]
    corrupted_scores = [float(row[args.metric]) for row in corrupted if row.get(args.metric) is not None]
    if not clean_scores or not corrupted_scores:
        raise ValueError(f"No usable {args.metric!r} scores found.")

    results: list[dict[str, float | int]] = []
    for threshold in thresholds(args.start, args.stop, args.step):
        # True positive: a deliberately corrupted answer correctly sent back
        # for regeneration. False positive: a clean answer unnecessarily sent
        # back for regeneration.
        true_positive_rate = mean(score < threshold for score in corrupted_scores)
        false_positive_rate = mean(score < threshold for score in clean_scores)
        true_negative_rate = 1 - false_positive_rate
        youden_j = true_positive_rate + true_negative_rate - 1
        results.append({
            "threshold": threshold,
            "corruption_detection_rate": round(true_positive_rate * 100, 2),
            "unnecessary_regeneration_rate": round(false_positive_rate * 100, 2),
            "clean_pass_rate": round(true_negative_rate * 100, 2),
            "balanced_accuracy": round((true_positive_rate + true_negative_rate) * 50, 2),
            "youden_j": round(youden_j, 4),
        })

    # On ties, prefer the lower threshold: it achieves the same separation
    # while asking the system to regenerate fewer otherwise acceptable answers.
    recommended = max(results, key=lambda row: (float(row["youden_j"]), -int(row["threshold"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Metric: {args.metric}")
    print(f"Clean answers: {len(clean_scores)}; corrupted answers: {len(corrupted_scores)}")
    print(f"Mean clean score: {mean(clean_scores):.2f}; mean corrupted score: {mean(corrupted_scores):.2f}")
    print("Recommended threshold (maximum Youden's J):")
    print(json.dumps(recommended, indent=2))
    print(f"Wrote threshold sweep to {args.output}")


if __name__ == "__main__":
    main()
