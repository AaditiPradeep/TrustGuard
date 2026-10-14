"""Calculate paired corruption-detection statistics from scored JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wilcoxon


METRICS = ("composite", "evidence", "source", "citation")


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())


def paired_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    clean = frame[frame["version"] == "clean"].drop_duplicates("pair_id").set_index("pair_id")
    records = []
    for corruption_type, corrupted in frame[frame["version"] == "corrupted"].groupby("corruption_type"):
        merged = corrupted.set_index("pair_id")[[metric]].join(clean[[metric]], lsuffix="_corrupted", rsuffix="_clean", how="inner")
        differences = (merged[f"{metric}_clean"] - merged[f"{metric}_corrupted"]).dropna()
        differences = differences[pd.to_numeric(differences, errors="coerce").notna()].astype(float)
        if differences.empty:
            continue
        detection_rate = float(((differences > 0).sum() + 0.5 * (differences == 0).sum()) / len(differences))
        std = float(differences.std(ddof=1)) if len(differences) > 1 else math.nan
        effect_size = float(differences.mean() / std) if std and not math.isnan(std) else None
        try:
            p_value = float(wilcoxon(differences, zero_method="wilcox").pvalue) if (differences != 0).any() else 1.0
        except ValueError:
            p_value = None
        records.append({
            "metric": metric,
            "corruption_type": corruption_type,
            "n": len(differences),
            "detection_rate": round(detection_rate, 4),
            "mean_drop": round(float(differences.mean()), 4),
            "median_drop": round(float(differences.median()), 4),
            "effect_size_cohens_d": round(effect_size, 4) if effect_size is not None else None,
            "wilcoxon_p": round(p_value, 6) if p_value is not None else None,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse TrustGuard corruption scores.")
    parser.add_argument("input", type=Path, help="Scored JSONL from score_corruption_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/analysis"))
    args = parser.parse_args()

    frame = read_jsonl(args.input)
    required = {"pair_id", "version", "corruption_type", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    report = pd.concat([paired_summary(frame, metric) for metric in METRICS], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "paired_metric_summary.csv", index=False)
    (args.output_dir / "paired_metric_summary.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(report.to_string(index=False))
    print(f"\nWrote {args.output_dir / 'paired_metric_summary.csv'}")


if __name__ == "__main__":
    main()
