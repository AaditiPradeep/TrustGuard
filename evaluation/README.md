# Corruption-evaluation dataset

This folder creates a reproducible paired dataset for evaluating whether TrustGuard scores fall when an answer or its citation is deliberately corrupted.

## 1. Create the clean set

Use [`questions_30.jsonl`](questions_30.jsonl) as the 30-question starting set. For two people, use [`questions_01_15.jsonl`](questions_01_15.jsonl) and [`questions_16_30.jsonl`](questions_16_30.jsonl). Generate an answer for each question through the normal pipeline, then review the answer and save a JSONL file with this shape:

```json
{"id":"q001","question":"...","response_text":"... [1]","citations":[{"marker":"[1]","source_id":"<evidence-passage-id>"}]}
```

`id`, `question`, and `response_text` are required. `citations` is required only for the citation-swap condition.

For a citation-swap experiment, each citation must also state which answer
sentence it supports:

```json
"citations": [{"source_id": "<evidence-passage-id>", "sentence_index": 0}]
```

Do not treat an unreviewed generated answer as clean. Keep only answers whose factual content and citation mapping you have reviewed. The target is 30 reviewed rows.

To generate the candidate clean rows and preserve their original pipeline scores, run:

```powershell
python evaluation/collect_clean_responses.py evaluation/questions_30.jsonl evaluation/clean_candidates.jsonl
```

For the split run:

```powershell
# Person 1
python evaluation/collect_clean_responses.py evaluation/questions_01_15.jsonl evaluation/clean_candidates_01_15.jsonl

# Person 2
python evaluation/collect_clean_responses.py evaluation/questions_16_30.jsonl evaluation/clean_candidates_16_30.jsonl
```

This calls the complete pipeline once per question. It saves `citation_candidates` from the retrieved evidence, but deliberately leaves `citations` empty: choose the source that actually supports each reviewed answer before adding a citation mapping.

## 2. Generate corruptions

```powershell
python evaluation/generate_corruption_dataset.py evaluation/clean_responses.jsonl evaluation/corruption_pairs.jsonl
```

The script creates one clean row plus up to three corrupted rows per response:

- `fact_change`: multiplies one dose, duration, or percentage by ten; when none is present, it swaps one known drug name.
- `negation`: reverses one simple affirmative or negative statement.
- `citation_swap`: replaces the first structured citation with one from another response.

It records the exact edit in `corruption`, uses a fixed random seed by default, and rejects text edits unless exactly one sentence changed. With 30 clean rows that contain an editable fact, a negatable statement, and at least one citation, the output is 120 rows: 30 clean and 90 corrupted.

## Citation-swap requirement

The default TrustGuard path automatically evaluates *retrieved* evidence as citations. It does not evaluate the citation markers inside an answer. For a valid citation-swap experiment, the scoring runner must pass the `citations` source mapping to `run_trustguard(..., citation_sources=...)` or call `citation.pipeline.evaluate_citations(...)` directly. Otherwise the pipeline will retrieve fresh relevant passages and the swap cannot lower Citation Quality.

The `source_id` should be an `EvidencePassage.passage_id` from the evidence corpus or from the clean pipeline result. Preserve the response marker (`[1]`) and substitute only its mapped source for the corrupted version.

## Validation before scoring

Spot-check 15 random clean/corrupted pairs. Confirm that the edited sentence is false or meaningfully mismatched, and that a citation donor comes from a different question. Record the number that pass review and exclude failed pairs before computing detection rates.

## Score and analyse the pairs

Run the generator after review, then score every clean/corrupted answer:

```powershell
python evaluation/generate_corruption_dataset.py evaluation/clean_responses.jsonl evaluation/corruption_pairs.jsonl
python evaluation/score_corruption_dataset.py evaluation/corruption_pairs.jsonl evaluation/scored_corruption_pairs.jsonl
python evaluation/analyze_corruption_results.py evaluation/scored_corruption_pairs.jsonl
```

The scorer deliberately excludes Answer Stability because answer text is fixed
in this experiment. The existing aggregate function redistributes Stability's
weight across the other available metrics. The analysis writes a paired summary
with detection rate, mean and median score drop, Cohen's *d*, and Wilcoxon
signed-rank *p*-value for the composite and each available metric.
### Attach the reviewed citation candidates

`clean_candidates.jsonl` keeps citations empty until a passage is checked
against the response. This repository includes manually reviewed mappings for
the rows with direct support; it intentionally leaves weak topical matches
uncited. Create the citation-aware clean dataset with:

```powershell
python evaluation/apply_curated_citations.py evaluation/clean_candidates.jsonl evaluation/clean_responses.jsonl
```

Use `clean_responses.jsonl`, rather than `clean_candidates.jsonl`, as input to
the corruption generator. Each citation has a `source_id` and `sentence_index`;
the scorer inserts the numbered markers when it evaluates the response.

### Calibrate the regeneration threshold

After scoring both clean and corrupted rows with the same configuration, sweep
candidate composite thresholds. The rule being tested is: regenerate when the
score is below the threshold.

```powershell
python evaluation/sweep_regeneration_thresholds.py evaluation/scored_corruption_pairs.jsonl evaluation/analysis/regeneration_threshold_sweep.csv
```

The script reports corruption detection, unnecessary clean-answer regeneration,
clean pass rate, balanced accuracy, and Youden's J for thresholds from 40 to 90.
It recommends the threshold with the largest Youden's J, preferring the lower
threshold if scores tie.

