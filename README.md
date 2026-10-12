# TrustGuard — Dataset + Evidence Corpus (Phase 1)

This is the foundation layer for the TrustGuard pipeline (Section 04/05 of
the review deck): a QA benchmark loader and a trusted-source evidence corpus
that later modules (Evidence Support, Source Reliability / RAVE, Citation
Quality) retrieve against.

## Layout

```
trustguard/
├── data/
│   ├── schema.py                 # EvidencePassage, QAItem, Claim + JSONL I/O
│   ├── source_registry.py        # trusted source tiers + Bi (baseline authority) for RAVE
│   ├── load_qa_dataset.py        # PubMedQA/MedMCQA loaders + offline seed QA set
│   ├── fetch_pubmed.py           # NCBI E-utilities -> PubMed abstracts
│   ├── fetch_reference_sources.py# MedlinePlus + openFDA fetchers
│   └── build_corpus.py           # orchestrates all fetchers -> one merged JSONL
├── retrieval/
│   ├── index.py                  # TF-IDF fallback retriever (offline)
│   ├── dense_index.py            # sentence-transformers + FAISS semantic retriever
│   └── build_dense_index.py      # chunks corpus passages and persists the FAISS index
├── tests/
│   ├── mock_corpus.py            # hand-written offline evidence for dev/testing
│   └── test_pipeline_smoke.py    # runs seed QA through retrieval, no network needed
└── requirements.txt
```

## Why two tiers of data

**QA benchmark** (`data/load_qa_dataset.py`)
- Tier 1 — real benchmarks: `load_pubmedqa()`, `load_medmcqa()` via HuggingFace
  `datasets`. PubMedQA is particularly useful early on because each item
  already comes with a supporting abstract, so you get claim+evidence pairs
  "for free" while you're still building claim extraction.
- Tier 2 — `load_seed_qa()`: five hand-written Med-HALT-style questions
  (including one adversarial "confidently wrong" trap: the insulin/OTC
  question) with zero dependencies. Use this to develop and unit-test every
  downstream module before you've wired up real datasets.

**Evidence corpus** (`data/fetch_*.py`, `data/build_corpus.py`)
Three source tiers matching `source_registry.py`'s baseline-authority
hierarchy. RAVE uses these tiers with DeBERTa's support or contradiction
confidence to score how strongly evidence supports the generated answer:

| Source | Tier | Bi | What it's good for |
|---|---|---|---|
| WHO, openFDA labels, CDC | regulatory | 0.95 | drug interactions, dosage, safety warnings — your highest-stakes claim category |
| MedlinePlus | reference | 0.85 | general consumer-health grounding |
| PubMed | primary_literature | 0.75 | broad claim coverage; independence is calculated per PMID |

RAVE uses explicit score bands. High-tier support scores `85–100`,
medium-tier support scores `75–90`, and lower-tier support scores `65–80`.
High-tier contradiction scores `0–15`, medium-tier contradiction scores
`10–25`, and lower-tier contradiction scores `20–35`. Evidence that is not
strong enough to support or contradict the claim scores `50` (inconclusive).
Peer agreement and the number of independent sources remain diagnostics; they
do not raise the score of evidence that contradicts the answer.

Baseline authority is a documented tier policy, not a learned probability.
`data.source_registry.authority_rubric()` exposes the weighted 1–5 authority
rubric behind each tier.

## Running it

**Fully offline (works right now, no setup):**
```bash
cd trustguard
python -m tests.test_pipeline_smoke
```
This loads the 5 seed QA items, builds a TF-IDF index over the 5 mock
evidence passages in `tests/mock_corpus.py`, and prints the top retrieved
passage per question — already tested, output below.

**With real data (needs internet — run outside this sandbox):**
```bash
pip install -r requirements.txt

# 1. Build a real evidence corpus for the topics/drugs you care about
python -m data.build_corpus \
  --topics "metformin lactic acidosis" "warfarin nsaid interaction" "penicillin cephalosporin cross reactivity" \
  --drugs metformin warfarin insulin \
  --email you@example.com \
  --out cache/evidence_corpus.jsonl

# 2. Load a real QA benchmark
python -c "from data.load_qa_dataset import load_pubmedqa; items = load_pubmedqa(limit=50); print(len(items))"

# 3. Query the real corpus the same way the smoke test does
python -c "
from retrieval.index import EvidenceIndex
idx = EvidenceIndex.from_jsonl('cache/evidence_corpus.jsonl')
for r in idx.query('is metformin safe with contrast dye', top_k=3):
    print(r.score, r.passage.source_name, r.passage.title)
"
```

### Build the dense evidence index

The corpus builder stores source passages and their provenance in JSONL. Build
the semantic retrieval layer immediately after it:

```bash
python -m retrieval.build_dense_index \
  --corpus cache/evidence_corpus.jsonl \
  --out cache/dense_evidence_index
```

This splits each source passage into overlapping 900-character chunks, embeds
them with `sentence-transformers/all-MiniLM-L6-v2`, normalizes the vectors, and
saves three linked artifacts: `evidence.faiss` (vectors), `passages.jsonl`
(chunk text plus source metadata), and `manifest.json` (model settings).
The first run downloads the embedding model. Query it using the same claim text
that will later be produced by claim extraction:

```bash
python -c "from retrieval.dense_index import DenseEvidenceIndex; idx = DenseEvidenceIndex.load('cache/dense_evidence_index'); [print(f'{r.score:.3f}', r.passage.source_name, r.passage.title) for r in idx.query('metformin and contrast dye safety', top_k=5)]"
```

## Notes / next steps for Phase 2

- `EvidenceIndex` remains the fully offline TF-IDF fallback. For the roadmap's
  dense retriever, use `DenseEvidenceIndex.query(claim_text, top_k)`. Its
  scores are cosine similarities after vector normalization, with a maximum of
  `1.0`; they express semantic retrieval relevance, not evidence support or
  source reliability.
- `Claim` in `schema.py` is a placeholder record for Phase 2's claim
  extraction output — it's defined now so the JSONL format is stable across
  phases.
- NCBI E-utilities: get a free API key to raise the rate limit from 3 to
  10 req/sec (`NCBI_API_KEY` env var); always pass a real `--email`.
- openFDA label pulls only return `warnings` / `drug_interactions` /
  `dosage_and_administration` / `boxed_warning` sections by design — these
  map directly to the highest-risk claim categories your risk table calls
  out (drug interaction / dosage errors).

## Integrated pipeline (Persons 1–3)

The current codebase now has a single orchestration entry point in `pipeline.py`:

```python
from pipeline import run_trustguard

result = run_trustguard(
    question=question,
    top_k=5,
)

print(result.to_dict())
```

The integrated flow is:

1. `verification.claim_extractor` — atomic claim extraction.
2. `retrieval.hybrid` — TF-IDF + dense FAISS retrieval with RRF and deduplication.
3. `verification.evidence_support` — local DeBERTa NLI passage-level support/contradiction judgement.
4. `metrics.evidence_support_score` — banded, relevance-weighted NLI Evidence Support scores: clear support `70-100`, mixed evidence `30-70`, and clear contradiction `0-30`.
5. `metrics.source_reliability` — banded RAVE scores that combine source authority with NLI support or contradiction.
6. `metrics.answer_stability` — a 70% MiniLM semantic cosine / 30% Jaccard hybrid score across three locally sampled Ollama responses (temperature 0.7).
7. `citation.pipeline` — citation extraction/resolution + MiniLM/NLI citation-quality scoring.
8. `trust_aggregator` — availability-aware weighted composite of the four existing metric outputs (40% evidence support, 25% source reliability, 20% answer consistency, 15% citation quality).
9. `decision_engine` and `trust_report` — accept/regenerate recommendation plus a plain-language, claim-by-claim explanation.

When `answer` is omitted, local Ollama runs `qwen3:1.7b` to generate a concise answer from `question` (up to 250 tokens). The same local model extracts claims. Local DeBERTa NLI judges Evidence Support, RAVE combines the authority tier and NLI relationship of those retrieved sources, and Citation Quality scores the same passages automatically. Answer Stability then samples three varied local answers at temperature 0.7 and scores their pairwise agreement. Supply `stability_responses` when you want to score a specific set of answers instead.

For an interactive run, use:

```bash
python run_integrated_test.py
```

Before the first run, install Ollama and download the model:

```bash
ollama pull qwen3:1.7b
```

The script asks for a question, generates an answer locally, and evaluates it. No `HF_TOKEN` or Hugging Face credits are needed.

### Integration notes

- `data.schema` is the canonical schema. `data.evidence_schema` is now only a backwards-compatible re-export so teammate code does not create a second incompatible `EvidencePassage` type.
- Hybrid RRF scores are ranking scores, not cosine similarities. Evidence Support normalizes the final fused retrieval score only to weight the NLI judgments relative to one another for a claim.
- RAVE combines source authority with whether retrieved evidence supports or contradicts the generated answer. Organisation-level independence is approximate because WHO, CDC, and MedlinePlus can rely on overlapping guidance.

Run the offline integration checks with:

```bash
pytest -q
```

The result dictionary also includes `trust_aggregation`, `decision`, and
`trust_report`. `trust_report` is intended for an end user: it spells out the
claims being checked, each claim's available scores, the overall score and
recommendation, and why TrustGuard reached that recommendation. Missing scores
are identified rather than treated as zero; the remaining metric weights are
redistributed for the composite.
