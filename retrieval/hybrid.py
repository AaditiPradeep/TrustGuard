"""
TrustGuard — hybrid (lexical + dense) evidence retrieval.

Combines the TF-IDF retriever (`retrieval.index.EvidenceIndex`) and the
FAISS semantic retriever (`retrieval.dense_index.DenseEvidenceIndex`) using
Reciprocal Rank Fusion (RRF), so a claim gets evidence that either retriever
alone tends to miss: TF-IDF is strong on exact drug names / numbers, dense
is strong on paraphrase and synonym matches.

Duplicate handling
-------------------
`DenseEvidenceIndex` is built over *chunks* of each source passage
(`retrieval.dense_index.chunk_passage`), and chunk identifiers look like
``"<original_identifier>:chunk:<n>"``. `EvidenceIndex` (TF-IDF), meanwhile,
is built over the *whole, unchunked* passages. That means the same
underlying source document can show up:

  - more than once in the dense results (two overlapping chunks of one
    long passage both looking relevant), and
  - under two different `passage_id`s across the two retrievers (whole
    passage vs. one of its chunks).

Both cases are "duplicate chunks" from the caller's point of view: nobody
downstream wants the same WHO fact sheet cited twice for one claim. This
module resolves both by collapsing every result — from either retriever —
back to a single *document key* (`source_name` + identifier with any
``:chunk:<n>`` suffix stripped), keeping only the best-scoring chunk/passage
per document before fusing ranks.

There's a second, unrelated source of duplication in the corpus itself:
openFDA carries a separate label filing per manufacturer, so a generic drug
(e.g. warfarin sodium) can appear under several distinct `identifier` UUIDs
whose section text (e.g. "7 DRUG INTERACTIONS ...") is copy-pasted
near-verbatim across all of them. Those aren't chunks of one document and
the ID-based dedup above can't (and structurally shouldn't try to) catch
them — from an ID standpoint they're legitimately different source records.
A second pass, `_content_key`, normalizes and hashes each passage's text and
collapses near-identical content across *different* documents too, keeping
whichever copy scored best.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from retrieval.dense_index import DEFAULT_MODEL, DenseEvidenceIndex
from retrieval.index import EvidenceIndex
from retrieval.types import ScoredPassage
from data.schema import EvidencePassage

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

DocKey = tuple[str, str]

# Standard RRF damping constant (Cormack et al., 2009). Higher = flatter
# fusion curve (top ranks matter less relative to lower ones).
DEFAULT_RRF_K = 60


@dataclass
class HybridResult:
    """A fused retrieval result. Duck-types `ScoredPassage` (`.passage`,
    `.score`) so it's a drop-in replacement anywhere a single-retriever
    result was expected, while also exposing the per-retriever detail.
    """

    passage: EvidencePassage
    score: float  # fused RRF score — NOT a similarity, just a fusion rank score
    tfidf_score: Optional[float] = None
    tfidf_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None

    @property
    def in_both(self) -> bool:
        return self.tfidf_rank is not None and self.dense_rank is not None

    def to_dict(self) -> dict:
        return {
            "passage": self.passage.to_dict(),
            "score": self.score,
            "tfidf_score": self.tfidf_score,
            "tfidf_rank": self.tfidf_rank,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "in_both": self.in_both,
        }


def _document_key(scored: ScoredPassage) -> DocKey:
    """Map a passage (or one chunk of it) back to its parent document."""
    passage = scored.passage
    base_identifier = passage.identifier.split(":chunk:")[0]
    return (passage.source_name, base_identifier)


def _dedupe_and_rank(results: list[ScoredPassage]) -> dict[DocKey, tuple[int, ScoredPassage]]:
    """Collapse a single retriever's raw results to one entry per document
    (keeping the best-scoring chunk/passage for that document), then
    re-rank 1..N over the deduped set.
    """
    best_per_doc: dict[DocKey, ScoredPassage] = {}
    for result in results:
        key = _document_key(result)
        current_best = best_per_doc.get(key)
        if current_best is None or result.score > current_best.score:
            best_per_doc[key] = result

    ordered = sorted(best_per_doc.items(), key=lambda kv: kv[1].score, reverse=True)
    return {key: (rank, result) for rank, (key, result) in enumerate(ordered, start=1)}


def _content_key(passage: EvidencePassage) -> str:
    """Exact fingerprint of a passage's text, for cheaply collapsing
    byte-for-byte (post-normalization) duplicates before the more expensive
    near-duplicate pass below. Lowercases and strips punctuation/whitespace
    differences, so purely cosmetic drift ("(5.1)" vs "( 5.1 )") still
    matches — but genuine wording differences (a hyphen that changes word
    boundaries, e.g. "non-steroidal" vs "nonsteroidal") will NOT match here,
    which is exactly why `_dedupe_by_content` follows up with a fuzzy pass.
    """
    normalized = _NON_ALNUM_RE.sub(" ", passage.text.lower())
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _word_shingles(text: str, n: int = 2) -> set[str]:
    """Word n-gram shingles of normalized text, for near-duplicate
    detection that's robust to a handful of scattered wording differences
    (typos, a stray hyphen, re-flowed punctuation) between two copies of
    what is otherwise the same document.
    """
    normalized = _NON_ALNUM_RE.sub(" ", text.lower())
    words = normalized.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Two passages sharing 90%+ of their word-bigrams are treated as the same
# underlying document (e.g. the same FDA label section re-filed by a
# different manufacturer with trivial wording drift), not two distinct
# pieces of evidence.
NEAR_DUPLICATE_THRESHOLD = 0.9


def _dedupe_by_content(results: list["HybridResult"]) -> list["HybridResult"]:
    """Second dedup pass over already ID-deduped, fused results: collapse
    near-identical passage text across *different* document identifiers
    (e.g. several manufacturers' filings of the same generic drug's label),
    keeping whichever copy has the higher fused score.

    Two-stage for efficiency: an exact-hash pass first removes byte-for-byte
    (post-normalization) duplicates cheaply, then a fuzzy word-shingle
    Jaccard pass — the part that actually matters here, since real-world
    duplicate filings are rarely 100% byte-identical — catches the rest.
    The fuzzy pass is O(n^2) but n is the already-deduped candidate pool
    (tens of items), not the whole corpus, so this is cheap in practice.
    """
    exact_deduped: dict[str, "HybridResult"] = {}
    for result in results:
        key = _content_key(result.passage)
        current = exact_deduped.get(key)
        if current is None or result.score > current.score:
            exact_deduped[key] = result

    candidates = sorted(exact_deduped.values(), key=lambda r: r.score, reverse=True)

    kept: list["HybridResult"] = []
    kept_shingles: list[set[str]] = []
    for result in candidates:
        shingles = _word_shingles(result.passage.text)
        if any(_jaccard(shingles, ks) >= NEAR_DUPLICATE_THRESHOLD for ks in kept_shingles):
            continue  # near-duplicate of something already kept, which scored >= this one
        kept.append(result)
        kept_shingles.append(shingles)
    return kept


class HybridRetriever:
    """Fuses TF-IDF and dense (FAISS) retrieval with Reciprocal Rank Fusion.

    The dense index is optional: if it isn't built yet (or faiss/
    sentence-transformers aren't installed), the retriever transparently
    falls back to TF-IDF-only ranking, still deduped to one hit per
    document, so downstream code (Evidence Support, Citation Quality) can
    depend on `HybridRetriever` unconditionally.
    """

    def __init__(
        self,
        tfidf_index: EvidenceIndex,
        dense_index: Optional[DenseEvidenceIndex] = None,
        rrf_k: int = DEFAULT_RRF_K,
        tfidf_pool_size: int = 20,
        dense_pool_size: int = 20,
    ):
        self.tfidf_index = tfidf_index
        self.dense_index = dense_index
        self.rrf_k = rrf_k
        self.tfidf_pool_size = tfidf_pool_size
        self.dense_pool_size = dense_pool_size

    @classmethod
    def from_jsonl(
        cls,
        *paths: str,
        dense_index_dir: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> "HybridRetriever":
        """Build (or load) both underlying indexes and wrap them.

        `dense_index_dir`, if given, must point at a directory already
        produced by `retrieval.build_dense_index` (i.e. contains
        evidence.faiss / passages.jsonl / manifest.json). If omitted, the
        retriever runs TF-IDF-only until you build/attach a dense index.
        """
        tfidf_index = EvidenceIndex.from_jsonl(*paths)
        dense_index = None
        if dense_index_dir is not None:
            dense_index = DenseEvidenceIndex.load(dense_index_dir)
        return cls(tfidf_index, dense_index, rrf_k=rrf_k)

    def query(
        self,
        claim_text: str,
        top_k: int = 5,
        require_both: bool = False,
    ) -> list[HybridResult]:
        """Fuse TF-IDF and dense results for `claim_text`.

        `require_both`, if True, drops any result that only one retriever
        found before ranking, keeping just the documents both TF-IDF and
        dense agreed on. Use this when a caller (e.g. Evidence Support)
        wants higher-precision evidence and would rather get fewer, more
        trustworthy candidates than pad out to `top_k` with single-retriever
        guesses. If dual agreement yields fewer than `top_k` results (or no
        dense index is attached at all), it's better to fall back to the
        full ranked list than return an artificially short one, so
        `require_both` only applies when it doesn't leave the caller with
        nothing.
        """
        tfidf_raw = self.tfidf_index.query(claim_text, top_k=self.tfidf_pool_size)
        tfidf_ranked = _dedupe_and_rank(tfidf_raw)

        dense_ranked: dict[DocKey, tuple[int, ScoredPassage]] = {}
        if self.dense_index is not None:
            dense_raw = self.dense_index.query(claim_text, top_k=self.dense_pool_size)
            dense_ranked = _dedupe_and_rank(dense_raw)

        fused: list[HybridResult] = []
        for key in set(tfidf_ranked) | set(dense_ranked):
            fused_score = 0.0
            tfidf_rank = tfidf_score = dense_rank = dense_score = None
            passage: Optional[EvidencePassage] = None

            if key in tfidf_ranked:
                rank, scored = tfidf_ranked[key]
                fused_score += 1.0 / (self.rrf_k + rank)
                tfidf_rank, tfidf_score = rank, scored.score
                passage = scored.passage

            if key in dense_ranked:
                rank, scored = dense_ranked[key]
                fused_score += 1.0 / (self.rrf_k + rank)
                dense_rank, dense_score = rank, scored.score
                passage = passage or scored.passage

            fused.append(
                HybridResult(
                    passage=passage,
                    score=fused_score,
                    tfidf_score=tfidf_score,
                    tfidf_rank=tfidf_rank,
                    dense_score=dense_score,
                    dense_rank=dense_rank,
                )
            )

        fused = _dedupe_by_content(fused)
        fused.sort(key=lambda r: r.score, reverse=True)

        if require_both:
            dual_hits = [r for r in fused if r.in_both]
            if dual_hits:
                return dual_hits[:top_k]
            # No dual-confirmed hits at all (e.g. dense index not attached,
            # or the two retrievers genuinely found no overlap) — fall back
            # rather than hand back an empty list.

        return fused[:top_k]

    def __len__(self) -> int:
        return len(self.tfidf_index)
