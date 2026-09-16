"""
TrustGuard — evidence retrieval index.

Phase-2 default is a TF-IDF + cosine-similarity retriever: zero extra
dependencies (sklearn is enough), fully offline, and good enough to
validate the rest of the pipeline end-to-end. Swap in a dense retriever
(e.g. sentence-transformers + FAISS) later without changing the interface —
`EvidenceIndex.query()` is what the Evidence Support module calls either way.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.schema import EvidencePassage, read_jsonl
from retrieval.types import ScoredPassage


class EvidenceIndex:
    """In-memory TF-IDF index over a list of EvidencePassage records."""

    def __init__(self, passages: list[EvidencePassage]):
        if not passages:
            raise ValueError("EvidenceIndex needs at least one passage.")
        self.passages = passages
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
        )
        self._matrix = self._vectorizer.fit_transform([p.text for p in passages])

    @classmethod
    def from_jsonl(cls, *paths: str) -> "EvidenceIndex":
        """Build an index from one or more evidence JSONL files (e.g. one per source)."""
        passages: list[EvidencePassage] = []
        for path in paths:
            for row in read_jsonl(path):
                passages.append(EvidencePassage(**row))
        return cls(passages)

    def query(self, claim_text: str, top_k: int = 3) -> list[ScoredPassage]:
        q_vec = self._vectorizer.transform([claim_text])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            ScoredPassage(passage=self.passages[i], score=float(sims[i]))
            for i in ranked_idx
            if sims[i] > 0.0
        ]

    def __len__(self) -> int:
        return len(self.passages)
