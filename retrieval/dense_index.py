"""Dense, persistent semantic retrieval for TrustGuard evidence passages.

Each source passage is split into overlapping chunks, encoded with the same
sentence-transformer used for claims, and stored in a normalized FAISS inner
product index.  With normalized vectors, the returned score is cosine
similarity (maximum 1.0).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from data.schema import EvidencePassage, read_jsonl, write_jsonl
from retrieval.types import ScoredPassage

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_passage(
    passage: EvidencePassage,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[EvidencePassage]:
    """Split one passage on word boundaries while retaining its provenance."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be smaller.")

    words = passage.text.split()
    if not words:
        return []

    chunks: list[EvidencePassage] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        additional = len(word) + (1 if current else 0)
        if current and current_length + additional > chunk_size:
            chunk_number = len(chunks)
            chunks.append(
                replace(
                    passage,
                    text=" ".join(current),
                    identifier=f"{passage.identifier}:chunk:{chunk_number}",
                    passage_id="",
                )
            )
            overlap: list[str] = []
            overlap_length = 0
            for previous in reversed(current):
                extra = len(previous) + (1 if overlap else 0)
                if overlap_length + extra > chunk_overlap:
                    break
                overlap.insert(0, previous)
                overlap_length += extra
            current = overlap
            current_length = len(" ".join(current))
        current.append(word)
        current_length += len(word) + (1 if len(current) > 1 else 0)

    if current:
        chunk_number = len(chunks)
        chunks.append(
            replace(
                passage,
                text=" ".join(current),
                identifier=f"{passage.identifier}:chunk:{chunk_number}",
                passage_id="",
            )
        )
    return chunks


def chunk_passages(
    passages: Iterable[EvidencePassage],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[EvidencePassage]:
    return [
        chunk
        for passage in passages
        for chunk in chunk_passage(passage, chunk_size, chunk_overlap)
    ]


class DenseEvidenceIndex:
    """Sentence-transformer embeddings plus an exact FAISS cosine index."""

    def __init__(
        self,
        passages: list[EvidencePassage],
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
    ):
        if not passages:
            raise ValueError("DenseEvidenceIndex needs at least one passage.")
        self.passages = passages
        self.model_name = model_name
        self._model, faiss = self._load_dependencies(model_name)
        embeddings = self._encode([passage.text for passage in passages], batch_size)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    @staticmethod
    def _load_dependencies(model_name: str):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "Dense retrieval requires sentence-transformers and faiss-cpu. "
                "Install them with: pip install -r requirements.txt"
            ) from error
        return SentenceTransformer(model_name, local_files_only=True), faiss

    def _encode(self, texts: list[str], batch_size: int):
        import numpy as np

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.ascontiguousarray(embeddings.astype("float32"))

    @classmethod
    def from_jsonl(
        cls,
        *paths: str,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        batch_size: int = 32,
    ) -> "DenseEvidenceIndex":
        passages = [
            EvidencePassage(**row)
            for path in paths
            for row in read_jsonl(path)
        ]
        return cls(
            chunk_passages(passages, chunk_size, chunk_overlap),
            model_name=model_name,
            batch_size=batch_size,
        )

    def query(self, claim_text: str, top_k: int = 5) -> list[ScoredPassage]:
        if top_k <= 0:
            return []
        query_vector = self._encode([claim_text], batch_size=1)
        scores, indexes = self._index.search(query_vector, min(top_k, len(self.passages)))
        return [
            ScoredPassage(self.passages[index], float(score))
            for score, index in zip(scores[0], indexes[0])
            if index >= 0
        ]

    def save(self, output_dir: str) -> None:
        """Save the FAISS vectors, chunk metadata, and model settings together."""
        _, faiss = self._load_dependencies(self.model_name)
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(target / "evidence.faiss"))
        write_jsonl(self.passages, str(target / "passages.jsonl"))
        (target / "manifest.json").write_text(
            json.dumps({"model_name": self.model_name, "vector_count": len(self)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: str) -> "DenseEvidenceIndex":
        """Open a saved index without re-encoding the evidence corpus."""
        target = Path(output_dir)
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        passages = [EvidencePassage(**row) for row in read_jsonl(str(target / "passages.jsonl"))]
        if not passages:
            raise ValueError("Saved dense index contains no passages.")
        instance = cls.__new__(cls)
        instance.passages = passages
        instance.model_name = manifest["model_name"]
        instance._model, faiss = instance._load_dependencies(instance.model_name)
        instance._index = faiss.read_index(str(target / "evidence.faiss"))
        if instance._index.ntotal != len(passages):
            raise ValueError("Saved FAISS vector count does not match passage metadata.")
        return instance

    def __len__(self) -> int:
        return len(self.passages)
