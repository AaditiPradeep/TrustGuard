"""Shared return types for lexical and dense evidence retrieval."""

from dataclasses import dataclass

from data.schema import EvidencePassage


@dataclass
class ScoredPassage:
    passage: EvidencePassage
    score: float
