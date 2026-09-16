"""
TrustGuard — core data schema.

Everything downstream (Evidence Support, Source Reliability / RAVE,
Citation Quality) consumes these three record types, so they're defined
once here and imported everywhere else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str) -> str:
    """Deterministic short id so re-running a fetch doesn't duplicate rows."""
    h = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


@dataclass
class EvidencePassage:
    """A single retrievable unit of trusted medical evidence."""

    text: str
    source_name: str          # e.g. "PubMed", "MedlinePlus", "WHO", "openFDA"
    source_tier: str          # "primary_literature" | "clinical_guideline" | "reference" | "regulatory"
    title: str = ""
    url: str = ""
    identifier: str = ""      # e.g. PMID, MedlinePlus topic id, WHO fact-sheet slug
    published_date: str = ""  # source's own publication/last-review date if known
    retrieved_at: str = field(default_factory=_now_iso)
    passage_id: str = ""

    def __post_init__(self):
        if not self.passage_id:
            self.passage_id = _stable_id(self.source_name, self.identifier, self.text[:200])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QAItem:
    """One question/answer(-context) item used to drive claim extraction + verification."""

    question: str
    answer: str = ""                # reference / gold answer, if any (may be blank for open QA)
    context: str = ""                # supporting abstract/snippet, if the source dataset provides one
    dataset: str = ""                # e.g. "pubmedqa", "medmcqa", "seed_medhalt_style"
    item_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.item_id:
            self.item_id = _stable_id(self.dataset, self.question[:200])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Claim:
    """An atomic factual claim extracted from an LLM response (populated in Phase 2)."""

    text: str
    source_response_id: str
    claim_id: str = ""
    evidence_support: Optional[str] = None   # "supported" | "contradicted" | "irrelevant"
    citation_url: str = ""

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = _stable_id(self.source_response_id, self.text[:200])

    def to_dict(self) -> dict:
        return asdict(self)


def write_jsonl(records: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
