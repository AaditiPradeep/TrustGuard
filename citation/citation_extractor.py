"""
TrustGuard — citation extraction and claim/citation mapping.

This module extracts citations from an LLM response and resolves them to the
actual evidence/source records supplied to the LLM. It does NOT decide whether
a citation supports or contradicts a claim; that is handled by
citation_quality.py.

Supported citation forms:
    [1], [2,3], [1-3]
    (1), (2)
    PMID: 12345678
    https://...
    http://...
    DOI: 10....
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from data.schema import Claim, EvidencePassage


@dataclass(frozen=True)
class Citation:
    citation_id: str
    raw_text: str
    kind: str
    reference: str
    source: Optional[EvidencePassage] = None

    @property
    def source_name(self) -> str:
        return self.source.source_name if self.source else ""

    @property
    def source_url(self) -> str:
        return self.source.url if self.source else self.reference

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.source is not None:
            result["source"] = self.source.to_dict()
        return result


@dataclass(frozen=True)
class ClaimCitation:
    """A claim linked to one citation and the original user question."""
    claim_id: str
    claim_text: str
    question: str
    citation_id: str
    citation_text: str
    source_name: str
    source_url: str
    source_identifier: str
    source_title: str
    source_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


_BRACKETED = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
_PAREN_NUMBER = re.compile(
    r"(?<![\w.])\((\d+(?:\s*[-,]\s*\d+)*)\)"
    r"(?!\s*(?:mg|g|kg|mL|%|years?|months?|days?))",
    re.I,
)
_URL = re.compile(r"https?://[^\s\]\[<>()\"']+")
_PMID = re.compile(r"\bPMID\s*[:#]?\s*(\d{6,9})\b", re.I)
_DOI = re.compile(r"\b(?:doi:\s*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.I)


def _expand_numbers(spec: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", spec):
        if "-" in part:
            left, right = [int(x.strip()) for x in part.split("-", 1)]
            if left <= right and right - left <= 50:
                numbers.extend(range(left, right + 1))
        else:
            numbers.append(int(part.strip()))
    return numbers


def extract_citations(response_text: str) -> list[Citation]:
    """Extract unique citations in order of appearance."""
    found: list[tuple[int, Citation]] = []

    for match in _BRACKETED.finditer(response_text):
        for number in _expand_numbers(match.group(1)):
            found.append(
                (match.start(), Citation(
                    citation_id=f"ref-{number}",
                    raw_text=match.group(0),
                    kind="numbered",
                    reference=str(number),
                ))
            )

    for match in _PAREN_NUMBER.finditer(response_text):
        for number in _expand_numbers(match.group(1)):
            found.append(
                (match.start(), Citation(
                    citation_id=f"ref-{number}",
                    raw_text=match.group(0),
                    kind="numbered",
                    reference=str(number),
                ))
            )

    for match in _URL.finditer(response_text):
        url = match.group(0).rstrip(".,;:")
        found.append(
            (match.start(), Citation(
                citation_id=f"url-{abs(hash(url))}",
                raw_text=match.group(0),
                kind="url",
                reference=url,
            ))
        )

    for match in _PMID.finditer(response_text):
        pmid = match.group(1)
        found.append(
            (match.start(), Citation(
                citation_id=f"pmid-{pmid}",
                raw_text=match.group(0),
                kind="pmid",
                reference=pmid,
            ))
        )

    for match in _DOI.finditer(response_text):
        doi = match.group(1)
        found.append(
            (match.start(), Citation(
                citation_id=f"doi-{doi.lower()}",
                raw_text=match.group(0),
                kind="doi",
                reference=doi,
            ))
        )

    found.sort(key=lambda item: item[0])

    unique: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for _, citation in found:
        key = (citation.kind, citation.reference.lower())
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


def _normalise(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"^https?://", "", value)
    return value.rstrip("/")


def resolve_citations(
    citations: Sequence[Citation],
    sources: Sequence[EvidencePassage],
) -> list[Citation]:
    """Resolve citation markers against the exact source list given to the LLM."""
    resolved: list[Citation] = []

    for citation in citations:
        source: Optional[EvidencePassage] = None

        if citation.kind == "numbered":
            index = int(citation.reference) - 1
            if 0 <= index < len(sources):
                source = sources[index]
        else:
            ref = _normalise(citation.reference)
            for candidate in sources:
                values = [
                    candidate.url,
                    candidate.identifier,
                    candidate.title,
                    candidate.source_name,
                ]
                if any(
                    ref == _normalise(v) or ref in _normalise(v)
                    for v in values if v
                ):
                    source = candidate
                    break

        resolved.append(
            Citation(
                citation_id=citation.citation_id,
                raw_text=citation.raw_text,
                kind=citation.kind,
                reference=citation.reference,
                source=source,
            )
        )

    return resolved


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (m.start(), m.end(), m.group(0).strip())
        for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text)
        if m.group(0).strip()
    ]


def map_claims_to_citations(
    response_text: str,
    claims: Sequence[Claim],
    citations: Sequence[Citation],
    question: str = "",
) -> list[ClaimCitation]:
    """
    Map each claim to citations in its sentence.

    `question` is retained on every pair because citation quality is evaluated
    in the context of the user's actual question, not just claim/source overlap.
    """
    spans = _sentence_spans(response_text)
    output: list[ClaimCitation] = []

    for claim in claims:
        claim_pos = response_text.lower().find(claim.text.lower())
        sentence = ""

        if claim_pos >= 0:
            for start, end, candidate in spans:
                if start <= claim_pos < end:
                    sentence = candidate
                    break

        if not sentence:
            claim_tokens = set(re.findall(
                r"\b[a-zA-Z0-9]{3,}\b", claim.text.lower()
            ))
            best_score = 0
            for _, _, candidate in spans:
                tokens = set(re.findall(
                    r"\b[a-zA-Z0-9]{3,}\b", candidate.lower()
                ))
                score = len(claim_tokens & tokens)
                if score > best_score:
                    best_score, sentence = score, candidate

        for citation in citations:
            if citation.raw_text not in sentence:
                continue

            source = citation.source
            output.append(
                ClaimCitation(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    question=question,
                    citation_id=citation.citation_id,
                    citation_text=citation.raw_text,
                    source_name=source.source_name if source else "",
                    source_url=source.url if source else citation.reference,
                    source_identifier=(
                        source.identifier if source else citation.reference
                    ),
                    source_title=source.title if source else "",
                    source_text=source.text if source else "",
                )
            )

    return output
