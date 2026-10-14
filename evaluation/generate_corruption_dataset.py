"""Create clean/corrupted response pairs for TrustGuard evaluation.

The input is JSONL. Each clean row requires ``id``, ``question``, and either
``response_text`` or ``answer``. ``citations`` is optional, but citation-swap
examples can only be made from rows that have it.

Example input row::

    {"id": "q001", "question": "...", "response_text": "... [1]",
     "citations": [{"source_id": "passage-a", "marker": "[1]"}]}

The output contains one clean row and up to three corrupted rows per input
row. It never calls an LLM: fact and negation edits are deliberately small,
reproducible rule-based edits. Every textual corruption is checked to ensure
that exactly one sentence changed. Citation swaps modify only the structured
``citations`` field, which is the behaviour the citation evaluator needs.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SENTENCE_RE = re.compile(r".*?(?:[.!?](?:\s+|$)|$)", re.DOTALL)
NUMBER_WITH_UNIT_RE = re.compile(
    r"\b(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|mL|%|"
    r"hours?|days?|weeks?|months?|years?)\b"
)

# These are only fallbacks when a response contains no dose, duration, or
# percentage. They are intentionally common medicines so the changed answer is
# very likely to become medically wrong and should lose evidence support.
DRUG_SWAPS = {
    "acetaminophen": "warfarin",
    "amoxicillin": "metformin",
    "aspirin": "insulin",
    "ibuprofen": "warfarin",
    "insulin": "ibuprofen",
    "metformin": "warfarin",
    "warfarin": "metformin",
}

# A small set of answer-independent factual substitutions used only when an
# answer contains neither a measurable number nor a medicine name. Each pair
# reverses a concrete medical fact without rewriting the surrounding sentence.
# Keeping this list short makes the generated corruption easy to audit.
FACTUAL_REVERSALS = {
    "fever": "hypothermia",
    "rapid heartbeat": "slow heartbeat",
    "too great": "too low",
}


@dataclass(frozen=True)
class Edit:
    sentence_index: int
    before: str
    after: str
    rule: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} of {path} is not an object.")
                rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sentence_parts(text: str) -> list[str]:
    return [match.group(0) for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]


def replace_sentence(parts: list[str], index: int, replacement: str) -> str:
    changed = list(parts)
    changed[index] = replacement
    return "".join(changed)


def fact_edit(sentence: str) -> tuple[str, str] | None:
    """Change a number, medicine, or concrete medical fact in one sentence."""
    match = NUMBER_WITH_UNIT_RE.search(sentence)
    if match:
        old = float(match.group("number"))
        new = old * 10 if old else 10.0
        rendered = str(int(new)) if new.is_integer() else f"{new:g}"
        changed = sentence[:match.start("number")] + rendered + sentence[match.end("number"):]
        return changed, "number_times_ten"

    for drug, replacement in DRUG_SWAPS.items():
        match = re.search(rf"\b{re.escape(drug)}\b", sentence, flags=re.IGNORECASE)
        if match:
            if match.group(0)[0].isupper():
                replacement = replacement.capitalize()
            changed = sentence[:match.start()] + replacement + sentence[match.end():]
            return changed, f"drug_swap_{drug}_to_{replacement.lower()}"

    for original, replacement in FACTUAL_REVERSALS.items():
        match = re.search(rf"\b{re.escape(original)}\b", sentence, flags=re.IGNORECASE)
        if match:
            if match.group(0)[0].isupper():
                replacement = replacement.capitalize()
            changed = sentence[:match.start()] + replacement + sentence[match.end():]
            return changed, f"factual_reversal_{original.replace(' ', '_')}_to_{replacement.lower().replace(' ', '_')}"
    return None


NEGATION_RULES: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str], str]] = [
    (re.compile(r"\bdoes not\b", re.I), lambda m: "does", "remove_does_not"),
    (re.compile(r"\bdo not\b", re.I), lambda m: "do", "remove_do_not"),
    (re.compile(r"\bis not\b", re.I), lambda m: "is", "remove_is_not"),
    (re.compile(r"\bare not\b", re.I), lambda m: "are", "remove_are_not"),
    (re.compile(r"\bcannot\b", re.I), lambda m: "can", "remove_cannot"),
    (re.compile(r"\bcan not\b", re.I), lambda m: "can", "remove_can_not"),
    (re.compile(r"\bincreases?\b", re.I), lambda m: "decreases" if m.group(0).islower() else "Decreases", "reverse_increase"),
    (re.compile(r"\bdecreases?\b", re.I), lambda m: "increases" if m.group(0).islower() else "Increases", "reverse_decrease"),
    (re.compile(r"\bcan\s+(?=[a-z])", re.I), lambda m: "cannot ", "add_cannot"),
    (re.compile(r"\bshould\s+(?=[a-z])", re.I), lambda m: "should not ", "add_should_not"),
    (re.compile(r"\bmay\s+(?=[a-z])", re.I), lambda m: "may not ", "add_may_not"),
    (re.compile(r"\bis\s+(?=[a-z])", re.I), lambda m: "is not ", "add_is_not"),
]


def negation_edit(sentence: str) -> tuple[str, str] | None:
    """Reverse a simple affirmative/negative statement without rewriting it."""
    for pattern, replacement, rule in NEGATION_RULES:
        match = pattern.search(sentence)
        if match:
            changed = sentence[:match.start()] + replacement(match) + sentence[match.end():]
            if changed != sentence:
                return changed, rule
    return None


def find_edit(text: str, editor: Callable[[str], tuple[str, str] | None]) -> Edit | None:
    for index, sentence in enumerate(sentence_parts(text)):
        result = editor(sentence)
        if result:
            after, rule = result
            return Edit(index, sentence, after, rule)
    return None


def validate_one_sentence_change(before: str, after: str) -> tuple[bool, str]:
    before_parts, after_parts = sentence_parts(before), sentence_parts(after)
    if len(before_parts) != len(after_parts):
        return False, "sentence_count_changed"
    changed = [i for i, (left, right) in enumerate(zip(before_parts, after_parts)) if left != right]
    if len(changed) != 1:
        return False, f"expected_one_changed_sentence_found_{len(changed)}"
    diff = "".join(difflib.unified_diff(before_parts, after_parts, lineterm=""))
    return bool(diff), "ok" if diff else "no_text_difference"


def normalise_clean(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    response = str(row.get("response_text") or row.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    if not question or not response:
        raise ValueError("Every input row needs non-empty question and response_text (or answer).")
    return {
        "pair_id": str(row.get("id") or row.get("pair_id") or f"clean_{ordinal:03d}"),
        "question": question,
        "response_text": response,
        "citations": copy.deepcopy(row.get("citations") or []),
        "metadata": copy.deepcopy(row.get("metadata") or {}),
    }


def text_corruption(clean: dict[str, Any], kind: str) -> dict[str, Any] | None:
    editor = fact_edit if kind == "fact_change" else negation_edit
    edit = find_edit(clean["response_text"], editor)
    if edit is None:
        return None
    parts = sentence_parts(clean["response_text"])
    response = replace_sentence(parts, edit.sentence_index, edit.after)
    valid, validation = validate_one_sentence_change(clean["response_text"], response)
    if not valid:
        return None
    return {
        **copy.deepcopy(clean),
        "version": "corrupted",
        "corruption_type": kind,
        "response_text": response,
        "corruption": {
            "rule": edit.rule,
            "sentence_index": edit.sentence_index,
            "original_sentence": edit.before.strip(),
            "corrupted_sentence": edit.after.strip(),
            "validation": validation,
        },
    }


def citation_swap(clean: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any] | None:
    citations = clean["citations"]
    donor_citations = donor["citations"]
    if not citations or not donor_citations or clean["pair_id"] == donor["pair_id"]:
        return None
    original, donor_citation = citations[0], donor_citations[0]
    if not isinstance(original, dict) or not isinstance(donor_citation, dict):
        return None
    original_source = original.get("source_id")
    replacement_source = donor_citation.get("source_id")
    if not original_source or not replacement_source or original_source == replacement_source:
        return None
    swapped = copy.deepcopy(citations)
    # Keep the original sentence placement and marker. Only the referenced
    # source changes, which is the citation-swap condition being measured.
    swapped[0]["source_id"] = replacement_source
    return {
        **copy.deepcopy(clean),
        "version": "corrupted",
        "corruption_type": "citation_swap",
        "citations": swapped,
        "corruption": {
            "rule": "replace_first_citation_with_unrelated_response_citation",
            "original_citation": original,
            "replacement_citation": donor_citation,
            "donor_pair_id": donor["pair_id"],
            "validation": "citation_metadata_changed_text_intentionally_unchanged",
        },
    }


def generate(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    clean_rows = [normalise_clean(row, i) for i, row in enumerate(rows, start=1)]
    if len({row["pair_id"] for row in clean_rows}) != len(clean_rows):
        raise ValueError("Input IDs must be unique.")
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    counts = {"clean": 0, "fact_change": 0, "negation": 0, "citation_swap": 0, "skipped": 0}

    for clean in clean_rows:
        output.append({**copy.deepcopy(clean), "version": "clean", "corruption_type": None, "corruption": None})
        counts["clean"] += 1
        for kind in ("fact_change", "negation"):
            record = text_corruption(clean, kind)
            if record is None:
                counts["skipped"] += 1
            else:
                output.append(record)
                counts[kind] += 1

        donors = [candidate for candidate in clean_rows if candidate["citations"] and candidate["pair_id"] != clean["pair_id"]]
        if donors:
            record = citation_swap(clean, rng.choice(donors))
        else:
            record = None
        if record is None:
            counts["skipped"] += 1
        else:
            output.append(record)
            counts["citation_swap"] += 1
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validated TrustGuard corruption pairs.")
    parser.add_argument("input", type=Path, help="JSONL clean-response dataset")
    parser.add_argument("output", type=Path, help="JSONL output path")
    parser.add_argument("--seed", type=int, default=20261014, help="Random seed for citation donors")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N clean rows")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[:args.limit]
    output, counts = generate(rows, args.seed)
    write_jsonl(output, args.output)
    print(f"Wrote {len(output)} rows to {args.output}")
    print("Clean: {clean}; fact changes: {fact_change}; negations: {negation}; citation swaps: {citation_swap}; skipped: {skipped}".format(**counts))
    print("A complete 30-response benchmark should contain 30 clean rows and 30 rows of each corruption type.")


if __name__ == "__main__":
    main()
