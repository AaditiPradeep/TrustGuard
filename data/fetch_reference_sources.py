"""
TrustGuard — MedlinePlus + openFDA evidence fetchers.

MedlinePlus: NIH/NLM's consumer health encyclopedia — good general
"reference tier" grounding for common conditions, drugs, and tests.

openFDA: FDA structured drug labeling — best "regulatory tier" source for
drug interaction / dosage / adverse-event claims specifically (the exact
claim category the project's own risk analysis flags as highest-stakes).

Both require internet access; run on your own machine, not the sandbox.
"""

from __future__ import annotations

import argparse
import re

import requests

from data.schema import EvidencePassage, write_jsonl

MEDLINEPLUS_SEARCH = "https://wsearch.nlm.nih.gov/ws/query"
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"


def _strip_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", " ", raw or "").strip()


def fetch_medlineplus(
    topic_query: str,
    max_results: int = 10
) -> list[EvidencePassage]:
    """
    Search the official NIH/NLM MedlinePlus healthTopics database.

    Returns clean EvidencePassage objects suitable for TrustGuard
    retrieval and claim verification.
    """
    params = {
        "db": "healthTopics",
        "term": topic_query,
        "retmax": max_results,
        "rettype": "all",
        "tool": "TrustGuard",
    }

    r = requests.get(
        MEDLINEPLUS_SEARCH,
        params=params,
        timeout=30,
        headers={"User-Agent": "TrustGuard/1.0"},
    )
    r.raise_for_status()

    import xml.etree.ElementTree as ET

    root = ET.fromstring(r.text)

    passages: list[EvidencePassage] = []

    for doc in root.findall(".//document"):
        url = doc.get("url", "").strip()

        title = ""
        summary = ""
        sections = []

        for content in doc.findall("content"):
            name = content.get("name", "").strip()

            text = _strip_html(
                " ".join(content.itertext())
            )

            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            if name.lower() == "title":
                title = text

            elif name.lower() == "fullsummary":
                summary = text

            else:
                # Other useful MedlinePlus sections
                sections.append(text)

        # Prefer the full summary. If unavailable, use other
        # textual content returned by the API.
        if summary:
            evidence_text = summary
        else:
            evidence_text = " ".join(sections)

        evidence_text = re.sub(r"\s+", " ", evidence_text).strip()

        if not evidence_text:
            continue

        passages.append(
            EvidencePassage(
                text=evidence_text,
                source_name="NIH / MedlinePlus",
                source_tier="reference",
                title=title,
                url=url,
                identifier=url or title,
            )
        )

    return passages


def fetch_openfda_label(drug_name: str, max_results: int = 5) -> list[EvidencePassage]:
    """Pull structured FDA label sections (warnings, interactions, dosage) for a drug."""
    params = {
        "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
        "limit": max_results,
    }
    r = requests.get(OPENFDA_LABEL, params=params, timeout=30)
    if r.status_code == 404:
        return []  # no matching label
    r.raise_for_status()
    data = r.json()

    passages: list[EvidencePassage] = []
    for result in data.get("results", []):
        brand = ", ".join(result.get("openfda", {}).get("brand_name", [])[:1]) or drug_name
        set_id = result.get("set_id", "")
        for field_name in ("warnings", "drug_interactions", "dosage_and_administration", "boxed_warning"):
            for chunk in result.get(field_name, []):
                text = _strip_html(chunk)
                if not text:
                    continue
                passages.append(
                    EvidencePassage(
                        text=text,
                        source_name="openFDA",
                        source_tier="regulatory",
                        title=f"{brand} — {field_name.replace('_', ' ')}",
                        url=f"https://labels.fda.gov/{set_id}" if set_id else "https://open.fda.gov",
                        identifier=f"{set_id}:{field_name}",
                    )
                )
    return passages


def main():
    ap = argparse.ArgumentParser(
        description="Fetch MedlinePlus and/or openFDA evidence."
    )

    ap.add_argument(
        "--medlineplus",
        nargs="+",
        help="One or more MedlinePlus topic queries."
    )

    ap.add_argument(
        "--openfda-drug",
        nargs="+",
        help="One or more drug names for openFDA."
    )

    ap.add_argument(
        "--max",
        type=int,
        default=10,
        dest="max_results"
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Output JSONL path"
    )

    args = ap.parse_args()

    passages: list[EvidencePassage] = []

    # -------------------------
    # MedlinePlus
    # -------------------------
    if args.medlineplus:

        for topic in args.medlineplus:

            print(f"\nSearching MedlinePlus: {topic}")

            try:
                results = fetch_medlineplus(
                    topic,
                    args.max_results
                )

                print(
                    f"  Results returned: {len(results)}"
                )

                passages.extend(results)

            except Exception as e:
                print(
                    f"  MedlinePlus search failed: {e}"
                )

    # -------------------------
    # openFDA
    # -------------------------
    if args.openfda_drug:

        for drug in args.openfda_drug:

            print(f"\nSearching openFDA: {drug}")

            try:
                results = fetch_openfda_label(
                    drug,
                    args.max_results
                )

                print(
                    f"  Results returned: {len(results)}"
                )

                passages.extend(results)

            except Exception as e:
                print(
                    f"  openFDA search failed: {e}"
                )

    # -------------------------
    # Deduplicate
    # -------------------------
    unique = {}

    for passage in passages:
        unique[passage.passage_id] = passage

    passages = list(unique.values())

    if not passages:
        print(
            "\nNo evidence found. "
            "Use --medlineplus and/or --openfda-drug."
        )
        return

    write_jsonl(passages, args.out)

    print(
        f"\nWrote {len(passages)} unique passages "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()

