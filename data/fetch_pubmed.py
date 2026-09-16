"""
TrustGuard — PubMed evidence fetcher (NCBI E-utilities).

Requires internet access to ncbi.nlm.nih.gov. Run this on your own machine /
CI, not inside a network-isolated sandbox.

Usage:
    python -m data.fetch_pubmed --query "metformin lactic acidosis" --max 40 \
        --out cache/pubmed_metformin.jsonl

NCBI etiquette:
    - Without an API key you're capped at 3 requests/sec; with one, 10/sec.
      Set NCBI_API_KEY env var if you have one.
    - Always set a real contact email (--email) per NCBI's usage policy.
"""

from __future__ import annotations

import argparse
import os
import time
import xml.etree.ElementTree as ET

import requests

from data.schema import EvidencePassage, write_jsonl

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _rate_limit(has_key: bool) -> None:
    time.sleep(0.11 if has_key else 0.34)  # stay under 10/sec or 3/sec


def esearch_pmids(query: str, max_results: int, api_key: str | None, email: str) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": email,
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    _rate_limit(bool(api_key))
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch_abstracts(pmids: list[str], api_key: str | None, email: str) -> list[EvidencePassage]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    _rate_limit(bool(api_key))

    passages: list[EvidencePassage] = []
    root = ET.fromstring(r.text)
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = []
        for ab_text in article.findall(".//Abstract/AbstractText"):
            label = ab_text.get("Label")
            text = "".join(ab_text.itertext()).strip()
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = "\n".join(p for p in abstract_parts if p)

        if not abstract:
            continue  # skip entries with no usable evidence text

        date_el = article.find(".//PubDate/Year")
        pub_year = date_el.text if date_el is not None else ""

        passages.append(
            EvidencePassage(
                text=abstract,
                source_name="PubMed",
                source_tier="primary_literature",
                title=title,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                identifier=pmid,
                published_date=pub_year,
            )
        )
    return passages


def fetch_pubmed_corpus(query: str, max_results: int, email: str) -> list[EvidencePassage]:
    api_key = os.environ.get("NCBI_API_KEY")
    pmids = esearch_pmids(query, max_results, api_key, email)
    return efetch_abstracts(pmids, api_key, email)


def main():
    ap = argparse.ArgumentParser(description="Fetch a PubMed evidence corpus for a topic query.")
    ap.add_argument("--query", required=True, help='e.g. "metformin lactic acidosis"')
    ap.add_argument("--max", type=int, default=40, dest="max_results")
    ap.add_argument("--email", required=True, help="Contact email required by NCBI usage policy")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    passages = fetch_pubmed_corpus(args.query, args.max_results, args.email)
    write_jsonl(passages, args.out)
    print(f"Wrote {len(passages)} PubMed passages -> {args.out}")


if __name__ == "__main__":
    main()
