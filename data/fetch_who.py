"""
TrustGuard — WHO evidence extractor.

Fetches health-topic information from the official WHO API
and stores it as EvidencePassage JSONL records.
"""

from __future__ import annotations

import re
import requests

from data.schema import EvidencePassage, write_jsonl


WHO_API = "https://www.who.int/api/whoreference/healthtopics"

OUTPUT_FILE = "cache/who_evidence.jsonl"


def clean_text(text) -> str:
    """Remove HTML and normalize whitespace."""

    if not text:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def fetch_who(limit: int | None = None) -> list[EvidencePassage]:
    """
    Fetch WHO health-topic records and convert them into
    TrustGuard EvidencePassage objects.
    """

    print("Fetching WHO health topics...")

    response = requests.get(
        WHO_API,
        timeout=30,
        headers={
            "User-Agent": "TrustGuard/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    print("WHO API response type:", type(data).__name__)

    # ---------------------------------------------------------
    # Handle the API response safely
    # ---------------------------------------------------------

    if isinstance(data, list):

        items = data

    elif isinstance(data, dict):

        print("WHO API response keys:", list(data.keys()))

        # Some APIs wrap the actual records in "value"
        if isinstance(data.get("value"), list):
            items = data["value"]

        # Handle a single WHO record
        elif "Title" in data:
            items = [data]

        else:
            raise ValueError(
                "Unexpected WHO API response format.\n"
                f"Keys received: {list(data.keys())}"
            )

    else:

        raise ValueError(
            f"Unexpected WHO API response type: {type(data)}"
        )

    print(f"WHO records received: {len(items)}")

    # ---------------------------------------------------------
    # Convert WHO records → EvidencePassage
    # ---------------------------------------------------------

    evidence = []

    for item in items:

        # This prevents the exact error you encountered:
        # AttributeError: 'str' object has no attribute 'get'

        if not isinstance(item, dict):
            print("Skipping unexpected WHO item:", item)
            continue

        # -----------------------------
        # Title
        # -----------------------------

        title = clean_text(
            item.get("Title")
        )

        # -----------------------------
        # Main WHO summary
        # -----------------------------

        summary = clean_text(
            item.get("Summary")
        )

        # WHO sometimes provides additional tab summaries
        tab1 = clean_text(
            item.get("Tab1Summary")
        )

        tab2 = clean_text(
            item.get("Tab2Summary")
        )

        tab3 = clean_text(
            item.get("Tab3Summary")
        )

        # Combine available text
        text_parts = [
            summary,
            tab1,
            tab2,
            tab3
        ]

        text_parts = [
            text for text in text_parts
            if text
        ]

        text = " ".join(text_parts)

        # Skip records without usable evidence
        if not text:
            continue

        # -----------------------------------------------------
        # URL
        # -----------------------------------------------------

        url = item.get("ItemDefaultUrl") or ""

        if url.startswith("/"):
            url = "https://www.who.int" + url

        # Fallback using WHO topic slug
        if not url:

            url_name = item.get("UrlName")

            if url_name:
                url = (
                    "https://www.who.int/health-topics/"
                    + str(url_name)
                )

        # -----------------------------------------------------
        # Identifier
        # -----------------------------------------------------

        identifier = (
            item.get("SystemSourceKey")
            or item.get("UrlName")
            or title
        )

        # -----------------------------------------------------
        # Publication / modification date
        # -----------------------------------------------------

        published_date = (
            item.get("LastModified")
            or item.get("PublicationDate")
            or ""
        )

        # -----------------------------------------------------
        # Create EvidencePassage
        # -----------------------------------------------------

        passage = EvidencePassage(
            text=text,
            source_name="WHO",
            source_tier="clinical_guideline",
            title=title,
            url=url,
            identifier=str(identifier),
            published_date=str(published_date)
        )

        evidence.append(passage)

        # -----------------------------------------------------
        # Limit number of records if requested
        # -----------------------------------------------------

        if limit is not None and len(evidence) >= limit:
            break

    print(
        f"Successfully converted "
        f"{len(evidence)} WHO records into EvidencePassage objects."
    )

    return evidence


def main():

    records = fetch_who()

    # Save to cache
    write_jsonl(records, OUTPUT_FILE)

    print()
    print("=" * 60)
    print("WHO extraction completed")
    print("=" * 60)
    print(f"Records saved: {len(records)}")
    print(f"Output file: {OUTPUT_FILE}")

    # Show a sample
    if records:

        print()
        print("Sample record:")
        print("-" * 60)

        sample = records[0]

        print("Title:")
        print(sample.title)

        print()
        print("Source:")
        print(sample.source_name)

        print()
        print("URL:")
        print(sample.url)

        print()
        print("Published:")
        print(sample.published_date)

        print()
        print("Passage ID:")
        print(sample.passage_id)

        print()
        print("Evidence text:")
        print(sample.text[:500])

        if len(sample.text) > 500:
            print("...")


if __name__ == "__main__":
    main()