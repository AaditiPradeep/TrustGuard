"""
TrustGuard - CDC Evidence Extractor

Uses the official CDC Content Services API.

Pipeline:

CDC Topics
    ↓
Topic IDs
    ↓
CDC Media
    ↓
Syndicated Content
    ↓
EvidencePassage
    ↓
cache/cdc_evidence.jsonl
"""

from __future__ import annotations

import html
import re
import requests

from data.schema import EvidencePassage, write_jsonl


# ============================================================
# CDC API
# ============================================================

CDC_BASE = "https://tools.cdc.gov/api/v2/resources"

TOPICS_API = f"{CDC_BASE}/topics"
MEDIA_API = f"{CDC_BASE}/media"

OUTPUT_FILE = "cache/cdc_evidence.jsonl"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text) -> str:

    if not text:
        return ""

    text = html.unescape(str(text))

    # Remove scripts
    text = re.sub(
        r"<script.*?>.*?</script>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove styles
    text = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# GET CDC TOPICS
# ============================================================

def get_cdc_topics():

    print("=" * 70)
    print("FETCHING CDC TOPICS")
    print("=" * 70)

    params = {
        "language": "English",
        "showChild": "true"
    }

    response = requests.get(
        TOPICS_API,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "TrustGuard/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    print(
        f"CDC topics received: {len(results)}"
    )

    return results


# ============================================================
# FIND RELEVANT TOPICS
# ============================================================

def find_medical_topics(topics):

    keywords = [
        "disease",
        "condition",
        "infectious",
        "vaccine",
        "vaccination",
        "health",
        "prevention",
        "chronic",
        "respiratory",
        "cancer",
        "diabetes",
        "heart",
        "mental",
        "nutrition",
        "medication",
        "public health"
    ]

    selected = []

    def inspect_topic(topic):

        if not isinstance(topic, dict):
            return

        name = clean_text(
            topic.get("name", "")
        )

        description = clean_text(
            topic.get("description", "")
        )

        combined = (
            f"{name} {description}"
        ).lower()

        topic_id = topic.get("id")

        if topic_id is not None:

            if any(
                keyword in combined
                for keyword in keywords
            ):

                selected.append({
                    "id": topic_id,
                    "name": name
                })

        # CDC can return nested child topics
        children = topic.get("items", [])

        if isinstance(children, list):

            for child in children:
                inspect_topic(child)

    for topic in topics:
        inspect_topic(topic)

    # Remove duplicate IDs
    unique = {}

    for topic in selected:

        unique[topic["id"]] = topic

    selected = list(unique.values())

    print()
    print(
        f"Relevant CDC topics found: "
        f"{len(selected)}"
    )

    for topic in selected[:30]:

        print(
            f"  {topic['id']} -> "
            f"{topic['name']}"
        )

    return selected


# ============================================================
# GET MEDIA FOR TOPIC
# ============================================================

def get_media_for_topic(
    topic_id,
    max_results=50
):

    params = {
        "topicids": str(topic_id),
        "max": max_results,
        "pagenum": 1,
        "sort": "-datePublished"
    }

    response = requests.get(
        MEDIA_API,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "TrustGuard/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    results = data.get(
        "results",
        []
    )

    pagination = data.get(
        "meta",
        {}
    ).get(
        "pagination",
        {}
    )

    print(
        f"Topic {topic_id}: "
        f"total={pagination.get('total', 0)}, "
        f"returned={len(results)}"
    )

    return results


# ============================================================
# GET ACTUAL CDC CONTENT
# ============================================================

def get_media_content(media_id):

    url = f"{MEDIA_API}/{media_id}/content"

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "TrustGuard/1.0",
            "Accept": "text/html, application/xhtml+xml, application/xml"
        }
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # IMPORTANT:
    # CDC content endpoint may return HTML/XML rather than JSON.
    # Therefore DO NOT use response.json() here.
    # --------------------------------------------------------

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    print(
        f"    Content-Type: {content_type}"
    )

    raw_content = response.text

    if not raw_content:
        return ""

    return clean_text(raw_content)


# ============================================================
# EXTRACT CDC EVIDENCE
# ============================================================

def fetch_cdc(
    max_topics=10,
    max_media_per_topic=20
):

    topics = get_cdc_topics()

    selected_topics = find_medical_topics(
        topics
    )

    selected_topics = selected_topics[
        :max_topics
    ]

    evidence = []

    seen_media = set()

    print()
    print("=" * 70)
    print("FETCHING CDC MEDICAL CONTENT")
    print("=" * 70)

    for topic in selected_topics:

        topic_id = topic["id"]
        topic_name = topic["name"]

        print()
        print(
            f"TOPIC: {topic_name} "
            f"(ID={topic_id})"
        )

        try:

            media_items = get_media_for_topic(
                topic_id,
                max_results=max_media_per_topic
            )

        except Exception as e:

            print(
                "Failed to retrieve media:",
                e
            )

            continue

        for item in media_items:

            if not isinstance(item, dict):
                continue

            # ------------------------------------------------
            # Media ID
            # ------------------------------------------------

            media_id = (
                item.get("id")
                or item.get("mediaId")
            )

            if media_id is None:
                continue

            media_id = str(media_id)

            # Prevent duplicates
            if media_id in seen_media:
                continue

            seen_media.add(media_id)

            # ------------------------------------------------
            # Media type
            # ------------------------------------------------

            media_type = str(
                item.get(
                    "mediaType",
                    ""
                )
            ).lower()

            # We want actual textual HTML content
            if media_type != "html":

                continue

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            title = clean_text(
                item.get(
                    "name"
                )
                or item.get(
                    "title"
                )
                or ""
            )

            description = clean_text(
                item.get(
                    "description",
                    ""
                )
            )

            source_url = (
                item.get(
                    "sourceUrl"
                )
                or item.get(
                    "targetUrl"
                )
                or item.get(
                    "persistentUrl"
                )
                or ""
            )

            published_date = (
                item.get(
                    "datePublished"
                )
                or item.get(
                    "dateModified"
                )
                or ""
            )

            print()
            print(
                f"  Fetching: "
                f"{title[:80]}"
            )

            # ------------------------------------------------
            # Actual CDC content
            # ------------------------------------------------

            try:

                content = get_media_content(
                    media_id
                )

            except Exception as e:

                print(
                    "  Content fetch failed:",
                    e
                )

                continue

            if not content:

                print(
                    "  No content returned."
                )

                continue

            # ------------------------------------------------
            # Build evidence
            # ------------------------------------------------

            if title:

                text = (
                    f"{title}. "
                    f"{content}"
                )

            else:

                text = content

            # Remove tiny records
            if len(text) < 200:

                print(
                    "  Skipped: content too short."
                )

                continue

            # ------------------------------------------------
            # EvidencePassage
            # ------------------------------------------------

            passage = EvidencePassage(
                text=text,
                source_name="CDC",
                source_tier="clinical_guideline",
                title=title,
                url=source_url,
                identifier=f"CDC-{media_id}",
                published_date=published_date
            )

            evidence.append(
                passage
            )

            print(
                "  ✓ Added"
            )

    return evidence


# ============================================================
# MAIN
# ============================================================

def main():

    records = fetch_cdc(
        max_topics=10,
        max_media_per_topic=20
    )

    print()
    print("=" * 70)
    print("CDC EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"Total usable CDC records: "
        f"{len(records)}"
    )

    if not records:

        print(
            "\nNo CDC evidence was extracted."
        )

        return

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_jsonl(
        records,
        OUTPUT_FILE
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Sample
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE CDC RECORD")
    print("=" * 70)

    sample = records[0]

    print()
    print("Title:")
    print(sample.title)

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
    print("Evidence:")
    print(sample.text[:1000])


if __name__ == "__main__":
    main()