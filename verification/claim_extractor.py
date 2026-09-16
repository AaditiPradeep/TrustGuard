"""
TrustGuard - Atomic Medical Claim Extraction

Extracts independently verifiable factual medical claims
from an LLM-generated answer using Qwen3-8B through the
Hugging Face Inference API.
"""

from __future__ import annotations

import json
import re
from typing import Any

from data.schema import Claim
from local_llm.ollama import DEFAULT_MODEL, chat


MODEL_NAME = DEFAULT_MODEL


SYSTEM_PROMPT = """
/no_think

You are a medical factual claim extraction system.

Your ONLY task is to extract atomic factual medical claims from an
LLM-generated answer.

Do not verify whether the claims are correct.
Do not use outside medical knowledge.
Do not add facts that are not stated or clearly implied by the answer.


ATOMIC CLAIM RULE

Each claim must contain exactly ONE independently verifiable factual
proposition.

If a sentence contains multiple facts that could be checked separately,
split them into separate claims.


Examples:

Input:
"Metformin may cause nausea and diarrhea."

Correct output:
{
  "claims": [
    "Metformin may cause nausea.",
    "Metformin may cause diarrhea."
  ]
}


Input:
"Influenza can cause fever, cough, and muscle aches."

Correct output:
{
  "claims": [
    "Influenza can cause fever.",
    "Influenza can cause cough.",
    "Influenza can cause muscle aches."
  ]
}


Input:
"Metformin is used for type 2 diabetes and may cause lactic acidosis."

Correct output:
{
  "claims": [
    "Metformin is used to treat type 2 diabetes.",
    "Metformin may cause lactic acidosis."
  ]
}


QUALIFIER RULE

Preserve important medical qualifiers such as:

- may
- can
- rare
- rarely
- common
- commonly
- usually
- sometimes
- always
- never
- mild
- severe
- serious
- dosage
- frequency
- duration
- age group
- population
- disease stage

Do not remove a qualifier if doing so changes the meaning of the claim.


Example:

Input:
"Lactic acidosis is a rare but serious adverse effect of metformin."

Correct output:
{
  "claims": [
    "Metformin can cause lactic acidosis.",
    "Metformin-associated lactic acidosis is rare.",
    "Metformin-associated lactic acidosis is serious."
  ]
}


CONTEXT RULE

Each claim should make sense independently.

Resolve pronouns such as:

- it
- they
- this drug
- this treatment
- this condition

using only information already present in the question or answer.


Example:

Question:
"What are the side effects of metformin?"

Answer:
"It may cause nausea."

Correct output:
{
  "claims": [
    "Metformin may cause nausea."
  ]
}


WHAT NOT TO EXTRACT

Do not extract:

- greetings
- conversational filler
- opinions
- rhetorical statements
- questions
- generic advice such as "consult your doctor"
- statements without factual medical content

Do not invent additional facts.


OUTPUT FORMAT

Return ONLY valid JSON.

Use exactly this structure:

{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}

If there are no factual medical claims, return:

{
  "claims": []
}

Do not return Markdown.
Do not return explanations.
Do not return reasoning.
Do not return any text before or after the JSON.
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    """
    Parse JSON returned by Qwen.

    Also handles accidental Markdown code fences or extra text.
    """

    if not text:
        raise ValueError("Cannot parse JSON from an empty response.")

    text = text.strip()

    # Remove Markdown code fences if present
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    # First try parsing the whole response
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Could not find valid JSON in Qwen response.\n\n"
            f"Raw response:\n{text}"
        )

    try:
        return json.loads(match.group(0))

    except json.JSONDecodeError as error:
        raise ValueError(
            "Could not parse JSON from Qwen response.\n\n"
            f"Raw response:\n{text}"
        ) from error


def _clean_claims(raw_claims: list[Any]) -> list[str]:
    """
    Clean and deduplicate extracted claims while preserving order.
    """

    claims: list[str] = []
    seen: set[str] = set()

    for item in raw_claims:

        if not isinstance(item, str):
            continue

        claim = item.strip()

        if not claim:
            continue

        # Remove accidental bullets or numbering
        claim = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            claim,
        )

        # Normalize whitespace
        claim = re.sub(
            r"\s+",
            " ",
            claim,
        ).strip()

        # Add punctuation for consistency
        if claim and claim[-1] not in ".!?":
            claim += "."

        normalized = claim.lower().strip()

        if normalized not in seen:
            seen.add(normalized)
            claims.append(claim)

    return claims


def _call_model(
    model_name: str,
    messages: list[dict],
) -> str | None:
    """
    Call a local Ollama model for claim extraction.
    """

    return chat(messages, model=model_name, max_tokens=500)


def extract_claims(
    question: str,
    answer: str,
    response_id: str,
    model_name: str = MODEL_NAME,
) -> list[Claim]:
    """
    Extract atomic factual medical claims from an LLM-generated answer.

    Parameters
    ----------
    question:
        Original medical question.

    answer:
        LLM-generated answer to analyse.

    response_id:
        Identifier for the generated response.

    model_name:
        Local Ollama model used for extraction.

    Returns
    -------
    list[Claim]
        Atomic factual claims represented using TrustGuard's
        existing Claim schema.
    """

    if not answer or not answer.strip():
        return []

    user_prompt = f"""
/no_think

QUESTION:
{question}

ANSWER:
{answer}

Extract every atomic factual medical claim from the answer.

Requirements:

- Exactly one independently verifiable fact per claim.
- Split compound claims.
- Split multiple symptoms, side effects, treatments, diseases,
  complications, outcomes, or properties when they can be verified
  independently.
- Preserve important qualifiers such as may, can, rare, common,
  serious, always, never, dosage, frequency, duration, population,
  age group, and severity.
- Resolve pronouns using the supplied question and answer.
- Do not introduce outside medical knowledge.
- Do not determine whether a claim is true or false.
- Return only JSON.

Required output format:

{{
  "claims": [
    "claim 1",
    "claim 2"
  ]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    content = _call_model(
        model_name=model_name,
        messages=messages,
    )

    # Retry once with an even stricter instruction if Qwen
    # unexpectedly consumes the output in reasoning mode.
    if not content:
        print(
            "Qwen returned empty content. "
            "Retrying with stricter non-thinking instruction..."
        )

        retry_messages = [
            {
                "role": "system",
                "content": """
/no_think

Do not reason aloud.
Do not produce hidden analysis.
Return only the requested JSON object.
""".strip(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        content = _call_model(
            model_name=model_name,
            messages=retry_messages,
        )

    if not content:
        raise RuntimeError(
            "Qwen returned empty message.content after two attempts. "
            "Check the debug output for finish_reason and reasoning_content."
        )

    parsed = _extract_json(content)

    raw_claims = parsed.get("claims")

    if raw_claims is None:
        raise ValueError(
            "Qwen response did not contain a 'claims' field.\n\n"
            f"Response:\n{content}"
        )

    if not isinstance(raw_claims, list):
        raise ValueError(
            "Expected 'claims' to be a list, but received "
            f"{type(raw_claims).__name__}."
        )

    claim_texts = _clean_claims(raw_claims)

    return [
        Claim(
            text=claim_text,
            source_response_id=response_id,
        )
        for claim_text in claim_texts
    ]


if __name__ == "__main__":

    test_question = (
        "What are the uses and side effects of metformin?"
    )

    test_answer = """
    Metformin is commonly used to treat type 2 diabetes.
    It may cause nausea and diarrhea.
    A rare but serious adverse effect is lactic acidosis.
    """

    print("\nExtracting claims...\n")

    extracted_claims = extract_claims(
        question=test_question,
        answer=test_answer,
        response_id="test_response_001",
    )

    print(
        f"\nExtracted {len(extracted_claims)} claims:\n"
    )

    for i, claim in enumerate(
        extracted_claims,
        start=1,
    ):
        print(f"Claim {i}")
        print(f"ID   : {claim.claim_id}")
        print(f"Text : {claim.text}")
        print()
