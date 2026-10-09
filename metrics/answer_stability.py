"""
TrustGuard — Answer Stability metric.

Idea: ask the same LLM the same question N times (with sampling, i.e.
temperature > 0). If the model "knows" the answer, the N responses should
largely agree; if it's confabulating, they'll drift claim-to-claim. We
measure that agreement with a hybrid score: semantic cosine similarity for
meaning-preserving paraphrases, plus Jaccard similarity for lexical overlap.

This module does not call any specific LLM API itself — it takes a
`generate_fn` (any zero-arg-per-call callable that returns a response
string for a given prompt) so it can be plugged into whatever LLM client
the rest of the pipeline uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, alnum-only token set. Simple and dependency-free; swap in
    stemming/lemmatization later if near-duplicate wording (e.g. "reduces"
    vs "reduce") is causing understated stability scores.
    """
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """|intersection| / |union| of the two texts' token sets, in [0, 1].

    Two empty responses are trivially identical (1.0); one empty and one
    non-empty share nothing (0.0).
    """
    tokens_a, tokens_b = _tokenize(text_a), _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


@dataclass
class PairwiseSimilarity:
    i: int          # index into the responses list
    j: int
    jaccard: float
    semantic: float | None
    hybrid: float


@dataclass
class StabilityResult:
    responses: list[str]
    pairwise: list[PairwiseSimilarity]
    average_stability: float  # mean of all pairwise hybrid scores
    average_jaccard: float
    average_semantic: float | None
    semantic_weight: float

    def to_dict(self) -> dict:
        return {
            "n_responses": len(self.responses),
            "pairwise": [
                {
                    "i": p.i,
                    "j": p.j,
                    "jaccard": p.jaccard,
                    "semantic_similarity": p.semantic,
                    "hybrid_similarity": p.hybrid,
                }
                for p in self.pairwise
            ],
            "average_stability": self.average_stability,
            "average_jaccard": self.average_jaccard,
            "average_semantic_similarity": self.average_semantic,
            "semantic_weight": self.semantic_weight,
        }


def generate_responses(
    generate_fn: Callable[[str], str],
    prompt: str,
    n: int = 5,
) -> list[str]:
    """Call `generate_fn(prompt)` n times and collect the responses.

    `generate_fn` should sample with temperature > 0 (e.g.
    `lambda p: client.generate(p, temperature=0.7)`); at temperature 0 every
    call returns the same text and stability will trivially be 1.0, which
    defeats the point of the metric.
    """
    if n < 2:
        raise ValueError("Need at least 2 responses to measure stability.")
    responses = []
    for index in range(1, n + 1):
        print(f"Sampling Answer Stability response {index}/{n}...")
        responses.append(generate_fn(prompt))
    return responses


def _semantic_embeddings(responses: list[str]):
    """Embed all responses together using the local MiniLM model."""
    try:
        from sentence_transformers import SentenceTransformer

        print("Computing semantic similarity with local MiniLM...")
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
        return model.encode(responses, normalize_embeddings=True)
    except (ImportError, OSError, RuntimeError):
        # On managed Windows systems, application-control policy can block
        # native extensions used by sentence-transformers. Stability remains
        # available through its lexical Jaccard fallback.
        return None


def compute_answer_stability(
    responses: list[str],
    *,
    semantic_weight: float = 0.7,
) -> StabilityResult:
    """Pairwise-compare responses with lexical and semantic similarity.

    ``semantic_weight`` defaults to 0.7, so paraphrases have greater impact
    than exact word overlap while Jaccard still contributes 30% of the score.
    """
    if len(responses) < 2:
        raise ValueError("Need at least 2 responses to compute pairwise stability.")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0 and 1.")

    embeddings = _semantic_embeddings(responses)
    lexical_weight = 1.0 - semantic_weight

    pairwise = []
    for i, j in combinations(range(len(responses)), 2):
        lexical = jaccard_similarity(responses[i], responses[j])
        semantic = (
            max(0.0, min(1.0, float(embeddings[i] @ embeddings[j])))
            if embeddings is not None
            else None
        )
        hybrid = (
            semantic_weight * semantic + lexical_weight * lexical
            if semantic is not None
            else lexical
        )
        pairwise.append(
            PairwiseSimilarity(
                i=i,
                j=j,
                jaccard=lexical,
                semantic=semantic,
                hybrid=hybrid,
            )
        )

    return StabilityResult(
        responses=responses,
        pairwise=pairwise,
        average_stability=sum(p.hybrid for p in pairwise) / len(pairwise),
        average_jaccard=sum(p.jaccard for p in pairwise) / len(pairwise),
        average_semantic=(
            sum(p.semantic for p in pairwise if p.semantic is not None)
            / len(pairwise)
            if embeddings is not None
            else None
        ),
        semantic_weight=semantic_weight if embeddings is not None else 0.0,
    )


def answer_stability(
    generate_fn: Callable[[str], str],
    prompt: str,
    n: int = 5,
) -> StabilityResult:
    """Convenience wrapper: generate N responses, then score them.

    Equivalent to:
        responses = generate_responses(generate_fn, prompt, n)
        result = compute_answer_stability(responses)
    """
    responses = generate_responses(generate_fn, prompt, n=n)
    return compute_answer_stability(responses)
