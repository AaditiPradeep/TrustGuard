from dataclasses import dataclass
from typing import Any, Dict, List



@dataclass
class CitationJudgement:
    claim_id: str
    claim_text: str
    citation_id: str
    source_name: str
    source_text: str

    # Final classification
    label: str

    # Continuous citation-quality score: 0-1
    score: float

    # Supporting model outputs
    claim_source_similarity: float
    entailment_probability: float
    contradiction_probability: float
    neutral_probability: float

    rationale: str


class MiniLMCitationJudge:
    """
    TrustGuard Citation Quality Judge.

    Each LLM claim is compared ONLY against its cited source.

    The user's original question is NOT used.

    Classification:
        DIRECT
        INDIRECT
        IRRELEVANT

    DIRECT:
        The cited source directly addresses the claim.
        This includes both agreement and contradiction.

    INDIRECT:
        The cited source is related to the claim,
        but does not directly address it.

    IRRELEVANT:
        The cited source is not meaningfully related
        to the claim.

    Scoring:
        DIRECT       -> 0.70 to 1.00
        INDIRECT     -> 0.30 to 0.70
        IRRELEVANT   -> 0.00 to 0.30

    The overall Citation Quality Score is the average
    of all individual claim-citation scores.
    """

    def __init__(
        self,
        nli_model_name: str = (
            "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        ),
        similarity_model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        direct_threshold: float = 0.55,
        irrelevant_threshold: float = 0.25,
    ) -> None:

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
        except ImportError as error:
            raise ImportError(
                "Citation quality requires sentence-transformers and transformers. "
                "Install them with: pip install -r requirements.txt"
            ) from error

        self._np = np

        # NLI model
        self.nli = pipeline(
            "text-classification",
            model=nli_model_name,
        )

        # Embedding model
        self.embedder = SentenceTransformer(
            similarity_model_name,
            local_files_only=True,
        )

        self.direct_threshold = direct_threshold
        self.irrelevant_threshold = irrelevant_threshold

    # =========================================================
    # NLI
    # =========================================================

    def _nli(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, float]:

        """
        Run NLI with:

            premise    = cited source
            hypothesis = LLM claim

        We care about:

            entailment
            contradiction
            neutral

        Both entailment and contradiction indicate that
        the source directly addresses the claim.
        """

        result = self.nli(
            premise,
            text_pair=hypothesis,
            truncation=True,
            top_k=None,
        )

        # -----------------------------------------------------
        # Normalize Transformers output.
        #
        # Depending on Transformers version, output may be:
        #
        # [
        #     {"label": "...", "score": ...},
        #     ...
        # ]
        #
        # OR:
        #
        # [
        #     [
        #         {"label": "...", "score": ...},
        #         ...
        #     ]
        # ]
        #
        # OR a single dictionary.
        # -----------------------------------------------------

        if isinstance(result, dict):
            result = [result]

        elif (
            isinstance(result, list)
            and len(result) > 0
            and isinstance(result[0], list)
        ):
            result = result[0]

        scores: Dict[str, float] = {
            "entailment": 0.0,
            "contradiction": 0.0,
            "neutral": 0.0,
        }

        # -----------------------------------------------------
        # Get model label mapping.
        #
        # Some models return:
        #
        # entailment
        # contradiction
        # neutral
        #
        # Others may return:
        #
        # LABEL_0
        # LABEL_1
        # LABEL_2
        #
        # The model config tells us what those labels mean.
        # -----------------------------------------------------

        id2label = getattr(
            self.nli.model.config,
            "id2label",
            {},
        )

        # -----------------------------------------------------
        # Process results
        # -----------------------------------------------------

        for item in result:

            if not isinstance(item, dict):
                continue

            raw_label = str(
                item.get("label", "")
            )

            score = float(
                item.get("score", 0.0)
            )

            label = raw_label.lower()

            # ---------------------------------------------
            # Handle normal textual labels
            # ---------------------------------------------

            if "entail" in label:

                scores["entailment"] = score

            elif "contrad" in label:

                scores["contradiction"] = score

            elif "neutral" in label:

                scores["neutral"] = score

            # ---------------------------------------------
            # Handle LABEL_0 / LABEL_1 / LABEL_2
            # ---------------------------------------------

            elif label.startswith("label_"):

                try:

                    label_id = int(
                        label.split("_")[-1]
                    )

                    mapped_label = str(
                        id2label.get(
                            label_id,
                            ""
                        )
                    ).lower()

                    if "entail" in mapped_label:

                        scores["entailment"] = score

                    elif "contrad" in mapped_label:

                        scores["contradiction"] = score

                    elif "neutral" in mapped_label:

                        scores["neutral"] = score

                except (
                    ValueError,
                    TypeError,
                ):
                    pass

        return scores

    # =========================================================
    # SEMANTIC SIMILARITY
    # =========================================================

    def _embedding_similarity(
        self,
        claim: str,
        source: str,
    ) -> float:

        """
        Calculate semantic similarity between:

            LLM claim
            cited source
        """

        embeddings = self.embedder.encode(
            [claim, source],
            normalize_embeddings=True,
        )

        similarity = float(
            self._np.dot(
                embeddings[0],
                embeddings[1],
            )
        )

        # Convert to safe 0-1 range
        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

    # =========================================================
    # CONTINUOUS SCORE
    # =========================================================

    def _calculate_score(
        self,
        label: str,
        similarity: float,
        direct_relationship: float,
    ) -> float:

        """
        Calculate continuous citation-quality score.

        DIRECT:
            0.70 - 1.00

        INDIRECT:
            0.30 - 0.70

        IRRELEVANT:
            0.00 - 0.30

        The score measures citation relevance/directness.

        It does NOT penalize a source simply because it
        contradicts the LLM claim.
        """

        # -----------------------------------------------------
        # DIRECT
        # -----------------------------------------------------

        if label == "DIRECT":

            # Map direct_relationship:
            #
            # 0.55 -> 0.70
            # 1.00 -> 1.00

            score = (
                0.70
                + 0.30
                * (
                    (
                        direct_relationship
                        - self.direct_threshold
                    )
                    / (
                        1.0
                        - self.direct_threshold
                    )
                )
            )

        # -----------------------------------------------------
        # INDIRECT
        # -----------------------------------------------------

        elif label == "INDIRECT":

            # Map similarity:
            #
            # 0.25 -> 0.30
            # 0.55 -> 0.70

            score = (
                0.30
                + 0.40
                * (
                    (
                        similarity
                        - self.irrelevant_threshold
                    )
                    / (
                        self.direct_threshold
                        - self.irrelevant_threshold
                    )
                )
            )

        # -----------------------------------------------------
        # IRRELEVANT
        # -----------------------------------------------------

        else:

            # Map similarity:
            #
            # 0.00 -> 0.00
            # 0.25 -> 0.30

            score = (
                0.30
                * (
                    similarity
                    / self.irrelevant_threshold
                )
            )

        # Keep score inside 0-1
        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    # =========================================================
    # JUDGE ONE CLAIM-CITATION PAIR
    # =========================================================

    def judge_pair(
        self,
        claim: str,
        source_text: str,
        claim_id: str = "",
        citation_id: str = "",
        source_name: str = "",
    ) -> CitationJudgement:

        # -----------------------------------------------------
        # 1. Semantic similarity
        # -----------------------------------------------------

        similarity = self._embedding_similarity(
            claim,
            source_text,
        )

        # -----------------------------------------------------
        # 2. NLI
        #
        # Source = premise
        # Claim  = hypothesis
        # -----------------------------------------------------

        nli = self._nli(
            premise=source_text,
            hypothesis=claim,
        )

        entailment = nli["entailment"]

        contradiction = nli[
            "contradiction"
        ]

        neutral = nli["neutral"]

        # -----------------------------------------------------
        # 3. Direct relationship
        #
        # BOTH entailment and contradiction mean that the
        # source directly addresses the claim.
        #
        # Therefore:
        #
        #     DIRECT = max(entailment, contradiction)
        #
        # Contradiction is NOT treated as low-quality
        # citation relevance.
        # -----------------------------------------------------

        direct_relationship = max(
            entailment,
            contradiction,
        )

        # -----------------------------------------------------
        # 4. Classification
        # -----------------------------------------------------

        if (
            direct_relationship
            >= self.direct_threshold
        ):

            label = "DIRECT"

            rationale = (
                "The cited source directly addresses "
                "the LLM claim. Directness is based on "
                "the relationship between the source and "
                "claim, so a source can be DIRECT even "
                "when it contradicts the claim. "
                f"Direct relationship="
                f"{direct_relationship:.3f}."
            )

        elif (
            similarity
            >= self.irrelevant_threshold
        ):

            label = "INDIRECT"

            rationale = (
                "The cited source is meaningfully related "
                "to the LLM claim but does not directly "
                "address it. "
                f"Similarity={similarity:.3f}, "
                f"direct relationship="
                f"{direct_relationship:.3f}."
            )

        else:

            label = "IRRELEVANT"

            rationale = (
                "The cited source does not appear "
                "meaningfully related to the LLM claim. "
                f"Similarity={similarity:.3f}, "
                f"direct relationship="
                f"{direct_relationship:.3f}."
            )

        # -----------------------------------------------------
        # 5. Continuous score
        # -----------------------------------------------------

        score = self._calculate_score(
            label=label,
            similarity=similarity,
            direct_relationship=direct_relationship,
        )

        # -----------------------------------------------------
        # 6. Return judgement
        # -----------------------------------------------------

        return CitationJudgement(
            claim_id=claim_id,
            claim_text=claim,
            citation_id=citation_id,
            source_name=source_name,
            source_text=source_text,
            label=label,
            score=score,
            claim_source_similarity=similarity,
            entailment_probability=entailment,
            contradiction_probability=contradiction,
            neutral_probability=neutral,
            rationale=rationale,
        )

    # =========================================================
    # JUDGE MULTIPLE CLAIM-CITATION PAIRS
    # =========================================================

    def judge_claims(
        self,
        claims: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        judgements: List[
            CitationJudgement
        ] = []

        # -----------------------------------------------------
        # Judge every claim against its own cited source
        # -----------------------------------------------------

        total_claims = len(claims)
        for index, item in enumerate(claims, start=1):

            print(
                f"Scoring citation {index}/{total_claims}: "
                f"{item.get('source_name', 'unknown source')}"
            )

            judgement = self.judge_pair(
                claim=item["claim"],
                source_text=item["source_text"],
                claim_id=str(
                    item.get(
                        "claim_id",
                        "",
                    )
                ),
                citation_id=str(
                    item.get(
                        "citation_id",
                        "",
                    )
                ),
                source_name=str(
                    item.get(
                        "source_name",
                        "",
                    )
                ),
            )

            judgements.append(
                judgement
            )

        # -----------------------------------------------------
        # Overall Citation Quality Score
        #
        # Every claim-citation pair contributes equally.
        # -----------------------------------------------------

        if judgements:

            citation_quality_score = float(
                sum(
                    judgement.score
                    for judgement in judgements
                )
                / len(judgements)
            )

        else:

            citation_quality_score = 0.0

        # -----------------------------------------------------
        # Counts
        # -----------------------------------------------------

        direct_count = sum(
            judgement.label == "DIRECT"
            for judgement in judgements
        )

        indirect_count = sum(
            judgement.label == "INDIRECT"
            for judgement in judgements
        )

        irrelevant_count = sum(
            judgement.label == "IRRELEVANT"
            for judgement in judgements
        )

        # -----------------------------------------------------
        # Final report
        # -----------------------------------------------------

        return {
            "judgements": judgements,

            "citation_quality_score": (
                citation_quality_score
            ),

            "citation_quality_percent": (
                citation_quality_score * 100.0
            ),

            "counts": {
                "DIRECT": direct_count,
                "INDIRECT": indirect_count,
                "IRRELEVANT": irrelevant_count,
            },

            "total_pairs": len(
                judgements
            ),
        }
