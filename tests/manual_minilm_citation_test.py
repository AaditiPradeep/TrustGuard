from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from citation.minilm_judge import MiniLMCitationJudge


def main():

    print("=" * 70)
    print("TrustGuard — Citation Quality Tester")
    print("=" * 70)

    print(
        "\nEach claim is compared ONLY against its cited source."
    )

    print(
        "\nClassification:"
        "\n  DIRECT     = source directly addresses the claim"
        "\n  INDIRECT   = source is related but does not directly address the claim"
        "\n  IRRELEVANT = source is not meaningfully related to the claim"
    )

    print(
        "\nScoring is CONTINUOUS from 0.0 to 1.0."
    )

    print(
        "\nThe final Citation Quality Score is the average "
        "of all individual claim-citation scores."
    )

    print("\n" + "=" * 70)

    try:

        number_of_claims = int(
            input(
                "\nHow many claims/citations? "
            )
        )

    except ValueError:

        print(
            "Please enter a valid number."
        )

        return

    claims = []

    for i in range(
        1,
        number_of_claims + 1,
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"CLAIM/CITATION PAIR {i}"
        )

        print(
            "-" * 70
        )

        claim = input(
            "LLM claim: "
        ).strip()

        source = input(
            "Cited source text: "
        ).strip()

        claims.append(
            {
                "claim_id": f"C{i}",
                "claim": claim,
                "citation_id": str(i),
                "source_name": "",
                "source_text": source,
            }
        )

    judge = MiniLMCitationJudge()

    report = judge.judge_claims(
        claims
    )

    # =============================================================
    # FINAL REPORT
    # =============================================================

    print("\n\n")

    print(
        "=" * 70
    )

    print(
        "TRUSTGUARD CITATION QUALITY REPORT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTotal claim-citation pairs: "
        f"{report['total_pairs']}"
    )

    print(
        f"\nDirect:                     "
        f"{report['counts']['DIRECT']}"
    )

    print(
        f"Indirect:                   "
        f"{report['counts']['INDIRECT']}"
    )

    print(
        f"Irrelevant:                 "
        f"{report['counts']['IRRELEVANT']}"
    )

    print(
        "\n" + "-" * 70
    )

    print(
        f"Citation Quality Score:     "
        f"{report['citation_quality_score']:.3f}"
    )

    print(
        f"Citation Quality:           "
        f"{report['citation_quality_percent']:.1f}%"
    )

    print(
        "-" * 70
    )

    # =============================================================
    # INDIVIDUAL RESULTS
    # =============================================================

    print(
        "\nINDIVIDUAL CLAIM-CITATION RESULTS"
    )

    print(
        "-" * 70
    )

    for i, judgement in enumerate(
        report["judgements"],
        start=1,
    ):

        print(
            f"\nPair {i}"
        )

        print(
            f"Claim:          "
            f"{judgement.claim_text}"
        )

        print(
            f"Classification: "
            f"{judgement.label}"
        )

        print(
            f"Score:          "
            f"{judgement.score:.3f}"
        )

        print(
            f"Similarity:     "
            f"{judgement.claim_source_similarity:.3f}"
        )

        print(
            f"Entailment:     "
            f"{judgement.entailment_probability:.3f}"
        )

        print(
            f"Contradiction:  "
            f"{judgement.contradiction_probability:.3f}"
        )

        print(
            f"Neutral:        "
            f"{judgement.neutral_probability:.3f}"
        )

        print(
            f"Rationale:      "
            f"{judgement.rationale}"
        )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()