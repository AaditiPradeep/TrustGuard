"""
Manual test for Answer Stability across a few different scenarios, so you
can see the score behave sensibly, not just run once on canned data.

Run with:  python test_stability_manual.py
"""

from metrics.answer_stability import compute_answer_stability

SCENARIOS = {
    "Identical responses (stable model, or temperature=0)": [
        "You should stop metformin before a contrast imaging procedure.",
        "You should stop metformin before a contrast imaging procedure.",
        "You should stop metformin before a contrast imaging procedure.",
    ],
    "Same answer, different phrasing (should be moderately stable)": [
        "You should stop metformin before a contrast imaging procedure.",
        "Metformin should be discontinued prior to contrast dye imaging.",
        "Stop taking metformin ahead of any contrast-based imaging test.",
    ],
    "Model contradicts itself across runs (should be unstable)": [
        "Metformin should be stopped before contrast imaging due to lactic acidosis risk.",
        "Metformin is generally safe with contrast dye and doesn't need to be stopped.",
        "There's no established interaction between metformin and contrast agents.",
    ],
    "Completely unrelated responses (edge case, should be near 0)": [
        "Bananas are a tropical fruit rich in potassium.",
        "The stock market rallied after strong earnings reports today.",
        "Quantum computers rely on qubits rather than classical bits.",
    ],
}


def main():
    for label, responses in SCENARIOS.items():
        result = compute_answer_stability(responses)
        print("=" * 100)
        print(label)
        print("=" * 100)
        for p in result.pairwise:
            print(
                f"  resp[{p.i}] vs resp[{p.j}]: Jaccard={p.jaccard:.3f}, "
                f"semantic={p.semantic:.3f}, hybrid={p.hybrid:.3f}"
            )
        print(f"  -> average_stability = {result.average_stability:.3f}\n")


if __name__ == "__main__":
    main()
