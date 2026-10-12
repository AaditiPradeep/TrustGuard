"""Ask a question; if the answer is low-trust, regenerate it (max 2 times).

Usage:  python run_regeneration_test.py
Optionally paste your own (possibly wrong) answer to force a regeneration.
"""

import json

from regeneration.regeneration_controller import run_with_regeneration

question = input("Enter your medical question: ").strip()
if not question:
    raise SystemExit("A question is required.")
answer = input("Optional: paste an answer to evaluate (Enter = let Qwen answer): ").strip() or None

outcome = run_with_regeneration(question, answer, top_k=3)

print("\n" + "=" * 80)
print("REGENERATION SUMMARY")
print("=" * 80)
print(f"Status: {outcome.status}  ({outcome.stop_reason})")
print(f"Needs human review: {outcome.needs_human_review}")
if outcome.safety_gate_triggered:
    print("Clinical safety gate triggered on a high-risk claim.")
for a in outcome.attempts:
    label = "original" if a.attempt == 0 else f"regeneration {a.attempt}"
    print(f"\n[{label}] score={a.composite_score} decision={a.decision} flagged={a.flagged}")
    print(a.answer)
print("\nFINAL ANSWER:\n" + outcome.final_answer)
if outcome.needs_human_review:
    print("\n>> Route this response to a human reviewer before showing it to a user.")
