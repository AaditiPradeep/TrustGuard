"""Ask a question, evaluate the local Qwen answer, and print full details."""

from pipeline import run_trustguard


question = input("Enter your medical question: ").strip()
if not question:
    raise SystemExit("A question is required.")

print("\n" + "=" * 80)
print("RUNNING TRUSTGUARD")
print("=" * 80)

result = run_trustguard(
    question=question,
    # answer is intentionally omitted: local Ollama generates it.
    top_k=3,
)

print("\n" + "=" * 80)
print("TRUSTGUARD EVALUATION REPORT")
print("=" * 80)
print(f"\nQuestion:\n{result.question}")
print(f"\nGenerated answer ({result.answer_origin}):\n{result.answer}")

print("\n" + "-" * 80)
print("METRIC SCORES")
print("-" * 80)
print(f"Evidence Support: {result.evidence_support['evidence_support_score']:.2f}/100")
print(f"Answer Stability: {result.answer_stability['stability_percent']:.2f}/100")
print(f"Citation Quality: {result.citation_quality['citation_quality_percent']:.2f}/100")

evidence = result.evidence_support
print("\n" + "-" * 80)
print("EVIDENCE SUPPORT DETAILS")
print("-" * 80)
print(f"Total claims: {evidence['total_claims']}")
print(f"Supported claims: {evidence['supported_claims']}")
print(f"Contradicted claims: {evidence['contradicted_claims']}")
print(f"Mixed claims: {evidence['mixed_claims']}")
print(f"Unverified claims: {evidence['unverified_claims']}")
print("\nClaim-level scores:")
for index, claim in enumerate(evidence["claim_scores"], start=1):
    print(f"{index}. {claim['claim_text']}")
    print(f"   Status: {claim['status']}")
    print(f"   Score: {claim['score']:.2f}/100")

citation = result.citation_quality
print("\n" + "-" * 80)
print("CITATION QUALITY DETAILS")
print("-" * 80)
print(f"Retrieved citations checked: {citation['citation_count']}")
print(f"Mapped claim-citation pairs: {citation['mapped_pair_count']}")
print(f"Citation coverage: {citation['citation_coverage_percent']:.2f}%")
print("\nCitation labels:")
for label, count in citation["counts"].items():
    print(f"  {label:<15}: {count}")

stability = result.answer_stability
print("\n" + "-" * 80)
print("ANSWER STABILITY DETAILS")
print("-" * 80)
print(f"Responses sampled: {stability['n_responses']}")
print(f"Semantic similarity: {stability['average_semantic_similarity']:.4f}")
print(f"Jaccard overlap: {stability['average_jaccard']:.4f}")
print(f"Hybrid stability score: {stability['stability_percent']:.2f}/100")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
