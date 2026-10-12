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
if result.source_reliability["rave_percent"] is None:
    print("Source Reliability (RAVE): unavailable")
else:
    print(f"Source Reliability (RAVE): {result.source_reliability['rave_percent']:.2f}/100")
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
print("\nClaim verification labels:")
for index, claim in enumerate(evidence["claim_scores"], start=1):
    print(f"{index}. {claim['claim_text']}")
    print(f"   Status: {claim['status']}")

rave = result.source_reliability
print("\n" + "-" * 80)
print("SOURCE RELIABILITY (RAVE) DETAILS")
print("-" * 80)
print(f"RAVE computed for {rave['claims_with_rave']} of {rave['total_claims']} claims ({rave['coverage_percent']:.2f}%)")
for index, claim in enumerate(rave["claim_results"], start=1):
    score = "unavailable" if claim["rave_score"] is None else f"{claim['rave_score'] * 100:.2f}/100"
    sources = ", ".join(
        f"{source['source_name']}: {source['status']}"
        for source in claim["source_results"]
    ) or "no usable sources"
    print(f"{index}. {claim['status']} — {score}")
    print(f"   Sources: {sources}")
print(rave["note"])
citation = result.citation_quality
print("\n" + "-" * 80)
print("CITATION QUALITY DETAILS")
print("-" * 80)
print(f"Method: {citation.get('method', 'minilm_deberta_nli')}")
print(f"Retrieved citations checked: {citation['citation_count']}")
print(f"Mapped claim-citation pairs: {citation['mapped_pair_count']}")
print(f"Citation coverage: {citation['citation_coverage_percent']:.2f}%")
"""print("\nCitation labels:")
for label, count in citation["counts"].items():
    print(f"  {label:<15}: {count}")"""


stability = result.answer_stability
print("\n" + "-" * 80)
print("ANSWER STABILITY DETAILS")
print("-" * 80)
print(f"Responses sampled: {stability['n_responses']}")
if stability["average_semantic_similarity"] is None:
    print("Semantic similarity: unavailable; using Jaccard fallback")
else:
    print(f"Semantic similarity: {stability['average_semantic_similarity']:.4f}")
print(f"Jaccard overlap: {stability['average_jaccard']:.4f}")
print(f"Hybrid stability score: {stability['stability_percent']:.2f}/100")

lay_report = result.trust_report
print("\n" + "-" * 80)
print("PLAIN-LANGUAGE TRUST SUMMARY")
print("-" * 80)
print(lay_report["plain_language_summary"])
print(f"\nRecommendation: {lay_report['recommendation']}")
print(lay_report["recommendation_reason"])
if lay_report["why"]:
    print("Why:")
    for reason in lay_report["why"]:
        print(f"- {reason}")

print("\nHow the answer was checked:")
for metric in lay_report["metric_scores"]:
    score = "unavailable" if metric["score"] is None else f"{metric['score']:.2f}/100"
    coverage = (
        f" RAVE was calculable for {metric['coverage']:.1f}% of claims."
        if metric["name"] == "Source reliability" and metric["coverage"] is not None else ""
    )
    print(f"- {metric['name']}: {score}. {metric['explanation']}{coverage}")

print("\nClaims in the answer:")
for index, claim in enumerate(lay_report["claims"], start=1):
    print(f"{index}. {claim['claim']}")
    print(f"   Result: {claim['status']} ({claim['category']} trust; score: {claim['trust_score']})")
    print(f"   Evidence support: {claim['metric_scores']['evidence_support']}")
    print(f"   Source reliability: {claim['metric_scores']['source_reliability']}")
    print(f"   Citation quality: {claim['metric_scores']['citation_quality']}")
    print(f"   Answer consistency: {claim['metric_scores']['answer_consistency']} (whole answer)")
    print(f"   {claim['explanation']}")

if lay_report["missing_information"]:
    print("\nUnavailable checks: " + ", ".join(lay_report["missing_information"]))

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
