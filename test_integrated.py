from pipeline import run_trustguard


question = "What should a patient know about warfarin?"

answer = """
Warfarin is used to help prevent harmful blood clots.
Warfarin can interact with NSAIDs and increase bleeding risk.
Warfarin is completely safe to take with ibuprofen.
Patients taking warfarin never need monitoring.
Warfarin commonly causes kidney stones.
"""


result = run_trustguard(
    question=question,
    answer=answer,
    top_k=5,
)

print("\n\n==============================")
print("TRUSTGUARD RESULT")
print("==============================")

print(result.to_dict())