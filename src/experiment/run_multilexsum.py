from datasets import load_dataset
import pandas as pd
import os

from src.summarize_pipeline import summarize_document
from src.evaluation.evaluate_all import evaluate_all


# ==========================================================
# Load MultiLexSum Dataset
# ==========================================================

print("Loading MultiLexSum dataset...")

dataset = load_dataset(
    "allenai/multi_lexsum",
    name="v20230518",
    trust_remote_code=True
)

sample = dataset["test"][0]

document = "\n\n".join(sample["sources"])
reference = sample["summary/long"]

print("\n==========================================")
print(f"Document ID : {sample['id']}")
print(f"Source Documents : {len(sample['sources'])}")
print(f"Document Length : {len(document.split())} words")
print("==========================================\n")


# ==========================================================
# Generate Summary
# ==========================================================

print("Generating summary...\n")

generated_summary = summarize_document(
    document=document,
    method="hmerge"
)

print("\nSummary generation completed.\n")


# ==========================================================
# Save Outputs
# ==========================================================

os.makedirs("outputs", exist_ok=True)

with open(
    "outputs/source_document.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(document)

with open(
    "outputs/reference_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(reference)

with open(
    "outputs/generated_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(generated_summary)

print("Saved source, reference and generated summaries.\n")


# ==========================================================
# Evaluate
# ==========================================================

print("Running evaluation metrics...\n")

scores = evaluate_all(
    generated_summary=generated_summary,
    reference_summary=reference,
    source_context=document
)

print("\nEvaluation completed.\n")


# ==========================================================
# Save Results
# ==========================================================

os.makedirs("results", exist_ok=True)

df = pd.DataFrame([scores])

df.to_csv(
    "results/evaluation_results.csv",
    index=False
)

print("========== FINAL RESULTS ==========\n")
print(df)

print("\nResults saved to:")
print("results/evaluation_results.csv")