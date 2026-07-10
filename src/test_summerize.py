import os
import pandas as pd

from src.summarize_pipeline import summarize_document
from src.evaluation.evaluate_all import evaluate_all


# =====================================================
# TOY DOCUMENT
# =====================================================

document = """
Artificial Intelligence is transforming healthcare.

Hospitals use AI to detect diseases earlier.

Machine learning models assist doctors in diagnosis.

AI also helps hospitals optimize scheduling and hospital resources.

Researchers continue improving AI systems to improve patient care.
"""

reference_summary = """
Artificial Intelligence improves healthcare by assisting disease detection,
medical diagnosis, hospital resource management, and patient care.
"""


# =====================================================
# RUN SUMMARIZATION
# =====================================================

generated_summary = summarize_document(
    document=document,
    method="hmerge",          # change if required
    integration="none"
)


# =====================================================
# RUN EVALUATION
# =====================================================

scores = evaluate_all(
    generated_summary=generated_summary,
    reference_summary=reference_summary,
    source_context=document
)


# =====================================================
# PREPARE RESULT ROW
# =====================================================

row = {
    "document": document,
    "reference_summary": reference_summary,
    "generated_summary": generated_summary,
    "method": "hmerge",
    "integration": "none"
}

# Flatten nested dictionaries
for metric_name, metric_values in scores.items():

    if isinstance(metric_values, dict):

        for key, value in metric_values.items():
            row[f"{metric_name}_{key}"] = value

    else:
        row[metric_name] = metric_values


# =====================================================
# SAVE CSV
# =====================================================

os.makedirs("results", exist_ok=True)

csv_file = "results/toy_pipeline_results.csv"

df = pd.DataFrame([row])

if os.path.exists(csv_file):
    existing = pd.read_csv(csv_file)
    df = pd.concat([existing, df], ignore_index=True)

df.to_csv(csv_file, index=False)

print("\n==============================")
print("FINAL SUMMARY")
print("==============================")
print(generated_summary)

print("\n==============================")
print("EVALUATION SCORES")
print("==============================")

for metric, value in scores.items():
    print(metric, ":", value)

print(f"\nResults saved to {csv_file}")