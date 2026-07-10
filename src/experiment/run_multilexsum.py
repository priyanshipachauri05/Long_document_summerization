from datasets import load_dataset
import pandas as pd
import os
import traceback

from src.summarize_pipeline import summarize_document
from src.evaluation.evaluate_all import evaluate_all

# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_DOCUMENTS = 1          # Change as needed
METHOD = "retrieve"
INTEGRATION="support"


# ==========================================================
# Load Dataset
# ==========================================================

print("Loading MultiLexSum dataset...")

dataset = load_dataset(
    "allenai/multi_lexsum",
    name="v20230518",
    trust_remote_code=True
)

samples = dataset["test"].select(range(NUM_DOCUMENTS))

os.makedirs("outputs", exist_ok=True)
os.makedirs("results", exist_ok=True)

csv_path = "results/smoke_test_results.csv"

# ==========================================================
# Create CSV if it doesn't exist
# ==========================================================

if not os.path.exists(csv_path):
    pd.DataFrame().to_csv(csv_path, index=False)

# ==========================================================
# Process Documents
# ==========================================================

for i, sample in enumerate(samples, start=1):

    print("\n" + "=" * 70)
    print(f"Document {i}/{NUM_DOCUMENTS}")
    print(f"ID : {sample['id']}")
    print("=" * 70)

    try:

        document = "\n\n".join(sample["sources"])
        reference = sample["summary/long"]

        # --------------------------------------------------
        # Summarization
        # --------------------------------------------------

        print("Generating summary...")

        generated_summary = summarize_document(
            document=document,
            method=METHOD,
            integration=INTEGRATION
        )

        print("✓ Summary generated")
        # --------------------------------------------------
        # Save text outputs
        # --------------------------------------------------

        folder = os.path.join("outputs", METHOD, sample["id"])
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "source_document.txt"), "w", encoding="utf-8") as f:
            f.write(document)

        with open(os.path.join(folder, "reference_summary.txt"), "w", encoding="utf-8") as f:
            f.write(reference)

        with open(os.path.join(folder, "generated_summary.txt"), "w", encoding="utf-8") as f:
            f.write(generated_summary)

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        print("Running evaluation...")

        scores = evaluate_all(
            generated_summary=generated_summary,
            reference_summary=reference,
            source_context=document
        )

        print("✓ Evaluation complete")

        

        # --------------------------------------------------
        # Save Results
        # --------------------------------------------------

        row = {
            "Document_ID": sample["id"],
            "Method": METHOD,
            "Generated Summary": generated_summary,
            "Reference Summary": reference
        }

        row.update(scores)

        row_df = pd.DataFrame([row])

        # Save one CSV per method
        csv_path = os.path.join("results", f"{METHOD}_results.csv")

        # Create the file if it doesn't exist
        if not os.path.exists(csv_path):
            row_df.to_csv(csv_path, index=False)

        # Otherwise append
        else:
            row_df.to_csv(csv_path, mode="a", header=False, index=False)

        print(f"✓ Results appended to {csv_path}")

    except Exception as e:

            print(f"\nError processing document {sample['id']}")
            print(e)

            traceback.print_exc()

            continue

    print("\n" + "=" * 70)
    print("Smoke test completed!")
    print(f"Results saved to: {csv_path}")
    print("=" * 70)