from datasets import load_dataset

from src.summarize_pipeline import summarize_document

dataset = load_dataset(
    "allenai/multi_lexsum",
    name="v20230518",
    trust_remote_code=True
)

sample = dataset["test"][0]

document = "\n\n".join(sample["sources"][:1])
reference = sample["summary/long"]

print("Document ID:", sample["id"])

generated_summary = summarize_document(
    document=document,
    method="hmerge"      # or whatever method you are evaluating
)

print("\n===== GENERATED SUMMARY =====\n")
print(generated_summary)