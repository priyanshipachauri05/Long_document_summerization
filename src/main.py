from src.data_processing.document_loader import load_document
from summarize_pipeline import summarize_document

document = load_document("documents/sample.txt")

summary = summarize_document(
    document,
    method="zero_shot",
    integration="none"
)

print(summary)