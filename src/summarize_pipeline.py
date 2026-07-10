from src.data_processing.chunking import chunk_text

from src.summerization.llama_summarizer import (
    summarize_chunk,
    zero_shot
)

from src.summerization.hierarchical_merge import (
    hierarchical_merge
)

# ==========================================
# CONFIGURATION
# ==========================================

def summarize_document(
        document,
        method="zeroshot",
        integration="none"
):

    valid_methods = [
        "zeroshot",
        "hmerge",
        "retrieve",
        "extract",
        "cite"
    ]

    if method not in valid_methods:
        raise ValueError(f"Unknown method: {method}")

    if method == "zeroshot":
        return zero_shot(document)

    if method == "hmerge":
        integration = None

    elif integration not in ["support", "replace"]:
        raise ValueError(
            "integration must be 'support' or 'replace'"
        )
    print("Entered summarize_document()")
    chunks = chunk_text(
        document,
        max_words=300
    )
    print(f"Number of chunks: {len(chunks)}")
    summaries = []

    for i, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {i}/{len(chunks)}")

        summaries.append(
            summarize_chunk(chunk)
    )
    print("Chunk summarization complete")

    print("Starting hierarchical merge...")
    print(method)
    print(integration)
    return hierarchical_merge(
        summaries=summaries,
        original_document=document,
        chunks=chunks,
        method=method,
        integration=integration
    )
