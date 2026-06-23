from data_processing.chunking import chunk_text

from summerization.llama_summarizer import (
    summarize_chunk,
    zero_shot
)

from summerization.hierarchical_merge import (
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

    chunks = chunk_text(
        document,
        max_words=75
    )

    summaries = []

    for chunk in chunks:
        summaries.append(
            summarize_chunk(chunk)
        )

    return hierarchical_merge(
        summaries=summaries,
        original_document=document,
        chunks=chunks,
        method=method,
        integration=integration
    )