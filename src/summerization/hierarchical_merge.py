from src.summerization.llama_summarizer import call_llm

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.extractive_selector import extract_context
from src.retrieval.cite_selector import select_passages_with_coverage


# ==========================================
# GROUP SUMMARIES
# ==========================================

def group_summaries(summaries, group_size=2):
    groups = []

    for i in range(0, len(summaries), group_size):
        groups.append(summaries[i:i + group_size])

    return groups


# ==========================================
# BUILD CONTEXT
# ==========================================

def get_context(
        group,
        method,
        original_document,
        chunks
):

    if method == "retrieve":

        retriever = BM25Retriever(chunks)

        query = " ".join(group)

        return retriever.retrieve(
            query,
            top_k=2
        )

    elif method == "extract":

        return extract_context(
            original_document,
            k=2
        )

    elif method == "cite":

        attr_texts = []

        for i, chunk in enumerate(chunks):

            attr_texts.append(
                {
                    "label": f"P{i+1}",
                    "text": chunk,
                    "position": i
                }
            )

        # Placeholder response.
        # Later this can be replaced with actual cited summaries.
        response = ""

        for i in range(len(chunks)):
            response += f"Chunk {i+1}. [P{i+1}] "

        selected = select_passages_with_coverage(
            attr_texts,
            response,
            k=2
        )

        return [
            passage["text"]
            for passage in selected
        ]

    return None


# ==========================================
# MERGE ONE GROUP
# ==========================================

def merge_group(
        group,
        method,
        integration,
        contexts=None,
        max_words=300
):

    summaries = "\n".join(group)

    # -------------------------
    # HMERGE
    # -------------------------

    if method == "hmerge":

        prompt = f"""
Below are several summaries of different parts
of a document:

---
{summaries}
---

Merge the given summaries into one
single summary containing all key
information.
The merged summary must be at most {max_words} words.

Do not mention words like
"document" or "summary".
"""

    # -------------------------
    # SUPPORT
    # -------------------------

    elif integration == "support":

        context = "\n".join(contexts)

        prompt = f"""
Below are several summaries of different parts
of a document:

---
{summaries}
---

Below are the supporting contexts:

---
{context}
---

Merge the given summaries into one
single summary containing all key
information.

The gist should come ONLY from
the summaries.

Use the supporting contexts ONLY
to verify factual correctness.

Do not mention words like
"document",
"context",
or
"summary".
"""

    # -------------------------
    # REPLACE
    # -------------------------

    elif integration == "replace":

        context = "\n".join(contexts)

        prompt = f"""
Below are relevant contexts extracted
from different parts of a document:

---
{context}
---

Generate one concise summary
containing all important information.

Do not mention words like
"document",
"context",
or
"summary".
"""

    else:
        raise ValueError("Invalid integration type.")
    print("\n========== MERGE DEBUG ==========")
    print(f"Number of summaries: {len(group)}")
    print(f"Prompt length (words): {len(prompt.split())}")
    print("===============================\n")

    return call_llm(prompt)


# ==========================================
# HIERARCHICAL MERGING
# ==========================================

def hierarchical_merge(
        summaries,
        original_document,
        chunks,
        method="hmerge",
        integration=None
):

    level = 1

    while len(summaries) > 1:

        print(f"\n========== LEVEL {level} ==========")

        groups = group_summaries(summaries)

        merged_summaries = []

        for idx, group in enumerate(groups):

            print(f"\nMerging Group {idx+1}")

            contexts = None

            if method != "hmerge":

                contexts = get_context(
                    group,
                    method,
                    original_document,
                    chunks
                )

            merged = merge_group(
                group,
                method,
                integration,
                contexts
            )

            print("\nMerged Summary:")
            print(merged)

            merged_summaries.append(merged)

        summaries = merged_summaries

        level += 1

    return summaries[0]