def support_summary(summary, retrieved_contexts):

    context = " ".join(retrieved_contexts)

    return f"""
SUMMARY:
{summary}

SUPPORTING CONTEXT:
{context}
"""
summary = "Alice visited Paris"

contexts = [
    "Alice went to Paris",
    "Alice enjoyed French food"
]

print(support_summary(summary, contexts))