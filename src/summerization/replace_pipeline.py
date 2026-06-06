def replace_summary(summary, retrieved_contexts):
    return " ".join(retrieved_contexts)
summary = "Alice visited Paris"

contexts = [
    "Alice went to Paris",
    "Alice enjoyed French food"
]

print(replace_summary(summary, contexts))