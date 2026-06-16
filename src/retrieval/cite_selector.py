import re
from collections import Counter

def extract_labels(response):

    return re.findall(
        r"\[(.*?)\]",
        response
    )
def count_label_frequencies(labels):

    return Counter(labels)
def sort_labels(label_counts):

    return sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
def get_passages_by_label(
    attr_texts,
    label
):

    passages = []

    for item in attr_texts:

        if item["label"] == label:
            passages.append(item)

    return passages
def select_top_cited_passages(
    attr_texts,
    response,
    k
):

    labels = extract_labels(
        response
    )

    label_counts = count_label_frequencies(
        labels
    )

    sorted_labels = sort_labels(
        label_counts
    )

    selected_passages = []

    for label, count in sorted_labels:

        passages = get_passages_by_label(
            attr_texts,
            label
        )

        for passage in passages:

            selected_passages.append(
                passage
            )

            if len(selected_passages) == k:

                return selected_passages

    return selected_passages