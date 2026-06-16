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

def select_passages_with_coverage(
    attr_texts,
    response,
    k
):

    labels = extract_labels(response)

    label_counts = count_label_frequencies(
        labels
    )

    sorted_labels = sort_labels(
        label_counts
    )

    selected_passages = []

    # Step 1: Select passages based on citation frequency
    for label, count in sorted_labels:

        passages = get_passages_by_label(
            attr_texts,
            label
        )

        for passage in passages:

            if passage not in selected_passages:

                selected_passages.append(
                    passage
                )

    # If already have exactly k passages
    if len(selected_passages) >= k:

        return selected_passages[:k]

    # Step 2: Create coverage regions
    total_passages = len(attr_texts)

    coverage_points = []

    for i in range(k):

        point = (
            (2 * i + 1)
            * total_passages
            / (2 * k)
        )

        coverage_points.append(point)

    # Step 3: Remove already covered regions
    uncovered_points = coverage_points.copy()

    for passage in selected_passages:

        pos = passage["position"]

        closest = min(
            uncovered_points,
            key=lambda p: abs(p - pos)
        )

        uncovered_points.remove(
            closest
        )

        if not uncovered_points:
            break

    # Step 4: Fill remaining slots using coverage
    remaining_passages = [

        passage

        for passage in attr_texts

        if passage not in selected_passages
    ]

    while (
        len(selected_passages) < k
        and uncovered_points
    ):

        best_passage = None
        best_point = None
        best_distance = float("inf")

        for passage in remaining_passages:

            pos = passage["position"]

            closest_point = min(
                uncovered_points,
                key=lambda p: abs(p - pos)
            )

            distance = abs(
                closest_point - pos
            )

            if distance < best_distance:

                best_distance = distance
                best_passage = passage
                best_point = closest_point

        selected_passages.append(
            best_passage
        )

        remaining_passages.remove(
            best_passage
        )

        uncovered_points.remove(
            best_point
        )

    return selected_passages    

attr_texts = [
    {"label":"P1","text":"Passage 1","position":0},
    {"label":"P2","text":"Passage 2","position":1},
    {"label":"P3","text":"Passage 3","position":2},
    {"label":"P4","text":"Passage 4","position":3},
    {"label":"P5","text":"Passage 5","position":4},
]
response = """
The environment is important [P1].
Pollution harms biodiversity [P2].
Conservation protects future generations [P2].
Students should plant trees [P4].
"""
output=select_passages_with_coverage(attr_texts,response,k=3)
print(output)