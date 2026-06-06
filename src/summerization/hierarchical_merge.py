def group_summaries(summaries, group_size=2):
    groups = []

    for i in range(0, len(summaries), group_size):
        groups.append(summaries[i:i + group_size])

    return groups


def merge_group(group):
    """
    Temporary merge function.
    Later this will call an LLM to generate
    a merged summary.
    """

    return " ".join(group)


def hierarchical_merge(summaries):

    level = 1

    while len(summaries) > 1:

        print(f"\nLevel {level}")
        print(f"Number of summaries: {len(summaries)}")

        groups = group_summaries(summaries)

        merged_summaries = []

        for group in groups:

            merged_summary = merge_group(group)

            merged_summaries.append(
                merged_summary
            )

        summaries = merged_summaries

        level += 1

    return summaries[0]
if __name__ == "__main__":

    summaries = [
        "Summary 1",
        "Summary 2",
        "Summary 3",
        "Summary 4",
        "Summary 5"
    ]

    final_summary = hierarchical_merge(
        summaries
    )

    print("\nFinal Summary:")
    print(final_summary)