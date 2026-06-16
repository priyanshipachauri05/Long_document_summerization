from summerization.chunk_summerizer import summarize_chunk
def group_summaries(summaries, group_size=2):
    groups = []

    for i in range(0, len(summaries), group_size):
        groups.append(summaries[i:i + group_size])

    return groups


def merge_group(group):

    combined_text = " ".join(group)

    merged_summary = summarize_chunk(
        combined_text
    )

    return merged_summary


def hierarchical_merge(summaries):

    level = 1

    while len(summaries) > 1:

        print(f"\nLevel {level}")
        print(f"Number of summaries: {len(summaries)}")

        groups = group_summaries(summaries)

        merged_summaries = []


        for group in groups:

            print("\nGROUP TO MERGE:")
            for summary in group:
                print("-", summary)

            merged_summary = merge_group(group)

            print("\nMERGED RESULT:")
            print(merged_summary)

            merged_summaries.append(
                merged_summary
    )

        summaries = merged_summaries

        level += 1

    return summaries[0]
