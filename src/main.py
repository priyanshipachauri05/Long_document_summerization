from data_processing.chunking import chunk_text
from retrieval.bm25_retriever import BM25Retriever
from summerization.support_pipeline import support_summary
from summerization.replace_pipeline import replace_summary
from summerization.hierarchical_merge import hierarchical_merge


def main():

    print("===== LONG DOCUMENT SUMMARIZATION =====\n")

    # Step 1: Sample document
    document = """
    Alice went to Paris and visited the Eiffel Tower.
    She enjoyed French food and museums.
    Bob bought a new laptop for college.
    The weather was sunny during the trip.
    """

    print("Document Loaded\n")

    # Step 2: Chunking
    chunks = chunk_text(document, chunk_size=10)

    print(f"Number of Chunks: {len(chunks)}\n")

    # Step 3: Placeholder summaries
    summaries = []

    for i, chunk in enumerate(chunks):
        summaries.append(f"Summary of chunk {i+1}")

    print("Chunk Summaries Generated\n")

    # Step 4: BM25 Retrieval
    retriever = BM25Retriever(chunks)

    enhanced_summaries = []

    for summary in summaries:

        contexts = retriever.retrieve(summary, top_k=2)

        enhanced_summary = support_summary(
            summary,
            contexts
        )

        enhanced_summaries.append(enhanced_summary)

    print("Context Retrieved and Support Applied\n")

    # Step 5: Hierarchical Merge
    final_summary = hierarchical_merge(
        enhanced_summaries
    )

    print("\n===== FINAL SUMMARY =====\n")
    print(final_summary)


if __name__ == "__main__":
    main()