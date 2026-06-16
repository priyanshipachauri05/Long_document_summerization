from data_processing.chunking import chunk_text

from retrieval.bm25_retriever import BM25Retriever
from retrieval.extractive_selector import extract_context
from retrieval.cite_selector import select_passages_with_coverage

from summerization.llama_summarizer import summarize_chunk
from summerization.support_pipeline import support_summary
from summerization.replace_pipeline import replace_summary
from summerization.hierarchical_merge import hierarchical_merge


# ==========================================
# CONFIGURATION
# ==========================================

METHOD = "retrieve"
# Options:
# "extract"
# "retrieve"
# "cite"

INTEGRATION = "support"
# Options:
# "support"
# "replace"


def main():

    print("===== LONG DOCUMENT SUMMARIZATION =====\n")

    document = """
    Our environment is made up of everything living and non-living around us—plants, animals, humans, water, soil, air, and even buildings.
    Both natural and man-made elements are part of the environment.
    The environment is very important because it provides us with everything necessary for survival, like clean air, safe water, fertile land, and biodiversity.
    Sadly, pollution, deforestation, and overuse of resources are harming the environment.
    This leads to problems like climate change, water shortages, and extinction of species.
    Everyone, especially students, should help protect the environment by planting trees, avoiding plastic, saving water and electricity, and joining awareness campaigns like World Environment Day.
    By caring for our surroundings, we protect our health and the future of our planet.
    """

    print("Document Loaded\n")

    # ==========================================
    # CHUNKING
    # ==========================================

    chunks = chunk_text(
        document,
        max_words=75
    )

    print(f"Number of Chunks: {len(chunks)}\n")

    # ==========================================
    # CHUNK SUMMARIZATION
    # ==========================================

    summaries = []

    for chunk in chunks:

        summary = summarize_chunk(chunk)

        summaries.append(summary)

    print("Chunk Summaries Generated\n")

    # ==========================================
    # CONTEXT SELECTION
    # ==========================================

    enhanced_summaries = []

    if METHOD == "extract":

        print("Using EXTRACT method\n")

        contexts = extract_context(
            document,
            k=3
        )

        for summary in summaries:

            if INTEGRATION == "support":

                enhanced_summary = support_summary(
                    summary,
                    contexts
                )

            else:

                enhanced_summary = replace_summary(
                    summary,
                    contexts
                )

            enhanced_summaries.append(
                enhanced_summary
            )

    elif METHOD == "retrieve":

        print("Using RETRIEVE method\n")

        retriever = BM25Retriever(chunks)

        for summary in summaries:

            contexts = retriever.retrieve(
                summary,
                top_k=2
            )

            if INTEGRATION == "support":

                enhanced_summary = support_summary(
                    summary,
                    contexts
                )

            else:

                enhanced_summary = replace_summary(
                    summary,
                    contexts
                )

            enhanced_summaries.append(
                enhanced_summary
            )

    elif METHOD == "cite":

        print("Using CITE method\n")

        attr_texts = []

        for i, chunk in enumerate(chunks):

            attr_texts.append(
                {
                    "label": f"P{i+1}",
                    "text": chunk,
                    "position": i
                }
            )

        # Dummy cited response
        response = ""

        for i in range(len(chunks)):

            response += (
                f"Information from chunk {i+1} "
                f"[P{i+1}]. "
            )

        selected_passages = select_passages_with_coverage(
            attr_texts,
            response,
            k=2
        )

        contexts = []

        for passage in selected_passages:

            contexts.append(
                passage["text"]
            )

        for summary in summaries:

            if INTEGRATION == "support":

                enhanced_summary = support_summary(
                    summary,
                    contexts
                )

            else:

                enhanced_summary = replace_summary(
                    summary,
                    contexts
                )

            enhanced_summaries.append(
                enhanced_summary
            )

    else:

        raise ValueError(
            "Invalid METHOD selected."
        )

    print("Context Selection Complete\n")

    # ==========================================
    # HIERARCHICAL MERGE
    # ==========================================

    final_summary = hierarchical_merge(
        enhanced_summaries
    )

    # ==========================================
    # OUTPUT
    # ==========================================

    print("\n===== FINAL SUMMARY =====\n")

    print(final_summary)


if __name__ == "__main__":
    main()