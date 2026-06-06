from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, passages):

        self.passages = passages

        tokenized_passages = [passage.split() for passage in passages]

        self.bm25 = BM25Okapi(tokenized_passages)

    def retrieve(self, query, top_k=2):

        tokenized_query = query.split()

        results = self.bm25.get_top_n(
            tokenized_query,
            self.passages,
            n=top_k
        )

        return results
    
if __name__ == "__main__":

    passages = [
        "Alice went to Paris and visited the Eiffel Tower.",
        "Bob bought a new laptop for college.",
        "Alice enjoyed French food and museums.",
        "The weather was sunny during the trip."
    ]

    retriever = BM25Retriever(passages)

    query = "Alice visited Paris"

    results = retriever.retrieve(query)

    print("Top Passages:\n")

    for passage in results:
        print(passage)