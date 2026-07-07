from summac.model_summac import SummaCZS


class SummaCEvaluator:

    def __init__(
        self,
        granularity="sentence",
        model_name="mnli",      # Much lighter than vitc
        device="cpu",
        chunk_size=1000
    ):

        print("Loading SummaC model...")

        self.model = SummaCZS(
            granularity=granularity,
            model_name=model_name,
            device=device
        )

        self.chunk_size = chunk_size

        print("SummaC model loaded.\n")

    def evaluate(self, generated_summary, source_context):

        words = source_context.split()

        chunks = [
            " ".join(words[i:i+self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]

        print(f"Evaluating {len(chunks)} chunks...\n")

        scores = []

        for i, chunk in enumerate(chunks):

            print(f"Chunk {i+1}/{len(chunks)}")

            result = self.model.score(
                [chunk],
                [generated_summary]
            )

            scores.append(result["scores"][0])

        final_score = sum(scores) / len(scores)

        return round(final_score, 4)