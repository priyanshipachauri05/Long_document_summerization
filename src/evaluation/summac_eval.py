from summac.model_summac import SummaCZS

class SummaCEvaluator:
    def __init__(self, granularity="sentence", model_name="vitc", device="cpu"):
        # Initialized once during class creation
        self.model = SummaCZS(
            granularity=granularity,
            model_name=model_name,
            device=device
        )

    def evaluate(self, generated_summary: str, source_context: str) -> float:
        """
        Evaluates factual consistency against the source document context.
        """
        result = self.model.score([source_context], [generated_summary])
        return round(float(result["scores"][0]), 4)