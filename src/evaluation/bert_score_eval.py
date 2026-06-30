from bert_score import score

class BertScoreEvaluator:
    def __init__(self, model_type="roberta-large", lang="en", rescale_with_baseline=True):
        self.model_type = model_type
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline

    def evaluate(self, generated_summary: str, reference_summary: str) -> dict:
        P, R, F1 = score(
            [generated_summary],
            [reference_summary],
            model_type=self.model_type,  
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline
        )
        return {
            "precision": round(P.item(), 4),
            "recall": round(R.item(), 4),
            "f1": round(F1.item(), 4)
        }