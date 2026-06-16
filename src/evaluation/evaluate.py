from rouge_score import rouge_scorer


def evaluate_summary(generated_summary, reference_summary):

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference_summary,
        generated_summary
    )

    print("\n===== ROUGE SCORES =====\n")

    print(
        f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}"
    )

    print(
        f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}"
    )

    print(
        f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}"
    )

    return scores
if __name__ == "__main__":

    generated = """
    The environment provides resources but faces threats from pollution.
    """

    reference = """
    Pollution harms the environment, making conservation efforts important.
    """

    evaluate_summary(
        generated,
        reference
    )