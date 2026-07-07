import os
from dotenv import load_dotenv

from src.evaluation.rouge_eval import RougeEvaluator
from src.evaluation.bert_score_eval import BertScoreEvaluator
from src.evaluation.alignscore_eval import AlignScoreEvaluator
from src.evaluation.summac_eval import SummaCEvaluator
from src.evaluation.prisma_eval import PRISMAEvaluator


load_dotenv()


def evaluate_all(
    generated_summary,
    reference_summary,
    source_context
):

    api_key = os.getenv("GEMINI_API_KEY")

    rouge = RougeEvaluator()
    bertscore = BertScoreEvaluator()

    alignscore = AlignScoreEvaluator(
        ckpt_path="./checkpoints/AlignScore-base.ckpt",
        use_gpu=False
    )

    summac = SummaCEvaluator()

    prisma = PRISMAEvaluator(api_key)

    print("Running ROUGE...")
    rouge_scores = rouge.evaluate(
        generated_summary,
        reference_summary
    )

    print("Running BERTScore...")
    bert_scores = bertscore.evaluate(
        generated_summary,
        reference_summary
    )

    print("Running AlignScore...")
    align_score = alignscore.evaluate(
        generated_summary,
        source_context
    )

    print("Running SummaC...")
    summac_score = summac.evaluate(
        generated_summary,
        source_context
    )

    print("Running PRISMA...")
    prisma_scores = prisma.evaluate(
        generated_summary,
        reference_summary
    )

    return {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
        "BERTScore": bert_scores["f1"],
        "AlignScore": align_score,
        "SummaC": summac_score,
        "Fact Precision": prisma_scores["fact_precision"],
        "Fact Recall": prisma_scores["fact_recall"],
        "PRISMA": prisma_scores["prisma_score"]
    }