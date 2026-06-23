import os
from dotenv import load_dotenv
import pandas as pd

from evaluation.rouge_eval import RougeEvaluator
from evaluation.bertscore_eval import BertScoreEvaluator
from evaluation.alignscore_eval import AlignScoreEvaluator
from evaluation.summac_eval import SummaCEvaluator
from evaluation.prisma_eval import PRISMAEvaluator


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


##########################################
# Load summaries
##########################################

with open("outputs/generated_summary.txt", "r", encoding="utf-8") as f:
    generated_summary = f.read()

with open("outputs/reference_summary.txt", "r", encoding="utf-8") as f:
    reference_summary = f.read()


##########################################
# Initialize evaluators
##########################################

rouge = RougeEvaluator()

bertscore = BertScoreEvaluator()

alignscore = AlignScoreEvaluator()

summac = SummaCEvaluator()

prisma = PRISMAEvaluator(api_key)


##########################################
# ROUGE
##########################################

print("Running ROUGE...")

rouge_scores = rouge.evaluate(
    generated_summary,
    reference_summary
)


##########################################
# BERTScore
##########################################

print("Running BERTScore...")

bert_scores = bertscore.evaluate(
    generated_summary,
    reference_summary
)


##########################################
# AlignScore
##########################################

print("Running AlignScore...")

align_score = alignscore.evaluate(
    generated_summary,
    reference_summary
)


##########################################
# SummaC
##########################################

print("Running SummaC...")

summac_score = summac.evaluate(
    generated_summary,
    reference_summary
)


##########################################
# PRISMA
##########################################

print("Running PRISMA...")

prisma_scores = prisma.evaluate(
    generated_summary,
    reference_summary
)


##########################################
# Collect Results
##########################################

results = {

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


##########################################
# Display
##########################################

df = pd.DataFrame([results])

print("\n")
print(df)


##########################################
# Save
##########################################

os.makedirs("results", exist_ok=True)

df.to_csv(
    "results/evaluation_results.csv",
    index=False
)

print("\nResults saved to results/evaluation_results.csv")