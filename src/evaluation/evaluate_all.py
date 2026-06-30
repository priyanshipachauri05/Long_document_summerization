import os
from dotenv import load_dotenv
import pandas as pd

from evaluation.rouge_eval import RougeEvaluator
from evaluation.bert_score_eval import BertScoreEvaluator
from evaluation.alignscore_eval import AlignScoreEvaluator
from evaluation.summac_eval import SummaCEvaluator
from evaluation.prisma_eval import PRISMAEvaluator

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

##########################################
# Load Text Data
##########################################

with open("outputs/generated_summary.txt", "r", encoding="utf-8") as f:
    generated_summary = f.read()

with open("outputs/reference_summary.txt", "r", encoding="utf-8") as f:
    reference_summary = f.read()

# Factuality metrics need the original source text to check for hallucinations
# Ensure you have the original input document saved here:
source_document_path = "outputs/source_document.txt" 
if os.path.exists(source_document_path):
    with open(source_document_path, "r", encoding="utf-8") as f:
        source_context = f.read()
else:
    print(f"⚠️ {source_document_path} missing! Defaulting factuality check to reference_summary.")
    source_context = reference_summary


##########################################
# Initialize evaluators
##########################################

rouge = RougeEvaluator()
bertscore = BertScoreEvaluator()

# Set up AlignScore pointing to your path
alignscore = AlignScoreEvaluator(
    ckpt_path="./checkpoints/AlignScore-base.ckpt", 
    use_gpu=False
)

summac = SummaCEvaluator()
prisma = PRISMAEvaluator(api_key)


##########################################
# Run Metrics
##########################################

print("Running ROUGE...")
rouge_scores = rouge.evaluate(generated_summary, reference_summary)

print("Running BERTScore...")
bert_scores = bertscore.evaluate(generated_summary, reference_summary)

print("Running AlignScore...")
align_score = alignscore.evaluate(generated_summary, source_context)

print("Running SummaC...")
summac_score = summac.evaluate(generated_summary, source_context)

print("Running PRISMA...")
prisma_scores = prisma.evaluate(generated_summary, reference_summary)


##########################################
# Collect & Save Results
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

df = pd.DataFrame([results])
print("\n")
print(df)

os.makedirs("results", exist_ok=True)
df.to_csv("results/evaluation_results.csv", index=False)
print("\nResults saved to results/evaluation_results.csv")