from dotenv import load_dotenv
import os

from evaluation.prisma_eval import PRISMAEvaluator


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

evaluator = PRISMAEvaluator(api_key)


summary = """
John travelled to Paris with Mary.
They visited the Eiffel Tower.
John bought a souvenir.
"""

facts = evaluator.extract_atomic_facts(summary)

print("\n===== FINAL ATOMIC FACTS =====\n")

for i, fact in enumerate(facts, start=1):
    print(f"{i}. {fact}")