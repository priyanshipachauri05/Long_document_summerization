from dotenv import load_dotenv
import os

from evaluation.prisma_eval import PRISMAEvaluator

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

evaluator = PRISMAEvaluator(api_key)

generated_summary = """
John travelled to Paris.
John bought a souvenir.
"""

reference_summary = """
John travelled to Paris with Mary.
They visited the Eiffel Tower.
John bought a souvenir.
"""

result = evaluator.evaluate(
    generated_summary,
    reference_summary
)

print(result)