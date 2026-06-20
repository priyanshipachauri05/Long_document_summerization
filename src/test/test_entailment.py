from dotenv import load_dotenv
import os
from evaluation.prisma_eval import PRISMAEvaluator

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")
evaluator=PRISMAEvaluator(api_key)
statement="John travelled to London."

reference="""
John travelled to Paris with Mary.
They visited the Eiffel Tower.
"""

result = evaluator.check_entailment(statement,reference)

print(result)