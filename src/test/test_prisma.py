import os
from src.evaluation.prisma_eval import PRISMAEvaluator  # Assuming the code above is saved here

def run_toy_test():
    # 1. Initialize the evaluator (reads GEMINI_API_KEY from environment)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set your GEMINI_API_KEY environment variable first!")
        return
        
    evaluator = PRISMAEvaluator(api_key=api_key)

    # 2. Setup toy summaries
    # Notice: The generated summary adds "red" (hallucination) and misses "to Boston" (omission)
    reference_summary = "Alice drove her car to Boston yesterday afternoon while listening to music."
    generated_summary = "Alice drove a red car yesterday afternoon while listening to music."

    print("--- Running PRISMA Evaluation Toy Test ---")
    print(f"Reference Summary: '{reference_summary}'")
    print(f"Generated Summary: '{generated_summary}'\n")

    # 3. Run evaluation
    metrics = evaluator.evaluate(
        generated_summary=generated_summary, 
        reference_summary=reference_summary
    )

    # 4. Print structured results
    print("=== Extracted Facts ===")
    print("Generated Facts:")
    for f in metrics["generated_facts"]:
        print(f"  - {f}")
        
    print("\nReference Facts:")
    for f in metrics["reference_facts"]:
        print(f"  - {f}")

    print("\n=== PRISMA Metrics ===")
    # Precision drops because "the car is red" cannot be confirmed by the reference.
    print(f"Fact Precision: {metrics['fact_precision']} (Penalized for the hallucinated red car detail)")
    # Recall drops because "to Boston" was completely missed by the generator.
    print(f"Fact Recall:    {metrics['fact_recall']} (Penalized for missing the destination)")
    print(f"PRISMA F1 Score: {metrics['prisma_score']}")

if __name__ == "__main__":
    run_toy_test()