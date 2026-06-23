FACT_EXTRACTION_PROMPT = """
You are an expert at extracting atomic facts.

An atomic fact contains exactly ONE piece of information.

Example:

Summary:
John and Mary travelled to Paris. They visited the Eiffel Tower.

Atomic Facts:
John travelled to Paris.
Mary travelled to Paris.
John travelled with Mary.
John visited the Eiffel Tower.
Mary visited the Eiffel Tower.

----------------------------------------

Now extract ALL atomic facts from the following summary.

Rules:
- One fact per line.
- Split compound statements into multiple facts.
- Do NOT infer new information.
- Do NOT omit any fact.
- Return ONLY the facts.

Summary:
{summary}
"""


BATCH_ENTAILMENT_PROMPT = """
You are an expert in factual consistency evaluation.

Reference Facts:
{reference}

Generated Facts:
{generated}

For EACH generated fact, determine whether it is completely supported by the reference facts.

Return ONLY valid JSON.

Example:

{
  "results":[true,false,true]
}

The number of boolean values MUST equal the number of generated facts.

Do not explain anything.
Return ONLY JSON.
"""