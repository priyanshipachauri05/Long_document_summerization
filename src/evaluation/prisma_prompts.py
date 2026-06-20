# evaluation/prisma_prompts.py

FACT_EXTRACTION_PROMPT = """
You are an expert at decomposing summaries into atomic facts.

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

-------------------------
Summary:
{summary}

Extract ALL atomic facts.



Rules:
1. Each fact must contain exactly ONE piece of information.
2. Preserve every factual statement.
3. Split compound sentences into separate facts.
4. Do not infer or add information.
5. Do not explain anything.

Return ONLY the facts.
One fact per line.
Do not number them.

Summary:
{summary}
"""

ENTAILMENT_PROMPT = """
You are an expert in natural language inference.

Determine whether the Statement is fully supported (entailed) by the Reference.

Statement:
{statement}

Reference:
{reference}

Instructions:
- Answer YES if the statement is completely supported.
- Answer NO if it is partially supported, contradicted, or not mentioned.
- Do not explain.
- Return ONLY YES or NO.
"""