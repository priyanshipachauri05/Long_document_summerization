# test_llm.py

from summarization.llm_summarizer import summarize_chunk

text = """
Alice went to Paris and visited the Eiffel Tower.
She enjoyed French food and museums.
"""

print(summarize_chunk(text))