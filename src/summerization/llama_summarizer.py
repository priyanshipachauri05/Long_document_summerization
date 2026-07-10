from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import os
import time
from groq import RateLimitError

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

project_root = Path(__file__).resolve().parents[2]
env_path = project_root / ".env"

load_dotenv(env_path, override=True)

api_key = os.getenv("GROQ_API_KEY")

if api_key is None:
    raise ValueError(
        f"GROQ_API_KEY not found in {env_path}"
    )

client = Groq(api_key=api_key)


# ==========================================
# GENERIC LLM CALL
# ==========================================

def call_llm(prompt):

    while True:
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )

            return response.choices[0].message.content

        except RateLimitError:
            print("Rate limit hit. Waiting 1 second...")
            time.sleep(1)


# ==========================================
# CHUNK SUMMARIZATION
# ==========================================

def summarize_chunk(chunk):

    prompt = f"""
You are an expert document summarizer.

Summarize the following text into 2–3 concise sentences.

Preserve all important factual information.

Do not add new information.

Text:
{chunk}

Summary:
"""

    return call_llm(prompt)


# ==========================================
# ZERO-SHOT BASELINE
# ==========================================

def zero_shot(document):

    prompt = f"""
You are an expert document summarizer.

Summarize the following document into a concise summary while preserving all important information.

Document:
{document}

Summary:
"""

    return call_llm(prompt)


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    text = """
    Our environment is made up of everything living and non-living around us.
    The environment provides clean air, water and biodiversity.
    Pollution and deforestation threaten natural resources.
    Everyone should protect the environment.
    """

    print(summarize_chunk(text))