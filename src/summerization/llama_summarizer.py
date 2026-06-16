from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
def summarize_chunk(chunk):

    prompt = f"""
Summarize the following text in 2-3 concise sentences.

Text:
{chunk}

Summary:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

if __name__=="__main__":
    text="""Our environment is made up of everything living and non-living around us—plants, animals, humans, water, soil, air, and even buildings.
    Both natural and man-made elements are part of the environment.
    The environment is very important because it provides us with everything necessary for survival, like clean air, safe water, fertile land, and biodiversity.
    Sadly, pollution, deforestation, and overuse of resources are harming the environment.
    This leads to problems like climate change, water shortages, and extinction of species.
    Everyone, especially students, should help protect the environment by planting trees, avoiding plastic, saving water and electricity, and joining awareness campaigns like World Environment Day.
    By caring for our surroundings, we protect our health and the future of our planet."""
    print(summarize_chunk(text))