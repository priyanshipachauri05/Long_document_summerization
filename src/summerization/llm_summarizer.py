from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
)

def summarize_chunk(chunk):

    result = summarizer(
        chunk,
        max_length=100,
        min_length=30,
        do_sample=False
    )

    return result[0]["summary_text"]

text = """
    Our environment is made up of everything living and non-living around us—plants, animals, humans, water, soil, air, and even buildings. 
    Both natural and man-made elements are part of the environment. 
    The environment is very important because it provides us with everything necessary for survival, like clean air, safe water, fertile land, and biodiversity. 
    Sadly, pollution, deforestation, and overuse of resources are harming the environment. 
    This leads to problems like climate change, water shortages, and extinction of species. 
    Everyone, especially students, should help protect the environment by planting trees, avoiding plastic, saving water and electricity, and joining awareness campaigns like World Environment Day. 
    By caring for our surroundings, we protect our health and the future of our planet.
"""

print(summarize_chunk(text))