def chunk_text(text, max_words=100):

    sentences = text.split(".")

    chunks = []
    current_chunk = ""
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count <= max_words:

            current_chunk += sentence + ". "
            current_word_count += sentence_word_count

        else:

            chunks.append(current_chunk.strip())

            current_chunk = sentence + ". "
            current_word_count = sentence_word_count

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
if __name__ == "__main__":

    sample_text = """
    Our environment is made up of everything living and non-living around us—plants, animals, humans, water, soil, air, and even buildings. 
    Both natural and man-made elements are part of the environment. 
    The environment is very important because it provides us with everything necessary for survival, like clean air, safe water, fertile land, and biodiversity. 
    Sadly, pollution, deforestation, and overuse of resources are harming the environment. 
    This leads to problems like climate change, water shortages, and extinction of species. 
    Everyone, especially students, should help protect the environment by planting trees, avoiding plastic, saving water and electricity, and joining awareness campaigns like World Environment Day. 
    By caring for our surroundings, we protect our health and the future of our planet.
    """

    chunks = chunk_text(sample_text)

    print(f"Number of chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {len(chunk.split())} words")
    
    print("\nCHUNKS:\n")

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}")
        print(chunk)
        print("-" * 50)