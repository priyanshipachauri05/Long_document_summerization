def chunk_text(text, chunk_size=8000):
    """
    Split text into chunks of approximately chunk_size words.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
if __name__ == "__main__":

    sample_text = "hello " * 25000

    chunks = chunk_text(sample_text)

    print(f"Number of chunks: {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {len(chunk.split())} words")