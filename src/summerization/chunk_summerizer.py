from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def summarize_chunk(chunk):

    result = summarizer(
        chunk,
        max_length=100,
        min_length=20,
        do_sample=False
    )

    return result[0]["summary_text"]