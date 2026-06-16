from collections import Counter
import string


def extract_context(document, k=3):
    """
    Select top-k important sentences from a document.
    """

    sentences = document.split(".")

    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if sentence:
            cleaned_sentences.append(sentence)

    # Build word frequency table
    words = []

    for sentence in cleaned_sentences:

        for word in sentence.lower().split():

            word = word.strip(string.punctuation)

            if word:
                words.append(word)

    word_freq = Counter(words)

    # Score each sentence
    sentence_scores = []

    for sentence in cleaned_sentences:

        score = 0

        for word in sentence.lower().split():

            word = word.strip(string.punctuation)

            score += word_freq[word]

        sentence_scores.append(
            (sentence, score)
        )

    # Select top-k sentences
    sentence_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    selected_sentences = [
        sentence
        for sentence, score in sentence_scores[:k]
    ]

    return selected_sentences
if __name__ == "__main__":

    document = """
    The environment provides clean air and water.
    Pollution harms ecosystems and biodiversity.
    Students should help protect nature.
    Trees improve air quality.
    Conservation protects future generations.
    """

    selected = extract_context(
        document,
        k=2
    )

    print("\nSelected Context:\n")

    for sentence in selected:
        print("-", sentence)