import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def lowercase_text(text):
    return text.lower()


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text)


def remove_stopwords(tokens):

    stop_words = set(
        stopwords.words("english")
    )

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


def stem(tokens):

    stemmer = PorterStemmer()

    return [
        stemmer.stem(token)
        for token in tokens
    ]


def preprocess(text):

    # Step 1: Lowercase
    text = lowercase_text(text)

    # Step 2: Tokenization
    tokens = tokenize(text)

    # Step 3: Stop-word removal
    tokens = remove_stopwords(tokens)

    # Step 4: Stemming
    tokens = stem(tokens)

    return tokens


def preprocess_documents(documents):
   
    processed_documents = {}

    for document_id, text in documents.items():
        processed_documents[document_id] = preprocess(text)

    return processed_documents


def vocabulary_size(tokenized_documents):

    vocabulary = set()

    for tokens in tokenized_documents.values():
        vocabulary.update(tokens)

    return len(vocabulary)


def tokenize_documents(documents):

    tokenized_documents = {}

    for document_id, text in documents.items():

        text = lowercase_text(text)

        tokens = tokenize(text)

        tokenized_documents[document_id] = tokens

    return tokenized_documents


def remove_stopwords_documents(tokenized_documents):


    processed_documents = {}

    for document_id, tokens in tokenized_documents.items():

        processed_documents[document_id] = (
            remove_stopwords(tokens)
        )

    return processed_documents


def stemming_experiment():

    words = [
        "connected",
        "connecting",
        "connection",
        "connections",
        "computers",
        "computing",
    ]

    stemmer = PorterStemmer()

    print("\n" + "=" * 60)
    print("STEMMING EXPERIMENT")
    print("=" * 60)

    print(
        f"{'Original Term':<20}"
        f"{'Stem'}"
    )

    print("-" * 40)

    for word in words:

        stemmed_word = stemmer.stem(word)

        print(
            f"{word:<20}"
            f"{stemmed_word}"
        )


def preprocessing_demo(text):

    print("\n" + "=" * 60)
    print("PREPROCESSING DEMONSTRATION")
    print("=" * 60)

    print("\nOriginal:")
    print(text)

    # Step 1
    lowercased = lowercase_text(text)

    print("\n1. Lowercasing:")
    print(lowercased)

    # Step 2
    tokens = tokenize(lowercased)

    print("\n2. Tokenization:")
    print(tokens)

    # Step 3
    without_stopwords = remove_stopwords(tokens)

    print("\n3. Stop-word removal:")
    print(without_stopwords)

    # Step 4
    stemmed = stem(without_stopwords)

    print("\n4. Porter stemming:")
    print(stemmed)

    # Complete pipeline
    final_terms = preprocess(text)

    print("\n5. Complete preprocessing:")
    print(final_terms)


if __name__ == "__main__":

    example_text = (
        "NASA launched a Spacecraft into ORBIT."
    )

    preprocessing_demo(example_text)

    stemming_experiment()