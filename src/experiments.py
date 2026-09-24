"""
Experiments for the Information Retrieval practical assignment.

Part 14:
    Experiment 1 - Stopword Removal
    Experiment 2 - Stemming
    Experiment 3 - Query Length

This file contains controlled experiments comparing
different preprocessing and query configurations.
"""

import os
import sys


# ============================================================
# PROJECT PATH
# ============================================================

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if project_root not in sys.path:
    sys.path.insert(
        0,
        project_root,
    )


# ============================================================
# IMPORTS
# ============================================================

from src.data_loader import load_documents

from src.preprocessing import (
    tokenize,
    remove_stopwords,
    stem,
    preprocess,
)

from src.indexing import (
    build_dictionary,
    build_inverted_index,
)

from src.retrieval import (
    build_document_vectors,
    process_query,
    search,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_without_stopwords(
    documents,
):
    """
    Preprocess documents using:

        Lowercase
        Tokenization
        Stemming

    Stopword removal is intentionally skipped.
    """

    processed_documents = {}

    for document_id, text in documents.items():

        tokens = tokenize(text)

        processed_tokens = stem(
            tokens
        )

        processed_documents[document_id] = (
            processed_tokens
        )

    return processed_documents


def preprocess_without_stemming(
    documents,
):
    """
    Preprocess documents using:

        Lowercase
        Tokenization
        Stopword removal

    Stemming is intentionally skipped.
    """

    processed_documents = {}

    for document_id, text in documents.items():

        tokens = tokenize(text)

        tokens_without_stopwords = (
            remove_stopwords(tokens)
        )

        processed_documents[document_id] = (
            tokens_without_stopwords
        )

    return processed_documents


def calculate_total_terms(
    processed_documents,
):
    """
    Calculate total processed terms across
    the entire collection.
    """

    return sum(
        len(document)
        for document in processed_documents.values()
    )


def calculate_average_terms(
    processed_documents,
):
    """
    Calculate average processed terms per document.
    """

    if not processed_documents:
        return 0.0

    return (
        calculate_total_terms(
            processed_documents
        )
        / len(processed_documents)
    )


def build_experiment_system(
    processed_documents,
):
    """
    Build dictionary, inverted index, and
    TF-IDF document vectors.
    """

    dictionary = build_dictionary(
        processed_documents
    )

    inverted_index = build_inverted_index(
        processed_documents
    )

    document_vectors = (
        build_document_vectors(
            processed_documents,
            inverted_index,
            dictionary,
        )
    )

    return (
        dictionary,
        inverted_index,
        document_vectors,
    )


def print_results(
    results,
):
    """
    Print ranked search results.
    """

    if not results:

        print(
            "No results found."
        )

        return

    print(
        f"{'Rank':<7}"
        f"{'Document':<12}"
        f"{'Score':<12}"
        f"Category"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['rank']:<7}"
            f"{result['document_id']:<12}"
            f"{result['score']:<12.6f}"
            f"{result['category']}"
        )


# ============================================================
# EXPERIMENT 1 - STOPWORD REMOVAL
# ============================================================

def stopword_removal_experiment(
    documents,
    categories,
):
    """
    Compare retrieval with and without stopword removal.
    """

    print("\n" + "=" * 80)
    print(
        "PART 14 - EXPERIMENT 1: "
        "STOPWORD REMOVAL"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Without stopword removal
    # --------------------------------------------------------

    print(
        "\nBuilding system WITHOUT stopword removal..."
    )

    no_stopword_documents = (
        preprocess_without_stopwords(
            documents
        )
    )

    (
        no_stopword_dictionary,
        no_stopword_index,
        no_stopword_vectors,
    ) = build_experiment_system(
        no_stopword_documents
    )

    # --------------------------------------------------------
    # With stopword removal
    # --------------------------------------------------------

    print(
        "Building system WITH stopword removal..."
    )

    stopword_documents = {}

    for document_id, text in documents.items():

        stopword_documents[document_id] = (
            preprocess(text)
        )

    (
        stopword_dictionary,
        stopword_index,
        stopword_vectors,
    ) = build_experiment_system(
        stopword_documents
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("COLLECTION COMPARISON")
    print("-" * 80)

    print(
        f"{'Measure':<35}"
        f"{'Without Stopwords':<22}"
        f"{'With Stopwords'}"
    )

    print("-" * 80)

    print(
        f"{'Vocabulary size':<35}"
        f"{no_stopword_dictionary.vocabulary_size():<22}"
        f"{stopword_dictionary.vocabulary_size()}"
    )

    print(
        f"{'Average terms per document':<35}"
        f"{calculate_average_terms(no_stopword_documents):<22.2f}"
        f"{calculate_average_terms(stopword_documents):.2f}"
    )

    print(
        f"{'Total processed terms':<35}"
        f"{calculate_total_terms(no_stopword_documents):<22}"
        f"{calculate_total_terms(stopword_documents)}"
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    query = "the space exploration"

    print("\n" + "-" * 80)
    print("RETRIEVAL COMPARISON")
    print("-" * 80)

    print(
        f"Query: '{query}'"
    )

    no_stopword_results = search(
        query,
        no_stopword_vectors,
        no_stopword_dictionary,
        no_stopword_index,
        categories,
        k=5,
    )

    stopword_results = search(
        query,
        stopword_vectors,
        stopword_dictionary,
        stopword_index,
        categories,
        k=5,
    )

    print(
        "\nWITHOUT STOPWORD REMOVAL"
    )

    print_results(
        no_stopword_results
    )

    print(
        "\nWITH STOPWORD REMOVAL"
    )

    print_results(
        stopword_results
    )

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    vocabulary_difference = (
        no_stopword_dictionary.vocabulary_size()
        - stopword_dictionary.vocabulary_size()
    )

    term_difference = (
        calculate_total_terms(
            no_stopword_documents
        )
        - calculate_total_terms(
            stopword_documents
        )
    )

    print("\n" + "-" * 80)
    print("EXPERIMENT OBSERVATION")
    print("-" * 80)

    print(
        "Stopword removal reduced the vocabulary by "
        f"{vocabulary_difference} terms."
    )

    print(
        "Stopword removal reduced the total processed "
        f"terms by {term_difference}."
    )

    if (
        [r["document_id"] for r in no_stopword_results]
        ==
        [r["document_id"] for r in stopword_results]
    ):

        print(
            "For this query, the top-five document "
            "ranking remained unchanged, although "
            "the similarity scores changed."
        )

    else:

        print(
            "For this query, stopword removal changed "
            "the top-five document ranking."
        )


# ============================================================
# EXPERIMENT 2 - STEMMING
# ============================================================

def stemming_experiment(
    documents,
    categories,
):
    """
    Compare retrieval with and without stemming.
    """

    print("\n" + "=" * 80)
    print(
        "PART 14 - EXPERIMENT 2: "
        "STEMMING"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Without stemming
    # --------------------------------------------------------

    print(
        "\nBuilding system WITHOUT stemming..."
    )

    no_stemming_documents = (
        preprocess_without_stemming(
            documents
        )
    )

    (
        no_stemming_dictionary,
        no_stemming_index,
        no_stemming_vectors,
    ) = build_experiment_system(
        no_stemming_documents
    )

    # --------------------------------------------------------
    # With stemming
    # --------------------------------------------------------

    print(
        "Building system WITH stemming..."
    )

    stemming_documents = {}

    for document_id, text in documents.items():

        stemming_documents[document_id] = (
            preprocess(text)
        )

    (
        stemming_dictionary,
        stemming_index,
        stemming_vectors,
    ) = build_experiment_system(
        stemming_documents
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("COLLECTION COMPARISON")
    print("-" * 80)

    print(
        f"{'Measure':<35}"
        f"{'Without Stemming':<22}"
        f"{'With Stemming'}"
    )

    print("-" * 80)

    print(
        f"{'Vocabulary size':<35}"
        f"{no_stemming_dictionary.vocabulary_size():<22}"
        f"{stemming_dictionary.vocabulary_size()}"
    )

    print(
        f"{'Average terms per document':<35}"
        f"{calculate_average_terms(no_stemming_documents):<22.2f}"
        f"{calculate_average_terms(stemming_documents):.2f}"
    )

    print(
        f"{'Total processed terms':<35}"
        f"{calculate_total_terms(no_stemming_documents):<22}"
        f"{calculate_total_terms(stemming_documents)}"
    )

    # --------------------------------------------------------
    # Stemming example
    # --------------------------------------------------------

    example_words = [
        "connected",
        "connecting",
        "connection",
        "connections",
        "computers",
        "computing",
    ]

    print("\n" + "-" * 80)
    print("STEMMING EXAMPLE")
    print("-" * 80)

    print(
        f"{'Original':<20}"
        f"{'Stemmed'}"
    )

    print("-" * 40)

    stemmed_examples = stem(
        example_words
    )

    for original, stemmed_word in zip(
        example_words,
        stemmed_examples,
    ):

        print(
            f"{original:<20}"
            f"{stemmed_word}"
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    query = "computer computing computers"

    print("\n" + "-" * 80)
    print("RETRIEVAL COMPARISON")
    print("-" * 80)

    print(
        f"Query: '{query}'"
    )

    no_stemming_results = search(
        query,
        no_stemming_vectors,
        no_stemming_dictionary,
        no_stemming_index,
        categories,
        k=5,
    )

    stemming_results = search(
        query,
        stemming_vectors,
        stemming_dictionary,
        stemming_index,
        categories,
        k=5,
    )

    print(
        "\nWITHOUT STEMMING"
    )

    print_results(
        no_stemming_results
    )

    print(
        "\nWITH STEMMING"
    )

    print_results(
        stemming_results
    )

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    vocabulary_difference = (
        no_stemming_dictionary.vocabulary_size()
        - stemming_dictionary.vocabulary_size()
    )

    print("\n" + "-" * 80)
    print("EXPERIMENT OBSERVATION")
    print("-" * 80)

    print(
        "Stemming reduced the vocabulary by "
        f"{vocabulary_difference} terms."
    )

    if (
        [r["document_id"] for r in no_stemming_results]
        ==
        [r["document_id"] for r in stemming_results]
    ):

        print(
            "For this query, the top-five document "
            "ranking remained unchanged."
        )

    else:

        print(
            "For this query, stemming changed the "
            "top-five document ranking."
        )


# ============================================================
# EXPERIMENT 3 - QUERY LENGTH
# ============================================================

def query_length_experiment(
    documents,
    categories,
):
    """
    Experiment 3:

        Compare retrieval for:

            1. One-word query
            2. Two-word query
            3. Longer query

    The same normal preprocessing pipeline is used
    for all three queries.
    """

    print("\n" + "=" * 80)
    print(
        "PART 14 - EXPERIMENT 3: "
        "QUERY LENGTH"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Use the normal production preprocessing pipeline.
    # --------------------------------------------------------

    print(
        "\nBuilding standard retrieval system..."
    )

    processed_documents = {}

    for document_id, text in documents.items():

        processed_documents[document_id] = (
            preprocess(text)
        )

    (
        dictionary,
        inverted_index,
        document_vectors,
    ) = build_experiment_system(
        processed_documents
    )

    print(
        f"Vocabulary size: "
        f"{dictionary.vocabulary_size()}"
    )

    # --------------------------------------------------------
    # Query configurations
    # --------------------------------------------------------

    queries = [
        (
            "One-word query",
            "fractal",
        ),
        (
            "Two-word query",
            "space orbit",
        ),
        (
            "Longer query",
            "space orbit nasa launch",
        ),
    ]

    # --------------------------------------------------------
    # Run each query
    # --------------------------------------------------------

    for query_type, query in queries:

        processed_query = process_query(
            query
        )

        results = search(
            query,
            document_vectors,
            dictionary,
            inverted_index,
            categories,
            k=5,
        )

        print("\n" + "-" * 80)
        print(
            f"{query_type.upper()}"
        )
        print("-" * 80)

        print(
            f"Original query: '{query}'"
        )

        print(
            f"Processed query: {processed_query}"
        )

        print(
            f"Number of processed terms: "
            f"{len(processed_query)}"
        )

        print(
            "\nTop 5 results:"
        )

        print_results(
            results
        )

    # --------------------------------------------------------
    # Observation
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("EXPERIMENT OBSERVATION")
    print("-" * 80)

    print(
        "Increasing query length introduces additional "
        "terms into the query TF-IDF vector."
    )

    print(
        "As more terms are added, document similarity "
        "is calculated against a richer representation "
        "of the user's information need."
    )

    print(
        "The exact ranking and scores depend on the "
        "terms present in the query and their IDF values "
        "in the document collection."
    )


# ============================================================
# MAIN - ALL PART 14 EXPERIMENTS
# ============================================================

def main():

    print(
        "Loading the 20 Newsgroups document collection..."
    )

    documents, categories = load_documents(
        documents_per_category=30
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    stopword_removal_experiment(
        documents,
        categories,
    )

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    stemming_experiment(
        documents,
        categories,
    )

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------

    query_length_experiment(
        documents,
        categories,
    )


if __name__ == "__main__":
    main()
