"""
Retrieval functionality for the Information Retrieval
practical assignment.

Part 10:
    Query Representation

Part 11:
    Vector Creation and Cosine Similarity

Part 12:
    Ranked Search
"""

import os
import sys

import numpy as np


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

from src.preprocessing import preprocess

from src.indexing import (
    calculate_tf,
    calculate_idf,
    calculate_tfidf,
)


# ============================================================
# PART 10 - QUERY REPRESENTATION
# ============================================================

def process_query(query):
    """
    Process a user query using the same preprocessing
    pipeline used for documents.

    Pipeline:

        Query
          ↓
        Lowercase
          ↓
        Tokenization
          ↓
        Stop-word removal
          ↓
        Stemming

    Parameters:
        query:
            Raw query string.

    Returns:
        List of processed query terms.
    """

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    return preprocess(query)


def calculate_query_tfidf(
    query_terms,
    inverted_index,
    total_documents,
):
    """
    Calculate the TF-IDF representation of a processed query.

    Formula:

        TFIDF(t,q) = TF(t,q) * IDF(t)

    Parameters:
        query_terms:
            Preprocessed query terms.

        inverted_index:
            Collection inverted index.

        total_documents:
            Total number of documents.

    Returns:
        Dictionary mapping terms to TF-IDF values.
    """

    if total_documents <= 0:
        raise ValueError(
            "total_documents must be greater than 0."
        )

    normalized_tf = calculate_tf(
        query_terms,
        normalized=True,
    )

    query_tfidf = {}

    for term, tf in normalized_tf.items():

        idf = calculate_idf(
            term,
            inverted_index,
            total_documents,
        )

        query_tfidf[term] = tf * idf

    return query_tfidf


def query_representation(
    query,
    inverted_index,
    total_documents,
):
    """
    Convert a raw query into its TF-IDF representation.

    Returns:

        {
            "original_query": "...",
            "processed_terms": [...],
            "tf": {...},
            "tfidf": {...}
        }
    """

    processed_terms = process_query(
        query
    )

    normalized_tf = calculate_tf(
        processed_terms,
        normalized=True,
    )

    query_tfidf = calculate_query_tfidf(
        processed_terms,
        inverted_index,
        total_documents,
    )

    return {
        "original_query": query,
        "processed_terms": processed_terms,
        "tf": normalized_tf,
        "tfidf": query_tfidf,
    }


# ============================================================
# PART 11 - VECTOR CREATION
# ============================================================

def create_vector(
    tfidf_values,
    dictionary,
):
    """
    Convert a sparse TF-IDF dictionary into a dense NumPy vector.

    The vector position is determined by the term ID stored
    in the Dictionary object.

    Parameters:
        tfidf_values:
            Dictionary mapping terms to TF-IDF weights.

        dictionary:
            Dictionary object containing term-to-ID mappings.

    Returns:
        NumPy array with one position for every vocabulary term.
    """

    vector = np.zeros(
        dictionary.vocabulary_size(),
        dtype=float,
    )

    for term, weight in tfidf_values.items():

        if dictionary.term_exists(term):

            term_id = dictionary.get_term_id(
                term
            )

            # Term IDs start at 1.
            # NumPy positions start at 0.
            vector[term_id - 1] = weight

    return vector


# ============================================================
# PART 11 - COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_vector,
    document_vector,
):
    """
    Calculate cosine similarity between two vectors.

    Formula:

        cosine(A, B) =
            (A . B) / (||A|| * ||B||)

    Returns:
        Cosine similarity as a float.
    """

    query_vector = np.asarray(
        query_vector,
        dtype=float,
    )

    document_vector = np.asarray(
        document_vector,
        dtype=float,
    )

    if query_vector.shape != document_vector.shape:

        raise ValueError(
            "Query vector and document vector "
            "must have the same dimensions."
        )

    query_magnitude = np.linalg.norm(
        query_vector
    )

    document_magnitude = np.linalg.norm(
        document_vector
    )

    # Avoid division by zero.
    if (
        query_magnitude == 0.0
        or document_magnitude == 0.0
    ):
        return 0.0

    dot_product = np.dot(
        query_vector,
        document_vector,
    )

    return float(
        dot_product
        / (
            query_magnitude
            * document_magnitude
        )
    )


# ============================================================
# PART 12 - BUILD DOCUMENT VECTORS
# ============================================================

def build_document_vectors(
    processed_documents,
    inverted_index,
    dictionary,
):
    """
    Build TF-IDF vectors for all documents.

    The vectors are calculated once and can then be reused
    for multiple queries.

    Returns:

        {
            "D1": numpy_vector,
            "D2": numpy_vector,
            ...
        }
    """

    total_documents = len(
        processed_documents
    )

    document_vectors = {}

    for document_id, document in (
        processed_documents.items()
    ):

        document_tfidf = calculate_tfidf(
            document,
            inverted_index,
            total_documents,
        )

        document_vectors[document_id] = (
            create_vector(
                document_tfidf,
                dictionary,
            )
        )

    return document_vectors


# ============================================================
# PART 12 - RANKED SEARCH
# ============================================================

def search(
    query,
    document_vectors,
    dictionary,
    inverted_index,
    categories,
    k=10,
):
    """
    Search the document collection and return the
    top-k documents ranked by cosine similarity.

    Parameters:
        query:
            Raw user query.

        document_vectors:
            Precomputed document TF-IDF vectors.

        dictionary:
            Term dictionary.

        inverted_index:
            Collection inverted index.

        categories:
            Mapping of document IDs to categories.

        k:
            Number of results to return.

    Returns:
        List of result dictionaries:

        [
            {
                "rank": 1,
                "document_id": "D1",
                "score": 0.123456,
                "category": "comp.graphics"
            },
            ...
        ]
    """

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    # --------------------------------------------------------
    # Process query
    # --------------------------------------------------------

    query_terms = process_query(
        query
    )

    # Empty query produces no results.
    if not query_terms:
        return []

    # --------------------------------------------------------
    # Query TF-IDF
    # --------------------------------------------------------

    query_tfidf = calculate_query_tfidf(
        query_terms,
        inverted_index,
        len(document_vectors),
    )

    # --------------------------------------------------------
    # Create query vector
    # --------------------------------------------------------

    query_vector = create_vector(
        query_tfidf,
        dictionary,
    )

    # --------------------------------------------------------
    # If the query contains only unknown terms,
    # its vector will be all zeros.
    # --------------------------------------------------------

    if np.linalg.norm(query_vector) == 0.0:
        return []

    # --------------------------------------------------------
    # Calculate similarity with every document
    # --------------------------------------------------------

    scored_documents = []

    for document_id, document_vector in (
        document_vectors.items()
    ):

        score = cosine_similarity(
            query_vector,
            document_vector,
        )

        # Only retain documents that have
        # a meaningful similarity with the query.
        if score > 0.0:

            scored_documents.append(
                {
                    "document_id": document_id,
                    "score": score,
                    "category": categories.get(
                        document_id,
                        "unknown",
                    ),
                }
            )

    # --------------------------------------------------------
    # Sort by similarity score
    # --------------------------------------------------------

    scored_documents.sort(
        key=lambda result: (
            -result["score"],
            result["document_id"],
        )
    )

    # --------------------------------------------------------
    # Return top-k results
    # --------------------------------------------------------

    top_results = scored_documents[:k]

    for rank, result in enumerate(
        top_results,
        start=1,
    ):
        result["rank"] = rank

    return top_results


# ============================================================
# PART 12 - SEARCH RESULT DISPLAY
# ============================================================

def print_search_results(
    query,
    results,
):
    """
    Print ranked search results.
    """

    print("\n" + "=" * 70)

    print(
        f"SEARCH QUERY: '{query}'"
    )

    print("=" * 70)

    if not results:

        print(
            "No results found."
        )

        return

    print(
        f"{'Rank':<8}"
        f"{'Document':<12}"
        f"{'Score':<15}"
        f"Category"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['rank']:<8}"
            f"{result['document_id']:<12}"
            f"{result['score']:<15.6f}"
            f"{result['category']}"
        )


# ============================================================
# PART 12 - RANKED SEARCH DEMONSTRATION
# ============================================================

def ranked_search_demo(
    processed_documents,
    inverted_index,
    dictionary,
    categories,
):
    """
    Demonstrate ranked retrieval using several queries.
    """

    print("\n" + "=" * 70)
    print(
        "PART 12 - RANKED SEARCH"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Build document vectors once.
    # --------------------------------------------------------

    print(
        "\nBuilding TF-IDF vectors for "
        f"{len(processed_documents)} documents..."
    )

    document_vectors = (
        build_document_vectors(
            processed_documents,
            inverted_index,
            dictionary,
        )
    )

    print(
        f"Document vectors created: "
        f"{len(document_vectors)}"
    )

    print(
        f"Vector dimension: "
        f"{dictionary.vocabulary_size()}"
    )

    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    test_queries = [
        "fractal",
        "space orbit",
        "computer graphics",
        "baseball game",
        "medical information",
    ]

    for query in test_queries:

        results = search(
            query,
            document_vectors,
            dictionary,
            inverted_index,
            categories,
            k=5,
        )

        print_search_results(
            query,
            results,
        )

    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    empty_results = search(
        "",
        document_vectors,
        dictionary,
        inverted_index,
        categories,
        k=5,
    )

    print(
        "\nEmpty query results:"
    )

    print(
        empty_results
    )

    # --------------------------------------------------------
    # Unknown query
    # --------------------------------------------------------

    unknown_query = (
        "thistermdoesnotexist"
    )

    unknown_results = search(
        unknown_query,
        document_vectors,
        dictionary,
        inverted_index,
        categories,
        k=5,
    )

    print(
        "\nUnknown query results:"
    )

    print(
        unknown_results
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    from src.data_loader import (
        load_documents,
    )

    from src.preprocessing import (
        preprocess_documents,
    )

    from src.indexing import (
        build_dictionary,
        build_inverted_index,
    )

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents, categories = (
        load_documents(
            documents_per_category=30
        )
    )

    # --------------------------------------------------------
    # Preprocess documents
    # --------------------------------------------------------

    processed_documents = (
        preprocess_documents(
            documents
        )
    )

    # --------------------------------------------------------
    # Build dictionary
    # --------------------------------------------------------

    dictionary = build_dictionary(
        processed_documents
    )

    # --------------------------------------------------------
    # Build inverted index
    # --------------------------------------------------------

    inverted_index = (
        build_inverted_index(
            processed_documents
        )
    )

    # --------------------------------------------------------
    # Part 12
    # --------------------------------------------------------

    ranked_search_demo(
        processed_documents,
        inverted_index,
        dictionary,
        categories,
    )