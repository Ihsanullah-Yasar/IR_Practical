"""
Retrieval and query representation functionality
for the Information Retrieval practical assignment.

Part 10:
    Query Representation

Part 11:
    Vector Creation and Cosine Similarity
"""

import math
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

    Normalized TF is used:

        TF(t,q) = count(t,q) / total query terms

    Parameters:
        query_terms:
            Preprocessed query terms.

        inverted_index:
            Collection inverted index.

        total_documents:
            Total number of documents.

    Returns:
        Dictionary mapping query terms to TF-IDF values.
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

    The vector position is determined by the term ID
    stored in the Dictionary object.

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

            # Term IDs start from 1, while NumPy
            # array positions start from 0.
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

    Parameters:
        query_vector:
            NumPy vector representing the query.

        document_vector:
            NumPy vector representing a document.

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
# PART 11 - VECTOR AND COSINE DEMONSTRATION
# ============================================================

def vector_and_cosine_demo(
    processed_documents,
    inverted_index,
    dictionary,
):
    """
    Demonstrate:

        1. Query TF-IDF
        2. Document TF-IDF
        3. Vector creation
        4. Dot product
        5. Vector magnitudes
        6. Cosine similarity

    This performs a manual test before implementing
    the complete ranked search in Part 12.
    """

    print("\n" + "=" * 60)
    print(
        "PART 11 - VECTOR CREATION "
        "AND COSINE SIMILARITY"
    )
    print("=" * 60)

    total_documents = len(
        processed_documents
    )

    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    query = "fractal"

    print(
        f"\nTest query: '{query}'"
    )

    query_terms = process_query(
        query
    )

    print(
        f"Processed query: {query_terms}"
    )

    query_tfidf = calculate_query_tfidf(
        query_terms,
        inverted_index,
        total_documents,
    )

    print(
        f"Query TF-IDF: {query_tfidf}"
    )

    # --------------------------------------------------------
    # Create query vector
    # --------------------------------------------------------

    query_vector = create_vector(
        query_tfidf,
        dictionary,
    )

    print(
        f"\nQuery vector dimension: "
        f"{len(query_vector)}"
    )

    print(
        f"Query vector non-zero values: "
        f"{np.count_nonzero(query_vector)}"
    )

    # --------------------------------------------------------
    # Select a document containing the query term
    # --------------------------------------------------------

    candidate_documents = (
        inverted_index.get(
            "fractal",
            [],
        )
    )

    if not candidate_documents:
        print(
            "\nNo document contains "
            "the test query term."
        )

        return

    document_id = candidate_documents[0]

    print(
        f"\nSelected document: "
        f"{document_id}"
    )

    # --------------------------------------------------------
    # Create document TF-IDF
    # --------------------------------------------------------

    from src.indexing import (
        calculate_tfidf,
    )

    document_tfidf = calculate_tfidf(
        processed_documents[document_id],
        inverted_index,
        total_documents,
    )

    print(
        f"Document TF-IDF terms: "
        f"{len(document_tfidf)}"
    )

    # --------------------------------------------------------
    # Create document vector
    # --------------------------------------------------------

    document_vector = create_vector(
        document_tfidf,
        dictionary,
    )

    print(
        f"Document vector dimension: "
        f"{len(document_vector)}"
    )

    print(
        f"Document vector non-zero values: "
        f"{np.count_nonzero(document_vector)}"
    )

    # --------------------------------------------------------
    # Dot product
    # --------------------------------------------------------

    dot_product = np.dot(
        query_vector,
        document_vector,
    )

    print(
        f"\nDot product: "
        f"{dot_product:.6f}"
    )

    # --------------------------------------------------------
    # Magnitudes
    # --------------------------------------------------------

    query_magnitude = np.linalg.norm(
        query_vector
    )

    document_magnitude = np.linalg.norm(
        document_vector
    )

    print(
        f"Query vector magnitude: "
        f"{query_magnitude:.6f}"
    )

    print(
        f"Document vector magnitude: "
        f"{document_magnitude:.6f}"
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    similarity = cosine_similarity(
        query_vector,
        document_vector,
    )

    print(
        f"\nCosine similarity:"
        f" {similarity:.6f}"
    )

    # --------------------------------------------------------
    # Manual formula verification
    # --------------------------------------------------------

    manual_similarity = (
        dot_product
        / (
            query_magnitude
            * document_magnitude
        )
    )

    print(
        "\nCOSINE FORMULA VERIFICATION"
    )

    print("-" * 60)

    print(
        "cosine(query, document) = "
        "(query · document) / "
        "(||query|| × ||document||)"
    )

    print(
        f"\n= {dot_product:.6f} / "
        f"({query_magnitude:.6f} × "
        f"{document_magnitude:.6f})"
    )

    print(
        f"= {manual_similarity:.6f}"
    )

    # --------------------------------------------------------
    # Zero-vector test
    # --------------------------------------------------------

    zero_vector = np.zeros(
        dictionary.vocabulary_size()
    )

    zero_similarity = cosine_similarity(
        query_vector,
        zero_vector,
    )

    print(
        "\nZero-vector similarity test:"
    )

    print(
        f"cosine(query, zero) = "
        f"{zero_similarity:.6f}"
    )

    # --------------------------------------------------------
    # Dimension mismatch test
    # --------------------------------------------------------

    print(
        "\nDimension mismatch test:"
    )

    try:

        cosine_similarity(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        )

    except ValueError as error:

        print(
            f"Correctly raised ValueError: "
            f"{error}"
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
    # Part 11 demonstration
    # --------------------------------------------------------

    vector_and_cosine_demo(
        processed_documents,
        inverted_index,
        dictionary,
    )