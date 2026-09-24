"""
Retrieval and query representation functionality
for the Information Retrieval practical assignment.

Part 10:
    Query Representation
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
# DEMONSTRATION
# ============================================================

def query_representation_demo(
    inverted_index,
    total_documents,
):
    """
    Demonstrate query processing and TF-IDF
    query representation.
    """

    print("\n" + "=" * 60)
    print("PART 10 - QUERY REPRESENTATION")
    print("=" * 60)

    example_queries = [
        "fractal",
        "space orbit",
        "computer graphics",
        "the medical information",
    ]

    for query in example_queries:

        result = query_representation(
            query,
            inverted_index,
            total_documents,
        )

        print("\n" + "-" * 60)

        print(
            f"Original query: "
            f"'{result['original_query']}'"
        )

        print(
            f"Processed query terms: "
            f"{result['processed_terms']}"
        )

        print(
            "\nQUERY TF-IDF"
        )

        print("-" * 60)

        print(
            f"{'Term':<20}"
            f"{'TF':>12}"
            f"{'IDF':>12}"
            f"{'TF-IDF':>15}"
        )

        print("-" * 60)

        for term, tfidf_value in (
            result["tfidf"].items()
        ):

            idf = calculate_idf(
                term,
                inverted_index,
                total_documents,
            )

            tf = result["tf"].get(
                term,
                0.0,
            )

            print(
                f"{term:<20}"
                f"{tf:>12.6f}"
                f"{idf:>12.6f}"
                f"{tfidf_value:>15.6f}"
            )

        print("-" * 60)

    # --------------------------------------------------------
    # Empty query test
    # --------------------------------------------------------

    empty_query = ""

    empty_result = query_representation(
        empty_query,
        inverted_index,
        total_documents,
    )

    print(
        "\nEmpty query:"
    )

    print(
        f"Processed terms: "
        f"{empty_result['processed_terms']}"
    )

    print(
        f"TF-IDF: "
        f"{empty_result['tfidf']}"
    )

    # --------------------------------------------------------
    # Unknown-term test
    # --------------------------------------------------------

    unknown_query = (
        "thistermdoesnotexist"
    )

    unknown_result = query_representation(
        unknown_query,
        inverted_index,
        total_documents,
    )

    print(
        "\nUnknown-term query:"
    )

    print(
        f"Processed terms: "
        f"{unknown_result['processed_terms']}"
    )

    print(
        f"TF-IDF: "
        f"{unknown_result['tfidf']}"
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
    # Build inverted index
    # --------------------------------------------------------

    from src.indexing import (
        build_inverted_index,
    )

    inverted_index = (
        build_inverted_index(
            processed_documents
        )
    )

    # --------------------------------------------------------
    # Part 10 demonstration
    # --------------------------------------------------------

    query_representation_demo(
        inverted_index,
        len(processed_documents),
    )