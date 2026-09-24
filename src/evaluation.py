"""
Final evaluation for the Information Retrieval practical assignment.

Part 15:
    Run at least 10 queries and record the top 5 results
    for each query.
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

from src.preprocessing import preprocess

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
# EVALUATION QUERIES
# ============================================================

EVALUATION_QUERIES = [
    "fractal",
    "computer graphics",
    "baseball game",
    "pitcher season",
    "medical information",
    "health disease",
    "space orbit",
    "nasa launch",
    "government policy",
    "political rights",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_standard_system(
    documents,
):
    """
    Build the complete standard retrieval system.

    Pipeline:

        Documents
            ↓
        Preprocessing
            ↓
        Dictionary
            ↓
        Inverted Index
            ↓
        TF-IDF Document Vectors
    """

    processed_documents = {}

    for document_id, text in documents.items():

        processed_documents[document_id] = (
            preprocess(text)
        )

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
        processed_documents,
        dictionary,
        inverted_index,
        document_vectors,
    )


def print_query_results(
    query_number,
    query,
    processed_query,
    results,
):
    """
    Print the results for one evaluation query.
    """

    print("\n" + "=" * 90)

    print(
        f"QUERY {query_number}: '{query}'"
    )

    print("=" * 90)

    print(
        f"Processed query: {processed_query}"
    )

    print(
        f"Number of processed terms: "
        f"{len(processed_query)}"
    )

    print()

    if not results:

        print(
            "No matching documents found."
        )

        return

    print(
        f"{'Rank':<7}"
        f"{'Document':<12}"
        f"{'Score':<12}"
        f"Category"
    )

    print("-" * 75)

    for result in results:

        print(
            f"{result['rank']:<7}"
            f"{result['document_id']:<12}"
            f"{result['score']:<12.6f}"
            f"{result['category']}"
        )


def create_report_line(
    query_number,
    query,
    processed_query,
    results,
):
    """
    Create a compact text representation of one
    evaluation query for the report file.
    """

    lines = []

    lines.append(
        f"QUERY {query_number}: {query}"
    )

    lines.append(
        f"Processed query: {processed_query}"
    )

    lines.append(
        f"Number of processed terms: "
        f"{len(processed_query)}"
    )

    lines.append(
        "Rank | Document | Score | Category"
    )

    for result in results:

        lines.append(
            f"{result['rank']} | "
            f"{result['document_id']} | "
            f"{result['score']:.6f} | "
            f"{result['category']}"
        )

    if not results:

        lines.append(
            "No matching documents found."
        )

    lines.append("")

    return lines


# ============================================================
# FINAL EVALUATION
# ============================================================

def run_final_evaluation(
    documents,
    categories,
):
    """
    Run the complete Part 15 evaluation.

    At least 10 queries are executed and the top
    5 results are recorded for each query.
    """

    print("\n" + "#" * 90)

    print(
        "PART 15 - FINAL EVALUATION"
    )

    print("#" * 90)

    print(
        f"\nNumber of documents: {len(documents)}"
    )

    print(
        f"Number of evaluation queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    # --------------------------------------------------------
    # Build retrieval system
    # --------------------------------------------------------

    print(
        "\nBuilding standard retrieval system..."
    )

    (
        processed_documents,
        dictionary,
        inverted_index,
        document_vectors,
    ) = build_standard_system(
        documents
    )

    print(
        f"Processed documents: "
        f"{len(processed_documents)}"
    )

    print(
        f"Vocabulary size: "
        f"{dictionary.vocabulary_size()}"
    )

    print(
        f"Document vectors: "
        f"{len(document_vectors)}"
    )

    print(
        f"Vector dimension: "
        f"{dictionary.vocabulary_size()}"
    )

    # --------------------------------------------------------
    # Run queries
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "PART 15 - FINAL EVALUATION"
    )

    report_lines.append(
        "=" * 90
    )

    report_lines.append(
        f"Documents: {len(documents)}"
    )

    report_lines.append(
        f"Vocabulary size: "
        f"{dictionary.vocabulary_size()}"
    )

    report_lines.append(
        f"Evaluation queries: "
        f"{len(EVALUATION_QUERIES)}"
    )

    report_lines.append("")

    for query_number, query in enumerate(
        EVALUATION_QUERIES,
        start=1,
    ):

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

        print_query_results(
            query_number,
            query,
            processed_query,
            results,
        )

        report_lines.extend(
            create_report_line(
                query_number,
                query,
                processed_query,
                results,
            )
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    reports_directory = os.path.join(
        project_root,
        "reports",
    )

    os.makedirs(
        reports_directory,
        exist_ok=True,
    )

    report_path = os.path.join(
        reports_directory,
        "part15_evaluation_results.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report_file:

        report_file.write(
            "\n".join(report_lines)
        )

    print("\n" + "=" * 90)

    print(
        "FINAL EVALUATION COMPLETED"
    )

    print("=" * 90)

    print(
        f"\nEvaluation report saved to:"
    )

    print(
        report_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading the 20 Newsgroups document collection..."
    )

    documents, categories = load_documents(
        documents_per_category=30
    )

    run_final_evaluation(
        documents,
        categories,
    )


if __name__ == "__main__":
    main()