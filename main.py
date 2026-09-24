from src.data_loader import (
    load_documents,
    dataset_statistics,
)

from src.preprocessing import (
    preprocess_documents,
    tokenize_documents,
    remove_stopwords_documents,
    vocabulary_size,
)

from src.indexing import (
    build_dictionary,
    print_dictionary_statistics,
    dictionary_demo,
    build_inverted_index,
)

from src.retrieval import (
    build_document_vectors,
    search,
)


# ============================================================
# PART 1 - DATASET STATISTICS
# ============================================================

def print_dataset_statistics(
    documents,
    categories,
):
    """
    Display the dataset statistics required
    by Part 1 of the assignment.
    """

    statistics = dataset_statistics(
        documents,
        categories,
    )

    print("=" * 60)
    print("INFORMATION RETRIEVAL PRACTICAL")
    print("=" * 60)

    print("\nDATASET STATISTICS")
    print("-" * 60)

    print(
        f"Number of documents: "
        f"{statistics['number_of_documents']}"
    )

    print(
        f"Number of categories: "
        f"{statistics['number_of_categories']}"
    )

    print(
        f"Average document length: "
        f"{statistics['average_document_length']:.2f}"
    )

    print(
        f"Shortest document: "
        f"{statistics['shortest_document']}"
    )

    print(
        f"Longest document: "
        f"{statistics['longest_document']}"
    )

    print("\nDocuments per category:")

    for category, count in (
        statistics["documents_per_category"].items()
    ):
        print(
            f"  {category}: {count}"
        )


# ============================================================
# PART 2 - PREPROCESSING DEMONSTRATION
# ============================================================

def print_preprocessing_demo(
    documents,
    categories,
    processed_documents,
):
    """
    Display preprocessing results for
    one real document from the dataset.
    """

    document_id = "D1"

    print("\n" + "=" * 60)
    print("REAL DOCUMENT PREPROCESSING")
    print("=" * 60)

    print(
        f"\nDocument ID: {document_id}"
    )

    print(
        f"Category: "
        f"{categories[document_id]}"
    )

    print("\nOriginal document:")

    print(
        documents[document_id][:1000]
    )

    print("\nPreprocessed terms:")

    print(
        processed_documents[document_id][:100]
    )


# ============================================================
# PART 2 - VOCABULARY ANALYSIS
# ============================================================

def print_vocabulary_analysis(
    documents,
    processed_documents,
):
    """
    Calculate and display vocabulary sizes
    before and after preprocessing.

    Required by Part 2.5.
    """

    # Stage 1:
    # Lowercase + tokenization
    tokenized_documents = tokenize_documents(
        documents
    )

    vocabulary_before = vocabulary_size(
        tokenized_documents
    )

    # Stage 2:
    # Stop-word removal
    no_stopword_documents = (
        remove_stopwords_documents(
            tokenized_documents
        )
    )

    vocabulary_after_stopwords = (
        vocabulary_size(
            no_stopword_documents
        )
    )

    # Stage 3:
    # Complete preprocessing
    vocabulary_after_stemming = (
        vocabulary_size(
            processed_documents
        )
    )

    print("\n" + "=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    print(
        "\nVocabulary after "
        "lowercasing + tokenization: "
        f"{vocabulary_before}"
    )

    print(
        "Vocabulary after "
        "stop-word removal: "
        f"{vocabulary_after_stopwords}"
    )

    print(
        "Vocabulary after "
        "stemming: "
        f"{vocabulary_after_stemming}"
    )

    print("\nEffect of preprocessing:")

    reduction = (
        vocabulary_before
        - vocabulary_after_stemming
    )

    print(
        f"Unique terms reduced by: "
        f"{reduction}"
    )


# ============================================================
# PART 13 - CLI
# ============================================================

def run_cli(
    documents,
    categories,
    document_vectors,
    dictionary,
    inverted_index,
):
    """
    Run the interactive command-line search interface.

    The user can enter queries repeatedly until
    'exit' is entered.
    """

    print("\n" + "=" * 70)
    print("PART 13 - COMMAND-LINE SEARCH")
    print("=" * 70)

    print(
        "\nThe IR system is ready."
    )

    print(
        "Enter a search query to retrieve ranked documents."
    )

    print(
        "Type 'exit' to quit."
    )

    print(
        "\nExample queries:"
    )

    print(
        "  fractal"
    )

    print(
        "  space orbit"
    )

    print(
        "  computer graphics"
    )

    print(
        "  baseball game"
    )

    print(
        "  medical information"
    )

    while True:

        print()

        query = input(
            "Enter your query: "
        ).strip()

        # ----------------------------------------------------
        # Exit command
        # ----------------------------------------------------

        if query.lower() == "exit":
            print(
                "\nExiting Information Retrieval system."
            )

            break

        # ----------------------------------------------------
        # Empty query
        # ----------------------------------------------------

        if not query:

            print(
                "Please enter a non-empty query."
            )

            continue

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = search(
            query,
            document_vectors,
            dictionary,
            inverted_index,
            categories,
            k=10,
        )

        # ----------------------------------------------------
        # Display query
        # ----------------------------------------------------

        print("\n" + "=" * 80)

        print(
            f"SEARCH QUERY: '{query}'"
        )

        print("=" * 80)

        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

        if not results:

            print(
                "No matching documents found."
            )

            continue

        # ----------------------------------------------------
        # Result header
        # ----------------------------------------------------

        print(
            f"{'Rank':<7}"
            f"{'Document':<11}"
            f"{'Score':<12}"
            f"{'Category':<28}"
            f"Preview"
        )

        print("-" * 100)

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        for result in results:

            document_id = (
                result["document_id"]
            )

            preview = documents[
                document_id
            ].replace(
                "\n",
                " ",
            ).strip()

            # Keep the preview short enough
            # for terminal output.
            if len(preview) > 60:
                preview = (
                    preview[:57]
                    + "..."
                )

            print(
                f"{result['rank']:<7}"
                f"{document_id:<11}"
                f"{result['score']:<12.6f}"
                f"{result['category']:<28}"
                f"{preview}"
            )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    # ========================================================
    # PART 1 - LOAD DOCUMENTS
    # ========================================================

    documents, categories = load_documents(
        documents_per_category=30
    )

    # ========================================================
    # PART 1 - DATASET STATISTICS
    # ========================================================

    print_dataset_statistics(
        documents,
        categories,
    )

    # ========================================================
    # PART 2 - PREPROCESSING
    # ========================================================

    processed_documents = (
        preprocess_documents(
            documents
        )
    )

    # Demonstrate preprocessing
    print_preprocessing_demo(
        documents,
        categories,
        processed_documents,
    )

    # Vocabulary analysis
    print_vocabulary_analysis(
        documents,
        processed_documents,
    )

    # ========================================================
    # PART 3 - DICTIONARY
    # ========================================================

    dictionary = build_dictionary(
        processed_documents
    )

    print_dictionary_statistics(
        dictionary,
        processed_documents,
    )

    dictionary_demo(
        dictionary
    )

    # ========================================================
    # PART 4 - INVERTED INDEX
    # ========================================================

    inverted_index = (
        build_inverted_index(
            processed_documents
        )
    )

    print("\n" + "=" * 60)
    print("INDEXING")
    print("=" * 60)

    print(
        f"\nIndexed terms: "
        f"{len(inverted_index)}"
    )

    # ========================================================
    # PART 12 - BUILD DOCUMENT VECTORS
    # ========================================================

    print("\n" + "=" * 60)
    print("BUILDING SEARCH INDEX")
    print("=" * 60)

    print(
        f"\nBuilding TF-IDF vectors for "
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

    # ========================================================
    # PART 13 - INTERACTIVE CLI
    # ========================================================

    run_cli(
        documents,
        categories,
        document_vectors,
        dictionary,
        inverted_index,
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
