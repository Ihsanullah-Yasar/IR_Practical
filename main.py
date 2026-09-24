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
)


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


def main():

    # ==================================================
    # PART 1: LOAD DOCUMENTS
    # ==================================================

    documents, categories = load_documents(
        documents_per_category=30
    )

    # ==================================================
    # PART 1: DATASET STATISTICS
    # ==================================================

    print_dataset_statistics(
        documents,
        categories,
    )

    # ==================================================
    # PART 2: PREPROCESSING
    # ==================================================

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

    # ==================================================
    # PART 3: DICTIONARY
    # ==================================================

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


if __name__ == "__main__":
    main()