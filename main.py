from src.data_loader import (
    load_documents,
    dataset_statistics,
)


def main():
    documents, categories = load_documents(
        documents_per_category=30
    )

    statistics = dataset_statistics(
        documents,
        categories
    )

    print("=" * 60)
    print("INFORMATION RETRIEVAL PRACTICAL")
    print("=" * 60)

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
        print(f"  {category}: {count}")

    print("\nSample document:")

    document_id = next(iter(documents))

    print(f"ID: {document_id}")
    print(f"Category: {categories[document_id]}")
    print(
        f"Text:\n{documents[document_id][:500]}"
    )


if __name__ == "__main__":
    main()