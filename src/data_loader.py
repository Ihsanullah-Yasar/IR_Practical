from collections import Counter

from sklearn.datasets import fetch_20newsgroups


CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]


def load_documents(documents_per_category=30):

    dataset = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=(),
        shuffle=False,
        random_state=42,
    )

    documents = {}
    categories = {}

    document_id = 1

    for category_index, category in enumerate(dataset.target_names):

        category_documents = [
            index
            for index, target in enumerate(dataset.target)
            if target == category_index
        ]

        selected_indices = category_documents[
            :documents_per_category
        ]

        for index in selected_indices:
            doc_id = f"D{document_id}"

            documents[doc_id] = dataset.data[index]
            categories[doc_id] = category

            document_id += 1

    return documents, categories


def document_length(text):
    return len(text.split())


def dataset_statistics(documents, categories):

    lengths = [
        document_length(text)
        for text in documents.values()
    ]

    category_counts = Counter(categories.values())

    if lengths:
        average_length = sum(lengths) / len(lengths)
        shortest_length = min(lengths)
        longest_length = max(lengths)
    else:
        average_length = 0
        shortest_length = 0
        longest_length = 0

    return {
        "number_of_documents": len(documents),
        "number_of_categories": len(category_counts),
        "average_document_length": average_length,
        "shortest_document": shortest_length,
        "longest_document": longest_length,
        "documents_per_category": dict(category_counts),
    }


if __name__ == "__main__":
    documents, categories = load_documents(
        documents_per_category=30
    )

    statistics = dataset_statistics(
        documents,
        categories,
    )

    print("=" * 60)
    print("DATASET INFORMATION")
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