from sklearn.datasets import fetch_20newsgroups


CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc",
]


def load_documents(
    categories=CATEGORIES,
    documents_per_category=30,
    random_state=42
):
    dataset = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=random_state,
    )

    documents = {}
    category_map = {}

    counts = {
        category: 0
        for category in categories
    }

    for text, target in zip(
        dataset.data,
        dataset.target
    ):
        category = dataset.target_names[target]

        if counts[category] >= documents_per_category:
            continue
        
        # Skip documents that became empty after
        # removing headers, footers, and quotes.
        if not text.strip():
            continue

        document_id = f"D{len(documents) + 1}"

        documents[document_id] = text
        category_map[document_id] = category

        counts[category] += 1

        if all(
            count == documents_per_category
            for count in counts.values()
        ):
            break

    return documents, category_map


def dataset_statistics(documents, categories):
    lengths = [
        len(text.split())
        for text in documents.values()
    ]

    return {
        "number_of_documents": len(documents),

        "number_of_categories": len(
            set(categories.values())
        ),

        "average_document_length": (
            sum(lengths) / len(lengths)
            if lengths
            else 0
        ),

        "shortest_document": (
            min(lengths)
            if lengths
            else 0
        ),

        "longest_document": (
            max(lengths)
            if lengths
            else 0
        ),

        "documents_per_category": {
            category: sum(
                1
                for c in categories.values()
                if c == category
            )
            for category in sorted(
                set(categories.values())
            )
        },
    }