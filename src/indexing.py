class Dictionary:
    """
    Dictionary for the Information Retrieval system.

    The dictionary maps every unique normalized term
    to a unique integer term ID.
    """

    def __init__(self):
        """
        Initialize an empty dictionary.
        """

        self.term_to_id = {}
        self.id_to_term = {}

    def add_term(self, term):
        """
        Add a term to the dictionary if it does not
        already exist.

        Args:
            term (str):
                Normalized/stemmed term.

        Returns:
            int:
                Term ID.
        """

        if term not in self.term_to_id:

            term_id = len(self.term_to_id) + 1

            self.term_to_id[term] = term_id
            self.id_to_term[term_id] = term

        return self.term_to_id[term]

    def term_exists(self, term):
        """
        Check whether a term exists in the dictionary.
        """

        return term in self.term_to_id

    def get_term_id(self, term):
        """
        Return the term ID of a term.

        Returns None if the term does not exist.
        """

        return self.term_to_id.get(term)

    def get_term(self, term_id):
        """
        Return the term associated with a term ID.

        Returns None if the ID does not exist.
        """

        return self.id_to_term.get(term_id)

    def vocabulary_size(self):
        """
        Return the total number of unique terms.
        """

        return len(self.term_to_id)

    def terms(self):
        """
        Return all terms in the dictionary.
        """

        return self.term_to_id.keys()


def build_dictionary(processed_documents):
    """
    Build the dictionary from all preprocessed documents.

    Terms receive IDs according to the order in which
    they are first encountered.
    """

    dictionary = Dictionary()

    for document_id, tokens in processed_documents.items():

        for term in tokens:
            dictionary.add_term(term)

    return dictionary


def unique_terms_per_document(processed_documents):
    """
    Calculate the number of unique terms in every document.

    Returns:
        dict:
            {
                "D1": number_of_unique_terms,
                "D2": number_of_unique_terms,
                ...
            }
    """

    unique_terms = {}

    for document_id, tokens in processed_documents.items():

        unique_terms[document_id] = len(set(tokens))

    return unique_terms


def average_unique_terms_per_document(processed_documents):
    """
    Calculate the average number of unique terms
    per document.
    """

    if not processed_documents:
        return 0.0

    unique_counts = unique_terms_per_document(
        processed_documents
    )

    total_unique_terms = sum(
        unique_counts.values()
    )

    return (
        total_unique_terms
        / len(processed_documents)
    )


def print_dictionary_statistics(
    dictionary,
    processed_documents,
):
    """
    Display the dictionary statistics required
    by Part 3 of the assignment.
    """

    average_unique_terms = (
        average_unique_terms_per_document(
            processed_documents
        )
    )

    print("\n" + "=" * 60)
    print("PART 3 - DICTIONARY")
    print("=" * 60)

    print(
        f"\nNumber of documents: "
        f"{len(processed_documents)}"
    )

    print(
        f"Vocabulary size: "
        f"{dictionary.vocabulary_size()}"
    )

    print(
        f"Average unique terms per document: "
        f"{average_unique_terms:.2f}"
    )


def dictionary_demo(dictionary):
    """
    Demonstrate the dictionary functionality.
    """

    print("\n" + "=" * 60)
    print("DICTIONARY DEMONSTRATION")
    print("=" * 60)

    print("\nFirst 20 dictionary terms:")

    for index, term in enumerate(
        dictionary.terms()
    ):

        if index >= 20:
            break

        term_id = dictionary.get_term_id(term)

        print(
            f"  {term_id:>4} -> {term}"
        )

    # Test term existence
    test_term = "fractal"

    print(
        f"\nDoes '{test_term}' exist?"
    )

    print(
        dictionary.term_exists(test_term)
    )

    # Get term ID
    print(
        f"\nTerm ID for '{test_term}':"
    )

    print(
        dictionary.get_term_id(test_term)
    )

    # Test an unknown term
    unknown_term = "thistermdoesnotexist"

    print(
        f"\nDoes '{unknown_term}' exist?"
    )

    print(
        dictionary.term_exists(unknown_term)
    )


if __name__ == "__main__":

    # This block allows the file to be executed
    # directly using:
    #
    #     python src/indexing.py
    #
    # while the normal application can still use:
    #
    #     python main.py

    import os
    import sys

    # Add the project root directory to Python's path.
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    if project_root not in sys.path:
        sys.path.insert(
            0,
            project_root
        )

    from src.data_loader import (
        load_documents
    )

    from src.preprocessing import (
        preprocess_documents
    )

    documents, categories = load_documents(
        documents_per_category=30
    )

    processed_documents = preprocess_documents(
        documents
    )

    dictionary = build_dictionary(
        processed_documents
    )

    print_dictionary_statistics(
        dictionary,
        processed_documents
    )

    dictionary_demo(
        dictionary
    )