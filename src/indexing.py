class Dictionary:
    """
    Dictionary for the Information Retrieval system.

    The dictionary maps every unique normalized term
    to a unique integer term ID.
    """

    def __init__(self):
        self.term_to_id = {}
        self.id_to_term = {}

    def add_term(self, term):
        """
        Add a term to the dictionary if it does not already exist.

        Returns:
            int: The term ID.
        """
        if term not in self.term_to_id:
            term_id = len(self.term_to_id) + 1

            self.term_to_id[term] = term_id
            self.id_to_term[term_id] = term

        return self.term_to_id[term]

    def term_exists(self, term):
        """
        Check whether a term exists in the dictionary.

        Returns:
            bool: True if the term exists, otherwise False.
        """
        return term in self.term_to_id

    def get_term_id(self, term):
        """
        Get the term ID for a given term.

        Returns:
            int or None: Term ID if the term exists.
        """
        return self.term_to_id.get(term)

    def get_term(self, term_id):
        """
        Get the term associated with a term ID.

        Returns:
            str or None: Term if the ID exists.
        """
        return self.id_to_term.get(term_id)

    def vocabulary_size(self):
        """
        Return the total number of unique terms.
        """
        return len(self.term_to_id)

    def terms(self):
        """
        Return all terms in dictionary insertion order.
        """
        return self.term_to_id.keys()


def build_dictionary(processed_documents):
    """
    Build a dictionary from all processed documents.

    Args:
        processed_documents: Dictionary in the form:
            {
                "D1": ["term1", "term2", ...],
                "D2": ["term2", "term3", ...]
            }

    Returns:
        Dictionary: The constructed term dictionary.
    """
    dictionary = Dictionary()

    for document_id, tokens in processed_documents.items():
        for term in tokens:
            dictionary.add_term(term)

    return dictionary


def unique_terms_per_document(processed_documents):
    """
    Calculate the number of unique terms in each document.

    Returns:
        dict: Document ID -> number of unique terms.
    """
    unique_terms = {}

    for document_id, tokens in processed_documents.items():
        unique_terms[document_id] = len(set(tokens))

    return unique_terms


def average_unique_terms_per_document(processed_documents):
    """
    Calculate the average number of unique terms per document.

    Returns:
        float: Average unique terms per document.
    """
    if not processed_documents:
        return 0.0

    unique_counts = unique_terms_per_document(processed_documents)

    total_unique_terms = sum(unique_counts.values())

    return total_unique_terms / len(processed_documents)


def print_dictionary_statistics(dictionary, processed_documents):
    """
    Print Part 3 dictionary statistics.
    """
    average_unique_terms = average_unique_terms_per_document(
        processed_documents
    )

    print("\n" + "=" * 60)
    print("PART 3 - DICTIONARY")
    print("=" * 60)

    print(f"\nNumber of documents: {len(processed_documents)}")
    print(f"Vocabulary size: {dictionary.vocabulary_size()}")

    print(
        f"Average unique terms per document: "
        f"{average_unique_terms:.2f}"
    )


def dictionary_demo(dictionary):
    """
    Demonstrate basic dictionary operations.
    """
    print("\n" + "=" * 60)
    print("DICTIONARY DEMONSTRATION")
    print("=" * 60)

    print("\nFirst 20 dictionary terms:")

    for index, term in enumerate(dictionary.terms()):
        if index >= 20:
            break

        term_id = dictionary.get_term_id(term)

        print(f"  {term_id:>4} -> {term}")

    test_term = "fractal"

    print(f"\nDoes '{test_term}' exist?")
    print(dictionary.term_exists(test_term))

    print(f"\nTerm ID for '{test_term}':")
    print(dictionary.get_term_id(test_term))

    unknown_term = "thistermdoesnotexist"

    print(f"\nDoes '{unknown_term}' exist?")
    print(dictionary.term_exists(unknown_term))


# ============================================================
# PART 4 - INVERTED INDEX
# ============================================================

def build_inverted_index(processed_documents):
    """
    Build an inverted index.

    The inverted index maps each term to a list
    of document IDs containing that term.

    Example:

        {
            "fractal": ["D1", "D7", "D15"],
            "space": ["D4", "D8", "D20"]
        }

    A document appears only once in a posting list,
    regardless of how many times the term occurs in
    that document.
    """

    inverted_index = {}

    for document_id, tokens in processed_documents.items():

        # Keep track of terms already added for this document.
        # This prevents duplicate document IDs in a posting list.
        seen_terms = set()

        for term in tokens:

            if term in seen_terms:
                continue

            seen_terms.add(term)

            if term not in inverted_index:
                inverted_index[term] = []

            inverted_index[term].append(document_id)

    return inverted_index


def search_term(term, inverted_index):
    """
    Search the inverted index for a term.

    Args:
        term: The normalized/stemmed term to search for.
        inverted_index: The inverted index.

    Returns:
        list: Posting list containing document IDs.
    """
    return inverted_index.get(term, [])


def document_frequency(term, inverted_index):
    """
    Calculate document frequency (DF) for a term.

    DF is the number of documents containing the term.

    Formula:

        DF(t) = number of documents containing term t
    """
    return len(search_term(term, inverted_index))


def document_frequency_table(inverted_index, terms):
    """
    Create a document-frequency table.

    Args:
        inverted_index: The inverted index.
        terms: List of terms.

    Returns:
        list of tuples:
            [
                ("term1", df1),
                ("term2", df2),
                ...
            ]
    """
    table = []

    for term in terms:
        df = document_frequency(term, inverted_index)

        table.append((term, df))

    return table


def print_inverted_index_statistics(
    inverted_index,
    dictionary=None,
    processed_documents=None,
):
    """
    Print inverted-index statistics.

    If a dictionary is supplied, vocabulary size can be
    compared with the number of indexed terms.

    If processed documents are supplied, the number of
    documents can also be displayed.
    """

    print("\n" + "=" * 60)
    print("PART 4 - INVERTED INDEX")
    print("=" * 60)

    if processed_documents is not None:
        print(f"\nNumber of documents: {len(processed_documents)}")

    print(f"Number of indexed terms: {len(inverted_index)}")

    total_postings = sum(
        len(document_ids)
        for document_ids in inverted_index.values()
    )

    print(f"Total postings: {total_postings}")

    if dictionary is not None:
        print(f"Dictionary vocabulary size: {dictionary.vocabulary_size()}")

        if len(inverted_index) == dictionary.vocabulary_size():
            print("Index/Dictionaries vocabulary check: PASSED")
        else:
            print("Index/Dictionaries vocabulary check: FAILED")


def inverted_index_demo(inverted_index):
    """
    Demonstrate inverted-index searching and document frequency.
    """

    print("\n" + "=" * 60)
    print("INVERTED INDEX DEMONSTRATION")
    print("=" * 60)

    # These terms are selected from the processed vocabulary.
    demonstration_terms = [
        "fractal",
        "space",
        "orbit",
        "game",
        "medic",
        "comput",
        "graphic",
        "basebal",
        "polit",
        "govern",
    ]

    print("\nDOCUMENT FREQUENCY TABLE")
    print("-" * 40)
    print(f"{'Term':<20}{'DF':>10}")
    print("-" * 40)

    table = document_frequency_table(
        inverted_index,
        demonstration_terms,
    )

    for term, df in table:
        print(f"{term:<20}{df:>10}")

    print("-" * 40)

    # Demonstrate a term search.
    test_term = "fractal"

    postings = search_term(
        test_term,
        inverted_index,
    )

    print(f"\nDocuments containing '{test_term}':")

    print(postings)

    print(
        f"\nDocument frequency of '{test_term}': "
        f"{len(postings)}"
    )

    # Demonstrate a term that does not exist.
    unknown_term = "thistermdoesnotexist"

    unknown_postings = search_term(
        unknown_term,
        inverted_index,
    )

    print(
        f"\nDocuments containing '{unknown_term}':"
    )

    print(unknown_postings)


# ============================================================
# PART 4 TEST / DEMONSTRATION
# ============================================================

if __name__ == "__main__":

    import os
    import sys

    # --------------------------------------------------------
    # Add project root to Python path.
    #
    # This allows us to run:
    #
    #     python src/indexing.py
    #
    # directly from the project root.
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # --------------------------------------------------------
    # Import Part 1 and Part 2 functionality.
    # --------------------------------------------------------

    from src.data_loader import load_documents

    from src.preprocessing import (
        preprocess_documents,
    )

    # --------------------------------------------------------
    # Load documents.
    # --------------------------------------------------------

    documents, categories = load_documents(
        documents_per_category=30
    )

    # --------------------------------------------------------
    # Preprocess documents.
    # --------------------------------------------------------

    processed_documents = preprocess_documents(
        documents
    )

    # --------------------------------------------------------
    # PART 3 - Build dictionary.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PART 4 - Build inverted index.
    # --------------------------------------------------------

    inverted_index = build_inverted_index(
        processed_documents
    )

    # --------------------------------------------------------
    # Print inverted-index statistics.
    # --------------------------------------------------------

    print_inverted_index_statistics(
        inverted_index,
        dictionary=dictionary,
        processed_documents=processed_documents,
    )

    # --------------------------------------------------------
    # Demonstrate inverted-index operations.
    # --------------------------------------------------------

    inverted_index_demo(
        inverted_index
    )