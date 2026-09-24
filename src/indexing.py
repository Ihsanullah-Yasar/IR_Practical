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


# ============================================================
# PART 3 - DICTIONARY
# ============================================================

def build_dictionary(processed_documents):
    """
    Build a dictionary from all processed documents.

    Args:
        processed_documents:
            {
                "D1": ["term1", "term2", ...],
                "D2": ["term2", "term3", ...]
            }

    Returns:
        Dictionary: The constructed dictionary.
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
        dict:
            Document ID -> number of unique terms.
    """

    unique_terms = {}

    for document_id, tokens in processed_documents.items():

        unique_terms[document_id] = len(set(tokens))

    return unique_terms


def average_unique_terms_per_document(processed_documents):
    """
    Calculate the average number of unique terms per document.
    """

    if not processed_documents:
        return 0.0

    unique_counts = unique_terms_per_document(
        processed_documents
    )

    total_unique_terms = sum(
        unique_counts.values()
    )

    return total_unique_terms / len(
        processed_documents
    )


def print_dictionary_statistics(
    dictionary,
    processed_documents,
):
    """
    Print Part 3 dictionary statistics.
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
    Demonstrate basic dictionary operations.
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

        term_id = dictionary.get_term_id(
            term
        )

        print(
            f"  {term_id:>4} -> {term}"
        )

    test_term = "fractal"

    print(
        f"\nDoes '{test_term}' exist?"
    )

    print(
        dictionary.term_exists(test_term)
    )

    print(
        f"\nTerm ID for '{test_term}':"
    )

    print(
        dictionary.get_term_id(test_term)
    )

    unknown_term = (
        "thistermdoesnotexist"
    )

    print(
        f"\nDoes '{unknown_term}' exist?"
    )

    print(
        dictionary.term_exists(
            unknown_term
        )
    )


# ============================================================
# PART 4 - INVERTED INDEX
# ============================================================

def build_inverted_index(processed_documents):
    """
    Build an inverted index.

    The inverted index maps each term to a list
    of document IDs containing that term.

    A document appears only once in a posting list,
    regardless of how many times the term occurs.
    """

    inverted_index = {}

    for document_id, tokens in (
        processed_documents.items()
    ):

        seen_terms = set()

        for term in tokens:

            if term in seen_terms:
                continue

            seen_terms.add(term)

            if term not in inverted_index:
                inverted_index[term] = []

            inverted_index[term].append(
                document_id
            )

    return inverted_index


def search_term(term, inverted_index):
    """
    Search the inverted index for a term.

    Returns:
        list:
            Posting list containing document IDs.
    """

    return inverted_index.get(
        term,
        []
    )


def document_frequency(
    term,
    inverted_index,
):
    """
    Calculate document frequency (DF).

    DF is the number of documents containing
    the term.
    """

    return len(
        search_term(
            term,
            inverted_index,
        )
    )


def document_frequency_table(
    inverted_index,
    terms,
):
    """
    Create a document-frequency table.
    """

    table = []

    for term in terms:

        df = document_frequency(
            term,
            inverted_index,
        )

        table.append(
            (term, df)
        )

    return table


def print_inverted_index_statistics(
    inverted_index,
    dictionary=None,
    processed_documents=None,
):
    """
    Print inverted-index statistics.
    """

    print("\n" + "=" * 60)
    print("PART 4 - INVERTED INDEX")
    print("=" * 60)

    if processed_documents is not None:

        print(
            f"\nNumber of documents: "
            f"{len(processed_documents)}"
        )

    print(
        f"Number of indexed terms: "
        f"{len(inverted_index)}"
    )

    total_postings = sum(
        len(document_ids)
        for document_ids in (
            inverted_index.values()
        )
    )

    print(
        f"Total postings: "
        f"{total_postings}"
    )

    if dictionary is not None:

        print(
            f"Dictionary vocabulary size: "
            f"{dictionary.vocabulary_size()}"
        )

        if (
            len(inverted_index)
            == dictionary.vocabulary_size()
        ):

            print(
                "Index/Dictionaries "
                "vocabulary check: PASSED"
            )

        else:

            print(
                "Index/Dictionaries "
                "vocabulary check: FAILED"
            )


def inverted_index_demo(
    inverted_index,
):
    """
    Demonstrate inverted-index searching
    and document frequency.
    """

    print("\n" + "=" * 60)
    print(
        "INVERTED INDEX DEMONSTRATION"
    )
    print("=" * 60)

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

    print(
        "\nDOCUMENT FREQUENCY TABLE"
    )

    print("-" * 40)

    print(
        f"{'Term':<20}{'DF':>10}"
    )

    print("-" * 40)

    table = document_frequency_table(
        inverted_index,
        demonstration_terms,
    )

    for term, df in table:

        print(
            f"{term:<20}{df:>10}"
        )

    print("-" * 40)

    test_term = "fractal"

    postings = search_term(
        test_term,
        inverted_index,
    )

    print(
        f"\nDocuments containing "
        f"'{test_term}':"
    )

    print(postings)

    print(
        f"\nDocument frequency of "
        f"'{test_term}': "
        f"{len(postings)}"
    )

    unknown_term = (
        "thistermdoesnotexist"
    )

    unknown_postings = search_term(
        unknown_term,
        inverted_index,
    )

    print(
        f"\nDocuments containing "
        f"'{unknown_term}':"
    )

    print(unknown_postings)


# ============================================================
# PART 5 - POSITIONAL INDEX
# ============================================================

def build_positional_index(
    processed_documents,
):
    """
    Build a positional index.

    The positional index maps:

        term -> document -> positions

    Position numbering starts at 1.

    Positions are assigned AFTER preprocessing.
    Therefore, removed stop words do not occupy
    positions in the positional index.
    """

    positional_index = {}

    for document_id, tokens in (
        processed_documents.items()
    ):

        for position, term in enumerate(
            tokens,
            start=1,
        ):

            if term not in positional_index:
                positional_index[term] = {}

            if (
                document_id
                not in positional_index[term]
            ):

                positional_index[term][
                    document_id
                ] = []

            positional_index[term][
                document_id
            ].append(position)

    return positional_index


def search_positional_term(
    term,
    positional_index,
):
    """
    Search for a term in the positional index.

    Returns:

        {
            "D1": [10, 20],
            "D8": [7, 18]
        }

    If the term does not exist, returns {}.
    """

    return positional_index.get(
        term,
        {}
    )


def positional_index_statistics(
    positional_index,
    processed_documents,
):
    """
    Calculate basic positional-index statistics.
    """

    total_positions = 0

    for document_positions in (
        positional_index.values()
    ):

        for positions in (
            document_positions.values()
        ):

            total_positions += len(
                positions
            )

    print("\n" + "=" * 60)
    print("PART 5 - POSITIONAL INDEX")
    print("=" * 60)

    print(
        f"\nNumber of documents: "
        f"{len(processed_documents)}"
    )

    print(
        f"Number of indexed terms: "
        f"{len(positional_index)}"
    )

    print(
        f"Total term positions: "
        f"{total_positions}"
    )


def positional_index_demo(
    positional_index,
):
    """
    Demonstrate positional-index operations.
    """

    print("\n" + "=" * 60)
    print(
        "POSITIONAL INDEX DEMONSTRATION"
    )
    print("=" * 60)

    demonstration_terms = [
        "fractal",
        "space",
        "orbit",
    ]

    for term in demonstration_terms:

        postings = search_positional_term(
            term,
            positional_index,
        )

        print(
            f"\nTerm: '{term}'"
        )

        print(
            f"Documents: "
            f"{len(postings)}"
        )

        for index, (
            document_id,
            positions,
        ) in enumerate(
            postings.items()
        ):

            if index >= 5:
                break

            print(
                f"  {document_id}: "
                f"{positions}"
            )

    unknown_term = (
        "thistermdoesnotexist"
    )

    print(
        f"\nUnknown term "
        f"'{unknown_term}':"
    )

    print(
        search_positional_term(
            unknown_term,
            positional_index,
        )
    )


# ============================================================
# PART 6 - PHRASE QUERIES
# ============================================================

def phrase_search(
    phrase,
    positional_index,
):
    """
    Search for an exact phrase using the positional index.

    The phrase is processed using the same preprocessing
    pipeline used for documents.

    Stop words removed during preprocessing do not occupy
    positions.
    """

    from src.preprocessing import preprocess

    # --------------------------------------------------------
    # Step 1:
    # Preprocess the phrase.
    # --------------------------------------------------------

    phrase_terms = preprocess(phrase)

    if not phrase_terms:
        return []

    # --------------------------------------------------------
    # Step 2:
    # Single-term phrase.
    # --------------------------------------------------------

    if len(phrase_terms) == 1:

        term = phrase_terms[0]

        return list(
            search_positional_term(
                term,
                positional_index,
            ).keys()
        )

    # --------------------------------------------------------
    # Step 3:
    # Find documents containing the first term.
    # --------------------------------------------------------

    first_term = phrase_terms[0]

    first_term_documents = (
        search_positional_term(
            first_term,
            positional_index,
        )
    )

    if not first_term_documents:
        return []

    # --------------------------------------------------------
    # Step 4:
    # Check consecutive positions.
    # --------------------------------------------------------

    matching_documents = []

    for document_id, start_positions in (
        first_term_documents.items()
    ):

        for start_position in start_positions:

            current_position = start_position
            phrase_matches = True

            for term in phrase_terms[1:]:

                term_documents = (
                    search_positional_term(
                        term,
                        positional_index,
                    )
                )

                if document_id not in term_documents:

                    phrase_matches = False
                    break

                expected_position = (
                    current_position + 1
                )

                term_positions = (
                    term_documents[
                        document_id
                    ]
                )

                if expected_position not in term_positions:

                    phrase_matches = False
                    break

                current_position = expected_position

            if phrase_matches:

                matching_documents.append(
                    document_id
                )

                break

    return matching_documents


def phrase_search_demo(
    positional_index,
):
    """
    Demonstrate phrase searching.
    """

    print("\n" + "=" * 60)
    print("PART 6 - PHRASE QUERIES")
    print("=" * 60)

    example_phrases = [
        "information retrieval",
        "space exploration",
        "medical information",
        "baseball game",
        "computer graphics",
    ]

    print(
        "\nPHRASE SEARCH RESULTS"
    )

    print("-" * 60)

    for phrase in example_phrases:

        results = phrase_search(
            phrase,
            positional_index,
        )

        print(
            f"\nPhrase: '{phrase}'"
        )

        print(
            f"Matching documents: "
            f"{len(results)}"
        )

        print(results[:10])

    unknown_phrase = (
        "thistermdoesnotexist"
    )

    unknown_results = phrase_search(
        unknown_phrase,
        positional_index,
    )

    print(
        f"\nUnknown phrase: "
        f"'{unknown_phrase}'"
    )

    print(
        f"Matching documents: "
        f"{unknown_results}"
    )


# ============================================================
# PART 7 - TERM FREQUENCY
# ============================================================

def calculate_tf(
    document,
    normalized=False,
):
    """
    Calculate term frequency for a document.

    Raw TF:
        TF(t, d) = count of term t in document d

    Normalized TF:
        TF(t, d) =
            count of term t /
            total number of terms in document d

    Args:
        document:
            List of processed terms.

        normalized:
            If False, return raw term frequencies.
            If True, return normalized term frequencies.

    Returns:
        dict:
            term -> TF value
    """

    term_counts = {}

    for term in document:

        term_counts[term] = (
            term_counts.get(term, 0) + 1
        )

    if not normalized:
        return term_counts

    total_terms = len(document)

    if total_terms == 0:
        return {}

    normalized_tf = {}

    for term, count in term_counts.items():

        normalized_tf[term] = (
            count / total_terms
        )

    return normalized_tf


def print_tf_table(
    document_id,
    document,
    terms,
):
    """
    Print raw and normalized TF values for
    selected terms in a document.
    """

    raw_tf = calculate_tf(
        document,
        normalized=False,
    )

    normalized_tf = calculate_tf(
        document,
        normalized=True,
    )

    print(
        f"\nDocument: {document_id}"
    )

    print(
        f"Total processed terms: "
        f"{len(document)}"
    )

    print(
        "\nTERM FREQUENCY TABLE"
    )

    print("-" * 55)

    print(
        f"{'Term':<20}"
        f"{'Raw TF':>12}"
        f"{'Normalized TF':>20}"
    )

    print("-" * 55)

    for term in terms:

        raw_value = raw_tf.get(
            term,
            0,
        )

        normalized_value = normalized_tf.get(
            term,
            0.0,
        )

        print(
            f"{term:<20}"
            f"{raw_value:>12}"
            f"{normalized_value:>20.6f}"
        )

    print("-" * 55)


def term_frequency_demo(
    processed_documents,
):
    """
    Demonstrate raw and normalized term frequency
    using real processed documents.
    """

    print("\n" + "=" * 60)
    print("PART 7 - TERM FREQUENCY")
    print("=" * 60)

    # --------------------------------------------------------
    # Demonstration 1:
    # Use D1 and inspect selected terms.
    # --------------------------------------------------------

    document_id = "D1"

    document = processed_documents[
        document_id
    ]

    demonstration_terms = [
        "fractal",
        "comput",
        "graphic",
        "space",
        "orbit",
    ]

    print(
        "\nRAW TF AND NORMALIZED TF"
    )

    print_tf_table(
        document_id,
        document,
        demonstration_terms,
    )

    # --------------------------------------------------------
    # Demonstration 2:
    # Verify that normalized TF values sum to 1.
    #
    # Because normalized TF is count / total terms,
    # the values of all unique terms should sum to 1.
    # --------------------------------------------------------

    normalized_tf = calculate_tf(
        document,
        normalized=True,
    )

    normalized_tf_sum = sum(
        normalized_tf.values()
    )

    print(
        f"\nSum of normalized TF values "
        f"for {document_id}: "
        f"{normalized_tf_sum:.6f}"
    )

    # --------------------------------------------------------
    # Demonstration 3:
    # Empty document.
    # --------------------------------------------------------

    empty_document = []

    print(
        "\nEmpty document TF:"
    )

    print(
        calculate_tf(
            empty_document
        )
    )

    print(
        "\nEmpty document normalized TF:"
    )

    print(
        calculate_tf(
            empty_document,
            normalized=True,
        )
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    import os
    import sys

    # --------------------------------------------------------
    # Add project root to Python path.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Import Part 1 and Part 2 functionality.
    # --------------------------------------------------------

    from src.data_loader import (
        load_documents,
    )

    from src.preprocessing import (
        preprocess_documents,
    )

    # --------------------------------------------------------
    # Load documents.
    # --------------------------------------------------------

    documents, categories = (
        load_documents(
            documents_per_category=30
        )
    )

    # --------------------------------------------------------
    # Preprocess documents.
    # --------------------------------------------------------

    processed_documents = (
        preprocess_documents(
            documents
        )
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

    print_inverted_index_statistics(
        inverted_index,
        dictionary=dictionary,
        processed_documents=processed_documents,
    )

    inverted_index_demo(
        inverted_index
    )

    # ========================================================
    # PART 5 - POSITIONAL INDEX
    # ========================================================

    positional_index = (
        build_positional_index(
            processed_documents
        )
    )

    positional_index_statistics(
        positional_index,
        processed_documents,
    )

    positional_index_demo(
        positional_index
    )

    # ========================================================
    # PART 6 - PHRASE QUERIES
    # ========================================================

    phrase_search_demo(
        positional_index
    )

    # ========================================================
    # PART 7 - TERM FREQUENCY
    # ========================================================

    term_frequency_demo(
        processed_documents
    )