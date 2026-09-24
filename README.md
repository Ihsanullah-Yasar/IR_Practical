# Information Retrieval Practical — Classical IR System

A classical **Information Retrieval (IR)** system implemented from scratch in Python for the Information Retrieval practical assignment.

The system uses the **20 Newsgroups** dataset and implements the complete retrieval pipeline:

```text
Documents
    ↓
Preprocessing
    ↓
Dictionary
    ↓
Inverted Index / Positional Index
    ↓
TF-IDF
    ↓
Query Representation
    ↓
Cosine Similarity
    ↓
Ranked Search
```

## 1. Project Overview

This project implements a small classical Information Retrieval system without using high-level search or retrieval frameworks.

The system supports:

- Document collection loading
- Text preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Porter stemming
- Vocabulary/dictionary construction
- Inverted indexing
- Positional indexing
- Term queries
- Phrase queries
- Term Frequency (TF)
- Inverse Document Frequency (IDF)
- TF-IDF weighting
- Query representation
- Vector creation
- Cosine similarity
- Ranked document retrieval
- Interactive command-line search
- Retrieval experiments
- Final evaluation

The implementation follows the requirements of the practical assignment and focuses on fundamental classical IR techniques.

---

## 2. Dataset

The project uses the **20 Newsgroups** dataset provided through scikit-learn.

Five categories are used:

```text
comp.graphics
rec.sport.baseball
sci.med
sci.space
talk.politics.misc
```

The implementation loads **30 documents from each category**, resulting in:

| Statistic                   |        Value |
| --------------------------- | -----------: |
| Total documents             |          150 |
| Categories                  |            5 |
| Documents per category      |           30 |
| Average raw document length | 330.55 terms |
| Shortest raw document       |     42 terms |
| Longest raw document        |  7,989 terms |

---

## 3. System Architecture

The main retrieval pipeline is:

```text
                 20 Newsgroups
                       │
                       ▼
              Document Collection
                       │
                       ▼
                 Preprocessing
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Lowercase      Tokenization   Stopword Removal
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼
                    Stemming
                       │
                       ▼
                Processed Terms
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Dictionary        Inverted Index
                                 │
                          Positional Index
             │                   │
             └─────────┬─────────┘
                       ▼
                    TF-IDF
                       │
                       ▼
                Document Vectors
                       │
               Query Processing
                       │
                       ▼
                  Query TF-IDF
                       │
                       ▼
                  Query Vector
                       │
                       ▼
              Cosine Similarity
                       │
                       ▼
                Ranked Results
```

---

## 4. Text Preprocessing

The preprocessing pipeline is:

```text
Raw Text
   ↓
Lowercasing
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Porter Stemming
   ↓
Processed Terms
```

### Tokenization

Alphabetic word tokens are extracted from the text.

Punctuation, numbers, and other non-alphabetic characters are excluded.

### Stopword Removal

Common English stopwords are removed using NLTK.

### Stemming

The NLTK **Porter Stemmer** is used to reduce related word forms to a common stem.

Example:

```text
connected    → connect
connecting   → connect
connection   → connect
connections  → connect
computers    → comput
computing    → comput
```

The same preprocessing pipeline is applied to both documents and user queries.

After preprocessing, the final vocabulary contains:

```text
5,912 unique terms
```

---

## 5. Dictionary

The dictionary maps each unique processed term to a numeric term ID.

Example:

```text
1  → davidr
2  → rincon
3  → ema
4  → rockwel
5  → com
...
```

The final vocabulary size is:

```text
5,912 terms
```

The implementation also calculates the average number of unique terms per document.

---

## 6. Inverted Index

The inverted index maps each term to the documents containing that term.

Conceptually:

```text
term → [document IDs]
```

Example:

```text
fractal → [D1, D8, D10, D15]
```

Therefore:

```text
DF(fractal) = 4
```

The inverted index is used to identify documents containing query terms.

---

## 7. Positional Index

The positional index extends the inverted index by storing the positions of terms inside each document.

Structure:

```text
term → document → positions
```

Example:

```text
fractal
    ├── D1  → [10, 20, 23, 48]
    ├── D8  → [24]
    ├── D10 → [8]
    └── D15 → [9, 52, 59, ...]
```

Positions are assigned **after preprocessing**, so removed stopwords do not occupy positions.

Positions start from `1`.

---

## 8. Phrase Queries

Phrase queries use the positional index.

For example:

```text
computer graphics
```

The system checks whether the processed terms occur consecutively in the same document.

The phrase-search algorithm:

1. Preprocess the phrase.
2. Find documents containing the first term.
3. Check the positions of subsequent terms.
4. Require consecutive positions.
5. Return matching document IDs.

Example:

```text
Phrase: computer graphics

Matching documents:
D14
D26
```

Because phrase matching is performed after preprocessing, the phrase terms are compared using their processed/stemmed forms.

---

## 9. Term Frequency

Raw Term Frequency is calculated as:

```text
TF(t,d) = count(t,d)
```

Normalized Term Frequency is:

```text
TF(t,d) = count(t,d) / |d|
```

where:

- `t` = term
- `d` = document
- `|d|` = total number of processed terms in the document

Normalized TF is used in the TF-IDF calculation.

---

## 10. Inverse Document Frequency

IDF is calculated using:

```text
IDF(t) = log(N / DF(t))
```

where:

- `N` = total number of documents
- `DF(t)` = number of documents containing term `t`

For this collection:

```text
N = 150
```

Example:

```text
IDF(fractal)
= log(150 / 4)
= 3.624341
```

Terms occurring in many documents receive lower IDF values, while less common terms receive higher IDF values.

---

## 11. TF-IDF

The TF-IDF weight is:

```text
TFIDF(t,d) = TF(t,d) × IDF(t)
```

Example:

```text
TF(fractal, D1) = 0.040404
IDF(fractal)    = 3.624341

TFIDF(fractal,D1)
= 0.040404 × 3.624341
= 0.146438
```

The system creates a sparse TF-IDF representation for every document.

---

## 12. Query Representation

Queries go through the same preprocessing pipeline as documents.

Example:

```text
Original query:
computer graphics

Processed query:
comput graphic
```

The query is then represented using TF-IDF weights based on the same document collection.

This ensures that document and query vectors use the same vocabulary and weighting scheme.

---

## 13. Vector Creation

Each document and query is represented as a vector whose dimension equals the dictionary vocabulary size.

Current vector dimension:

```text
5,912
```

Term IDs are mapped to vector positions.

NumPy is used for numerical vector operations.

---

## 14. Cosine Similarity

Document ranking is based on cosine similarity:

```text
              q · d
cos(θ) = --------------
            ||q|| ||d||
```

where:

- `q` = query vector
- `d` = document vector
- `q · d` = dot product
- `||q||` = query vector magnitude
- `||d||` = document vector magnitude

A zero-vector check is included to avoid invalid division.

Example:

```text
Query: fractal
Document: D1

Cosine similarity = 0.341174
```

---

## 15. Ranked Search

For a query, the system:

1. Preprocesses the query.
2. Creates the query TF-IDF representation.
3. Creates the query vector.
4. Calculates cosine similarity against document vectors.
5. Removes zero-score results.
6. Sorts documents by descending similarity.
7. Returns the top `k` documents.

Each result contains:

- Rank
- Document ID
- Similarity score
- Category

Example:

```text
SEARCH QUERY: 'space orbit'

Rank   Document   Score       Category
1      D95        0.286791    sci.space
2      D94        0.259213    sci.space
3      D96        0.205946    sci.space
4      D113       0.174378    sci.space
5      D100       0.143766    sci.space
```

---

## 16. Command-Line Interface

The system provides an interactive command-line search interface.

Run:

```bash
python main.py
```

Example:

```text
Enter your query: space orbit

SEARCH QUERY: 'space orbit'

Rank   Document   Score       Category
1      D95        0.286791    sci.space
2      D94        0.259213    sci.space
3      D96        0.205946    sci.space
...
```

Type:

```text
exit
```

to terminate the application.

---

## 17. Experiments

Part 14 contains three controlled experiments.

### Experiment 1 — Stopword Removal

Comparison:

| Metric                 | Without Stopwords | With Stopwords |
| ---------------------- | ----------------: | -------------: |
| Vocabulary             |             6,038 |          5,912 |
| Average terms/document |            337.31 |         189.66 |
| Total processed terms  |            50,596 |         28,449 |

Test query:

```text
the space exploration
```

For this query, the top-five document ordering remained unchanged, although the similarity scores changed slightly.

---

### Experiment 2 — Stemming

Comparison:

| Metric                 | Without Stemming | With Stemming |
| ---------------------- | ---------------: | ------------: |
| Vocabulary             |            9,035 |         5,912 |
| Average terms/document |           210.47 |        189.66 |
| Total processed terms  |           31,571 |        28,449 |

Stemming reduced the vocabulary by:

```text
3,123 terms
```

Test query:

```text
computer computing computers
```

The non-stemmed system returned no results for this query, while the stemmed system retrieved documents because related word forms were mapped to the common stem:

```text
comput
```

---

### Experiment 3 — Query Length

Three query lengths were compared.

**One-word query:**

```text
fractal
```

**Two-word query:**

```text
space orbit
```

**Longer query:**

```text
space orbit nasa launch
```

The resulting document rankings and similarity scores changed as additional query terms were introduced.

This demonstrates that query length and query composition affect the TF-IDF query representation and consequently the cosine similarity ranking.

---

## 18. Final Evaluation

The final evaluation uses **10 queries** and records the top five retrieval results where available.

### Evaluation Queries

```text
1. fractal
2. computer graphics
3. baseball game
4. pitcher season
5. medical information
6. health disease
7. space orbit
8. nasa launch
9. government policy
10. political rights
```

Evaluation configuration:

| Item                      | Value |
| ------------------------- | ----: |
| Documents                 |   150 |
| Vocabulary                | 5,912 |
| Evaluation queries        |    10 |
| Document vector dimension | 5,912 |

The complete recorded results are available in:

```text
reports/part15_evaluation_results.txt
```

The evaluation queries cover all five selected Newsgroups categories:

```text
comp.graphics
rec.sport.baseball
sci.med
sci.space
talk.politics.misc
```

Note that the evaluation is a retrieval demonstration rather than a formal precision/recall evaluation because no manually labeled relevance judgments were created.

---

## 19. Project Structure

```text
IR_Practical/
│
├── reports/
│   └── part15_evaluation_results.txt
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── retrieval.py
│   ├── experiments.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 20. Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ihsanullah-Yasar/IR_Practical.git
cd IR_Practical
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For **Git Bash on Windows**:

```bash
source .venv/Scripts/activate
```

For **Windows Command Prompt**:

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- Python 3.10+
- NumPy
- scikit-learn
- NLTK
- Jupyter

---

## 21. Running the System

### Main interactive search

```bash
python main.py
```

### Run Part 14 experiments

```bash
python src/experiments.py
```

### Run Part 15 final evaluation

```bash
python src/evaluation.py
```

The final evaluation results are written to:

```text
reports/part15_evaluation_results.txt
```

---

## 22. Required Functions

The implementation provides the required core functions from the practical assignment:

```text
load_documents()
tokenize()
remove_stopwords()
stem()
preprocess()
build_dictionary()
build_inverted_index()
build_positional_index()
search_term()
phrase_search()
calculate_tf()
calculate_idf()
calculate_tfidf()
create_vector()
cosine_similarity()
search()
```

---

## 23. Design Constraints

This project intentionally implements the main IR components manually.

The retrieval system does **not** use:

```text
TfidfVectorizer
```

or a high-level search/retrieval framework for document ranking.

Instead:

- NumPy is used for numerical vector operations.
- scikit-learn is used to obtain the 20 Newsgroups dataset.
- NLTK is used for stopword removal and Porter stemming.
- TF-IDF weighting is implemented manually.
- Cosine similarity is implemented manually.
- Inverted and positional indexes are implemented manually.

This design follows the educational purpose of the assignment.

---

## 24. Limitations

This is an educational classical Information Retrieval implementation.

Current limitations include:

- Small document collection of 150 documents.
- Ranking is based on TF-IDF and cosine similarity only.
- No learning-to-rank model is used.
- No semantic embeddings are used.
- No query expansion is implemented.
- No relevance feedback is implemented.
- The preprocessing pipeline does not perform advanced linguistic analysis.
- The system does not explicitly remove email headers or metadata from the original Newsgroups documents.
- Phrase retrieval is limited to the implemented positional matching approach.

These limitations are intentional because the assignment focuses on implementing fundamental classical IR concepts.

---

## 25. Learning Outcomes

Through this implementation, the following Information Retrieval concepts were implemented and demonstrated:

- Text preprocessing
- Tokenization
- Lowercasing
- Stopword removal
- Stemming
- Vocabulary construction
- Dictionary indexing
- Inverted indexing
- Positional indexing
- Phrase searching
- Term Frequency
- Inverse Document Frequency
- TF-IDF weighting
- Vector-space representation
- Query processing
- Cosine similarity
- Ranked retrieval
- Retrieval experiments
- Search evaluation

---

## 26. Author

**Ahsanullah Faizy**

Information Retrieval Practical Assignment

**2026**
