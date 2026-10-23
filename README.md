# Evaluate Your Search System

Evaluates ChromaDB semantic search quality using **precision** and **recall** across different similarity thresholds and result counts.

## Overview

This project builds a small knowledge base in ChromaDB, tests it against a hand-labeled evaluation set, and measures how well the search finds the right documents under different settings.

- **14 documents** across 4 topics: APIs/FastAPI, Streamlit, embeddings/vector search, and web frontend
- **7 test queries**, each with a list of expected relevant document IDs
- **3 settings** compared by varying `threshold` and `n_results`

## Key Concepts

- **Precision**: the fraction of retrieved documents that are relevant
- **Recall**: the fraction of relevant documents that are retrieved
- **Threshold**: the minimum similarity score a result needs to be kept
- **n_results**: the maximum number of documents returned per search

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Run

```bash
python my_eval.py
```

The first run downloads ChromaDB's default embedding model.

## Results

| Threshold | n_results | Avg Precision | Avg Recall |
|---|---|---|---|
| 0.3 | 3 | 85.7% | 100.0% |
| 0.5 | 3 | 57.1% | 35.7% |
| 0.2 | 5 | 52.1% | 100.0% |

**Best setting:** threshold 0.3 with 3 results. It found every relevant document while letting in only a few extras.

## Findings

- **Worked well:** Query 4 spans two topics (Streamlit layout and CSS Flexbox) and still scored perfectly at 0.3 and 0.2.
- **Failed:** Queries 1, 2, and 7 returned nothing at threshold 0.5. Their wording differs from the documents, so their best matches scored between 0.3 and 0.5.
- **Tradeoff:** A strict threshold kept returned results correct but dropped relevant ones. A loose threshold found everything but let in unrelated documents.

## Possible Improvements

1. Test thresholds between 0.3 and 0.4
2. Add more detail and synonyms to short documents
3. Use ChromaDB metadata filtering by topic
4. Try a larger embedding model

## Tech Stack

- Python
- ChromaDB (in-memory client, cosine distance)