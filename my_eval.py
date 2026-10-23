# my_eval.py 
# Evaluates ChromaDB search quality using precision and recall
# across different threshold and n_results settings.
# Precision = the fraction of retrieved documents that are relevant.
# Recall = the fraction of relevant documents that are retrieved.


import chromadb

# Create an in-memory client (starts fresh every time the script is run)
client = chromadb.Client()

# Create a collection that measures cosine distance (similarity = 1 - distance)
collection = client.create_collection(
    name="eval_docs",
    metadata={"hnsw:space": "cosine"},
)

# Documents being added to the collection
# --- Knowledge base: 14 documents across 4 topics ---
documents = [
    # Topic 1: APIs / FastAPI
    {
        "id": "api_jwt",
        "topic": "api",
        "text": "JWT tokens provide stateless authentication for REST APIs. The server creates a signed token containing user info.",
    },
    {
        "id": "api_pydantic",
        "topic": "api",
        "text": "Pydantic models define data schemas for FastAPI. They automatically validate incoming request data.",
    },
    {
        "id": "api_cors",
        "topic": "api",
        "text": "CORS middleware in FastAPI allows cross-origin requests from frontend applications running on different ports.",
    },
    {
        "id": "api_status",
        "topic": "api",
        "text": "HTTP status codes tell the client what happened. 200 means success, 404 means not found, and 401 means unauthorized.",
    },
    # Topic 2: Streamlit
    {
        "id": "st_session",
        "topic": "streamlit",
        "text": "Streamlit session state persists data across re-runs. Initialize with: if key not in st.session_state.",
    },
    {
        "id": "st_widgets",
        "topic": "streamlit",
        "text": "Streamlit widgets like st.button, st.slider, and st.text_input collect user input and trigger a re-run.",
    },
    {
        "id": "st_layout",
        "topic": "streamlit",
        "text": "Streamlit layouts use st.columns, st.tabs, and st.sidebar to organize content on the page.",
    },
    # Topic 3: Embeddings / Vector Search
    {
        "id": "vec_embed",
        "topic": "search",
        "text": "Embeddings convert text into numerical vectors capturing semantic meaning. Similar texts get similar vectors.",
    },
    {
        "id": "vec_cosine",
        "topic": "search",
        "text": "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction.",
    },
    {
        "id": "vec_chunk",
        "topic": "search",
        "text": "Chunking splits documents into smaller pieces for embedding. Chunk size affects search precision and recall.",
    },
    {
        "id": "vec_chroma",
        "topic": "search",
        "text": "ChromaDB is a vector database for storing and querying embeddings. It supports metadata filtering.",
    },
    # Topic 4: Web Frontend
    {
        "id": "web_flex",
        "topic": "frontend",
        "text": "CSS Flexbox arranges elements in rows or columns. Use display:flex on the container.",
    },
    {
        "id": "web_dom",
        "topic": "frontend",
        "text": "The DOM is the browser's tree representation of HTML. JavaScript uses it to modify page content.",
    },
    {
        "id": "web_fetch",
        "topic": "frontend",
        "text": "The JavaScript fetch function sends HTTP requests to an API and returns a promise with the response.",
    },
]

# --- Add EVERYTHING to ChromaDB ---
collection.add(
    ids=[d["id"] for d in documents],  # Means give me the id of every document
    documents=[d["text"] for d in documents],
    metadatas=[{"topic": d["topic"]} for d in documents],
)

print(f"Loaded {collection.count()} documents into ChromaDB.")

# --- Evaluation set (the "answer key") ---
# Each query is phrased differently from the docs on purpose,
# to test the semantic understanding rather than just the "keywords" matching
# Queries 4 and 7 deliberately span two topics to test harder cases
# Also I labeled a doc as "relevant" only if it actually answers the question,
# not just if it shared a keyword in the text.
eval_set = [
    {
        # query 1
        "query": "How do I secure my API so only logged-in users can access it?",
        "relevant": ["api_jwt"], 
    },
    {
        # query 2
        "query": "How do I validate the data sent to my API?",
        "relevant": ["api_pydantic"],
    },
    {
        # query 3
        "query": "How can a Streamlit app remember values after a button click?",
        "relevant": ["st_session", "st_widgets"],
    },
    {
        # query 4
        "query": "How do I arrange content side by side on a page?",
        "relevant": ["st_layout", "web_flex"],
    },
    {
        # query 5
        "query": "How does semantic search find similar text?",
        "relevant": ["vec_embed", "vec_cosine"],
    },
    {
        # query 6
        "query": "Where should I store embeddings and how big should my pieces of text be?",
        "relevant": ["vec_chroma", "vec_chunk"],
    },
    {
        # query 7
        "query": "How does my frontend talk to the backend?",
        "relevant": ["web_fetch", "api_cors"],
    },
]

print(f"Evaluation set has {len(eval_set)} test queries.")


# --- Search function ---
def search(query, n_results=3, threshold=0.0):
    """Query ChromaDB and keep results with similarity >= threshold."""
    results = collection.query(query_texts=[query], n_results=n_results)

    # ChromaDB returns lists of lists (one inner list per query)
    ids = results["ids"][0]
    distances = results["distances"][0]

    hits = []
    for doc_id, distance in zip(ids, distances):
        similarity = 1 - distance  # convert distance -> similarity
        if similarity >= threshold:
            hits.append((doc_id, similarity))
    return hits


# --- Evaluate function ---
def evaluate(eval_set, n_results=3, threshold=0.0, show_details=False):
    """Run every test query, print per-query scores, return averages."""
    total_precision = 0
    total_recall = 0

    print(f"\n=== Evaluation at threshold={threshold}, n_results={n_results} ===")

    for i, item in enumerate(eval_set, start=1):
        query = item["query"]
        relevant = set(item["relevant"])

        hits = search(query, n_results, threshold)
        returned = set(doc_id for doc_id, _ in hits)

        # Precision: of what was returned, how much was relevant?
        # Recall: of what was relevant, how much was returned?
        true_positives = returned & relevant
        precision = len(true_positives) / len(returned) if returned else 0
        recall = len(true_positives) / len(relevant) if relevant else 0

        total_precision += precision
        total_recall += recall

        print(f"Query {i}: P={precision:.1%} R={recall:.1%}")

        # Optional: show what was expected vs. returned (helpful for analysis)
        if show_details:
            print(f"   {query}")
            print(f"   Expected: {sorted(relevant)}")
            print(f"   Got:      {[(d, round(s, 3)) for d, s in hits]}")

    avg_precision = total_precision / len(eval_set)
    avg_recall = total_recall / len(eval_set)

    print(f"AVERAGE: P={avg_precision:.1%} R={avg_recall:.1%}")

    return avg_precision, avg_recall


# --- Run evaluation at 3 different settings (0.3, 0.5, and 0.2) ---
settings = [
    {"threshold": 0.3, "n_results": 3},  # baseline
    {"threshold": 0.5, "n_results": 3},  # stricter threshold
    {"threshold": 0.2, "n_results": 5},  # looser threshold, more results
]

summary = []
for s in settings:
    p, r = evaluate(eval_set, n_results=s["n_results"], threshold=s["threshold"])
    summary.append((s["threshold"], s["n_results"], p, r))

# --- Side-by-side comparison ---
print("\n=== Comparison ===")
for t, n, p, r in summary:
    print(f"threshold={t}, n_results={n}: P={p:.1%} R={r:.1%}")

# --- Analysis of Results---
print("\n=== ANALYSIS ===")
print(
    "BEST SETTING: threshold=0.3, n_results=3 gave the best balance "
    "(P=85.7%, R=100.0%). It found every relevant document while letting "
    "in only a few extras that were not present in my original answer key."
)

print(
    "\nWORKED WELL: Query 4 (arranging content side by side) scored "
    "P=100% R=100% at 0.3 and 0.2, even though it spans two topics "
    "(Streamlit layout and CSS Flexbox). Query 3 was the only query "
    "that stayed perfect at the strict 0.5 threshold, because its "
    "wording closely matches the session state and widget documents."
)

print(
    "\nFAILED: Queries 1, 2, and 7 returned nothing at threshold=0.5. "
    "Their best matches scored between 0.3 and 0.5 because the queries "
    "use different words than the documents (e.g. 'logged-in users' vs. "
    "'authentication'). At threshold=0.2, n_results=5, precision dropped "
    "for most queries (Query 2 fell to 25%) because unrelated documents "
    "scored above the 0.2 cutoff."
)

print(
    "\nTRADEOFF: Raising the threshold to 0.5 made every returned result "
    "correct, but Queries 1, 2, and 7 returned nothing and scored 0% "
    "precision, which lowered the average. Lowering the threshold to 0.2 "
    "kept recall at 100% but let in more unrelated documents. More results "
    "(n_results=5) mainly added noise since most queries only have 1-2 "
    "relevant documents."
)

print(
    "\nIMPROVEMENTS: (1) Test thresholds between 0.3 and 0.4 to find the "
    "best cutoff. (2) Add more detail and synonyms to short documents so "
    "differently-worded queries match better. (3) Use ChromaDB metadata "
    "filtering by topic to remove off-topic results. (4) Try a larger "
    "embedding model for better semantic matching."
)
