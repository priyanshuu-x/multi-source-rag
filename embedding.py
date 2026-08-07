import re
import chromadb
from chromadb.utils import embedding_functions
from langsmith import traceable

_client = None            # singleton in-memory ChromaDB client - lives for the process lifetime
_embedding_function = None  # singleton embedding model - loading it is slow, so load once


def get_client():
    """Returns the single shared in-memory ChromaDB client for this process.
    Ephemeral by design (Phase 3 decision) - data resets when the process restarts."""
    global _client
    if _client is None:
        _client = chromadb.Client()
    return _client


def get_embedding_function(embedding_fn=None):
    """Returns the embedding function to use. Pass embedding_fn to override
    (used by tests to avoid downloading the real model)."""
    global _embedding_function
    if embedding_fn is not None:
        return embedding_fn
    if _embedding_function is None:
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedding_function


def get_collection(collection_name: str = "rag_chunks", embedding_fn=None):
    client = get_client()
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(embedding_fn),
    )


def _sanitize_id(text: str) -> str:
    """Chroma IDs just need to be unique strings, but keeping them clean
    (no slashes/spaces/etc. from URLs or file paths) avoids surprises."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


@traceable(run_type="tool", name="add_chunks_to_store")
def add_chunks_to_store(chunks: list[dict], collection_name: str = "rag_chunks", embedding_fn=None) -> int:
    """Embeds and stores chunks (from chunk_document()) into ChromaDB.
    Uses upsert, not add: re-ingesting the same source overwrites its old
    chunks instead of duplicating them."""
    if not chunks:
        raise ValueError("No chunks to add")

    collection = get_collection(collection_name, embedding_fn)

    ids = [f"{_sanitize_id(c['source'])}_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"source": c["source"], "source_type": c["source_type"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


@traceable(run_type="tool", name="dense_search")
def dense_search(query: str, top_k: int = 5, collection_name: str = "rag_chunks", embedding_fn=None) -> list[dict]:
    """Semantic search over stored chunks. Returns [{text, metadata, distance}, ...]
    sorted by relevance (lowest distance first)."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    collection = get_collection(collection_name, embedding_fn)
    count = collection.count()
    if count == 0:
        raise ValueError(f"No documents in '{collection_name}' yet - add chunks before searching")

    results = collection.query(query_texts=[query], n_results=min(top_k, count))

    return [
        {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def clear_collection(collection_name: str = "rag_chunks"):
    """Deletes a collection entirely. Useful for tests and for manually resetting the store."""
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection didn't exist - nothing to clear