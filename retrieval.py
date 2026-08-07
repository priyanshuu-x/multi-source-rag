import re
from rank_bm25 import BM25Okapi
from langsmith import traceable
from embedding import add_chunks_to_store, dense_search

_bm25_chunks: list[dict] = []   # mirrors chunks added - needed because BM25Okapi has no add/remove API
_bm25_index = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


@traceable(run_type="tool", name="add_chunks_to_sparse_index")
def add_chunks_to_sparse_index(chunks: list[dict]):
    """Adds chunks to the BM25 corpus and rebuilds the index.
    Known limitation: rank_bm25 has no incremental-update API, so this rebuilds
    the whole index on every call - fine for a small/medium beginner-scale corpus,
    but would need a different BM25 library (or periodic rebuilding) to scale to
    a large, frequently-updated corpus in real production use."""
    global _bm25_index
    if not chunks:
        raise ValueError("No chunks to add")

    _bm25_chunks.extend(chunks)
    tokenized_corpus = [_tokenize(c["text"]) for c in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized_corpus)


@traceable(run_type="tool", name="sparse_search")
def sparse_search(query: str, top_k: int = 5) -> list[dict]:
    """Keyword search via BM25. Returns [{text, metadata, score}, ...] sorted by relevance."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if _bm25_index is None or not _bm25_chunks:
        raise ValueError("No documents in the sparse index yet - add chunks before searching")

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    ranked = sorted(zip(_bm25_chunks, scores), key=lambda pair: pair[1], reverse=True)
    top = ranked[:top_k]

    return [
        {
            "text": c["text"],
            "metadata": {"source": c["source"], "source_type": c["source_type"], "chunk_index": c["chunk_index"]},
            "score": float(score),
        }
        for c, score in top
    ]


def clear_sparse_index():
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = []


@traceable(run_type="chain", name="index_chunks")
def index_chunks(chunks: list[dict], collection_name: str = "rag_chunks", embedding_fn=None) -> int:
    """Single entry point for indexing: adds chunks to BOTH the dense store (Chroma)
    and the sparse index (BM25) together. Use this instead of calling
    add_chunks_to_store / add_chunks_to_sparse_index separately - keeping them behind
    one function prevents the two indices from silently drifting out of sync with
    each other, which hybrid_search below depends on."""
    dense_count = add_chunks_to_store(chunks, collection_name=collection_name, embedding_fn=embedding_fn)
    add_chunks_to_sparse_index(chunks)
    return dense_count


@traceable(run_type="chain", name="hybrid_search")
def hybrid_search(query: str, top_k: int = 5, rrf_k: int = 60,
                   collection_name: str = "rag_chunks", embedding_fn=None) -> list[dict]:
    """Combines dense (semantic) and sparse (keyword/BM25) search using Reciprocal
    Rank Fusion: a chunk's fused score = sum of 1/(rrf_k + rank) across whichever
    list(s) it appears in. Chunks found by both methods naturally rank higher than
    chunks found by only one - that's the point of doing hybrid search at all."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    fetch_k = top_k * 2   # over-fetch from each method before fusing, for better recall
    dense_results = dense_search(query, top_k=fetch_k, collection_name=collection_name, embedding_fn=embedding_fn)
    sparse_results = sparse_search(query, top_k=fetch_k)

    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, result in enumerate(dense_results):
        key = f"{result['metadata']['source']}_{result['metadata']['chunk_index']}"
        scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank + 1)
        chunk_lookup[key] = result

    for rank, result in enumerate(sparse_results):
        key = f"{result['metadata']['source']}_{result['metadata']['chunk_index']}"
        scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank + 1)
        chunk_lookup[key] = result

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [
        {**chunk_lookup[key], "fused_score": scores[key]}
        for key in ranked_keys[:top_k]
    ]