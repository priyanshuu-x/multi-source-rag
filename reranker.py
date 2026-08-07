from sentence_transformers import CrossEncoder
from langsmith import traceable

_reranker_model = None
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_reranker(reranker_fn=None):
    """Returns a callable that scores (query, doc) pairs -> list[float].
    Pass reranker_fn to override (used by tests to avoid downloading the real model)."""
    global _reranker_model
    if reranker_fn is not None:
        return reranker_fn
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker_model.predict


@traceable(run_type="tool", name="rerank")
def rerank(query: str, candidates: list[dict], top_k: int = 5, reranker_fn=None) -> list[dict]:
    """Re-scores each candidate chunk against the query with a cross-encoder
    (more accurate but slower than embedding similarity - fine here since it
    only runs on the small candidate set from hybrid_search, not the whole corpus).
    Returns the top_k candidates re-sorted by that score, highest first.

    candidates: list of dicts with at least a "text" key (e.g. hybrid_search's output)."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not candidates:
        raise ValueError("No candidates to rerank")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    for c in candidates:
        if "text" not in c:
            raise ValueError(f"Candidate missing required 'text' key: {c}")

    predict_fn = get_reranker(reranker_fn)
    pairs = [(query, c["text"]) for c in candidates]
    scores = predict_fn(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in scored[:top_k]
    ]