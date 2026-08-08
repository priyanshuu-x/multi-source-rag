"""
Phase 5 tests - run this after implementing/changing reranker.py

Uses a fake reranker function (scores by word overlap between query and text)
so these tests run instantly with no cross-encoder model download.

Usage: python test_reranker.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reranker import rerank


def fake_reranker_fn(pairs: list[tuple]) -> list[float]:
    """Deterministic fake: score = number of shared words between query and doc."""
    scores = []
    for query, text in pairs:
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        scores.append(float(len(query_words & text_words)))
    return scores


def sample_candidates():
    return [
        {"text": "FastAPI is a modern Python web framework", "metadata": {"source": "a"}},
        {"text": "Rerankers use a cross-encoder to score relevance", "metadata": {"source": "b"}},
        {"text": "Hybrid search combines dense and sparse retrieval", "metadata": {"source": "c"}},
    ]


def test_empty_query_raises():
    try:
        rerank("   ", sample_candidates(), reranker_fn=fake_reranker_fn)
        raise AssertionError("FAILED: should raise on empty query")
    except ValueError:
        print("PASS: empty query raises ValueError")


def test_empty_candidates_raises():
    try:
        rerank("hybrid search", [], reranker_fn=fake_reranker_fn)
        raise AssertionError("FAILED: should raise on empty candidates")
    except ValueError:
        print("PASS: empty candidates list raises ValueError")


def test_invalid_top_k_raises():
    try:
        rerank("hybrid search", sample_candidates(), top_k=0, reranker_fn=fake_reranker_fn)
        raise AssertionError("FAILED: should raise on top_k=0")
    except ValueError:
        print("PASS: top_k < 1 raises ValueError")


def test_candidate_missing_text_key_raises():
    bad_candidates = [{"metadata": {"source": "a"}}]  # no "text" key
    try:
        rerank("hybrid search", bad_candidates, reranker_fn=fake_reranker_fn)
        raise AssertionError("FAILED: should raise on missing 'text' key")
    except ValueError:
        print("PASS: candidate missing 'text' key raises a clear ValueError")


def test_rerank_reorders_by_relevance():
    results = rerank("hybrid search retrieval", sample_candidates(), top_k=3, reranker_fn=fake_reranker_fn)
    assert len(results) == 3
    assert results[0]["metadata"]["source"] == "c"  # most word overlap with the query
    for r in results:
        assert "rerank_score" in r
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    print("PASS: rerank reorders candidates by relevance score, highest first")


def test_top_k_larger_than_candidates_is_clamped():
    results = rerank("hybrid search", sample_candidates(), top_k=10, reranker_fn=fake_reranker_fn)
    assert len(results) == 3  # only 3 candidates exist, no crash
    print("PASS: top_k larger than candidate count is clamped, not an error")


if __name__ == "__main__":
    test_empty_query_raises()
    test_empty_candidates_raises()
    test_invalid_top_k_raises()
    test_candidate_missing_text_key_raises()
    test_rerank_reorders_by_relevance()
    test_top_k_larger_than_candidates_is_clamped()
    print("\nAll Phase 5 tests passed (fake reranker - plumbing fully verified).")
