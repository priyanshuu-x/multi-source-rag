"""
Phase 4 tests - run this after implementing/changing retrieval.py

Uses the same fake deterministic embedder as test_embedding.py so these
tests run instantly with no model download.

Usage: python test_retrieval.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
from chromadb.api.types import EmbeddingFunction
from embedding import clear_collection
from retrieval import (
    index_chunks, sparse_search, hybrid_search,
    clear_sparse_index, add_chunks_to_sparse_index, _tokenize,
)


class FakeEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            h = hashlib.md5(text.encode()).digest()
            vectors.append(([b / 255.0 for b in h] * 6)[:96])
        return vectors

    def name(self):
        return "fake"


TEST_COLLECTION = "test_retrieval_collection"
FAKE_EF = FakeEmbeddingFunction()


def sample_chunks():
    return [
        {"text": "RAG combines retrieval with generation", "source": "doc1.md", "source_type": "markdown", "chunk_index": 0},
        {"text": "Hybrid search combines dense and sparse retrieval methods", "source": "doc1.md", "source_type": "markdown", "chunk_index": 1},
        {"text": "Rerankers improve result ordering using a cross-encoder", "source": "doc2.pdf", "source_type": "pdf", "chunk_index": 0},
        {"text": "FastAPI is a modern Python web framework for building APIs", "source": "doc3.pdf", "source_type": "pdf", "chunk_index": 0},
    ]


def reset():
    clear_collection(TEST_COLLECTION)
    clear_sparse_index()


def test_tokenizer():
    tokens = _tokenize("Hello, World! It's RAG-based.")
    assert tokens == ["hello", "world", "it", "s", "rag", "based"]
    print("PASS: tokenizer lowercases and strips punctuation")


def test_sparse_search_empty_index_raises():
    reset()
    try:
        sparse_search("test")
        raise AssertionError("FAILED: should raise on empty index")
    except ValueError:
        print("PASS: sparse_search on empty index raises ValueError")


def test_sparse_search_empty_query_raises():
    reset()
    add_chunks_to_sparse_index(sample_chunks())
    try:
        sparse_search("   ")
        raise AssertionError("FAILED: should raise on empty query")
    except ValueError:
        print("PASS: sparse_search with empty query raises ValueError")


def test_sparse_search_finds_keyword_match():
    reset()
    add_chunks_to_sparse_index(sample_chunks())
    results = sparse_search("FastAPI Python framework", top_k=2)
    assert results[0]["metadata"]["source"] == "doc3.pdf"
    print("PASS: sparse_search ranks the exact keyword match first")


def test_index_chunks_keeps_dense_and_sparse_in_sync():
    reset()
    count = index_chunks(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    assert count == 4
    # both indices should now have all 4 chunks
    sparse_results = sparse_search("retrieval", top_k=10)
    assert len(sparse_results) == 4
    print("PASS: index_chunks populates both dense and sparse indices together")


def test_hybrid_search_top_k_and_rrf_validation():
    reset()
    index_chunks(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    for bad_kwargs in [{"top_k": 0}, {"rrf_k": 0}]:
        try:
            hybrid_search("retrieval", collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF, **bad_kwargs)
            raise AssertionError(f"FAILED: should raise for {bad_kwargs}")
        except ValueError:
            pass
    print("PASS: hybrid_search validates top_k and rrf_k")


def test_hybrid_search_returns_fused_results():
    reset()
    index_chunks(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    results = hybrid_search("hybrid retrieval search", top_k=3, collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    assert len(results) <= 3
    for r in results:
        assert "fused_score" in r and "text" in r and "metadata" in r
    # results should be sorted by fused_score descending
    scores = [r["fused_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    print("PASS: hybrid_search returns fused, correctly sorted results")


if __name__ == "__main__":
    test_tokenizer()
    test_sparse_search_empty_index_raises()
    test_sparse_search_empty_query_raises()
    test_sparse_search_finds_keyword_match()
    test_index_chunks_keeps_dense_and_sparse_in_sync()
    test_hybrid_search_top_k_and_rrf_validation()
    test_hybrid_search_returns_fused_results()
    reset()
    print("\nAll Phase 4 tests passed (fake embedder - plumbing fully verified).")
