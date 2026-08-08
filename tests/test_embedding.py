"""
Phase 3 tests - run this after implementing/changing embedding.py

Note: uses a fake, deterministic embedding function so these tests run
instantly and don't require downloading the real sentence-transformers
model. Run test_embedding_real_model() separately (needs internet) to
confirm the real model works too.

Usage: python test_embedding.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hashlib
from chromadb.api.types import EmbeddingFunction
from embedding import add_chunks_to_store, dense_search, clear_collection, get_client


class FakeEmbeddingFunction(EmbeddingFunction):
    """Deterministic fake embedder - same text always -> same vector, no download needed."""
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


TEST_COLLECTION = "test_collection"
FAKE_EF = FakeEmbeddingFunction()


def sample_chunks():
    return [
        {"text": "RAG combines retrieval with generation", "source": "doc1.md", "source_type": "markdown", "chunk_index": 0},
        {"text": "Hybrid search combines dense and sparse retrieval", "source": "doc1.md", "source_type": "markdown", "chunk_index": 1},
        {"text": "Rerankers improve result ordering after retrieval", "source": "doc2.pdf", "source_type": "pdf", "chunk_index": 0},
    ]


def test_add_empty_chunks_raises():
    clear_collection(TEST_COLLECTION)
    try:
        add_chunks_to_store([], collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
        raise AssertionError("FAILED: should have raised ValueError for empty chunks")
    except ValueError:
        print("PASS: adding empty chunk list raises ValueError")


def test_search_before_adding_raises():
    clear_collection(TEST_COLLECTION)
    try:
        dense_search("test query", collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
        raise AssertionError("FAILED: should have raised ValueError for empty collection")
    except ValueError:
        print("PASS: searching an empty collection raises a clear ValueError")


def test_empty_query_raises():
    try:
        dense_search("   ", collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
        raise AssertionError("FAILED: should have raised ValueError for empty query")
    except ValueError:
        print("PASS: empty query raises ValueError")


def test_add_and_search():
    clear_collection(TEST_COLLECTION)
    count = add_chunks_to_store(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    assert count == 3

    results = dense_search("hybrid search dense sparse", top_k=2, collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    assert len(results) == 2
    for r in results:
        assert "text" in r and "metadata" in r and "distance" in r
        assert r["metadata"]["source"] in ("doc1.md", "doc2.pdf")
    print("PASS: add + search round-trip returns expected shape")


def test_upsert_overwrites_not_duplicates():
    clear_collection(TEST_COLLECTION)
    add_chunks_to_store(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    add_chunks_to_store(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)  # re-ingest same source

    client = get_client()
    collection = client.get_or_create_collection(name=TEST_COLLECTION, embedding_function=FAKE_EF)
    assert collection.count() == 3, f"FAILED: expected 3 (upsert), got {collection.count()} (duplicated)"
    print("PASS: re-ingesting the same source upserts instead of duplicating")


def test_top_k_larger_than_store_is_clamped():
    clear_collection(TEST_COLLECTION)
    add_chunks_to_store(sample_chunks(), collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    results = dense_search("retrieval", top_k=50, collection_name=TEST_COLLECTION, embedding_fn=FAKE_EF)
    assert len(results) == 3  # clamped to what's actually in the store, no crash
    print("PASS: top_k larger than store size is clamped, not an error")


if __name__ == "__main__":
    test_add_empty_chunks_raises()
    test_search_before_adding_raises()
    test_empty_query_raises()
    test_add_and_search()
    test_upsert_overwrites_not_duplicates()
    test_top_k_larger_than_store_is_clamped()
    clear_collection(TEST_COLLECTION)
    print("\nAll Phase 3 tests passed (using fake embedder - plumbing verified).")
    print("Real model (all-MiniLM-L6-v2) not tested here - see note in this file.")
