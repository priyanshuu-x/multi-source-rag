from langsmith import traceable


@traceable(run_type="tool", name="chunk_text")
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-windows.

    chunk_size: words per chunk
    overlap: words shared between consecutive chunks (must be < chunk_size)
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("chunk_size must be positive and overlap must be non-negative")

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


@traceable(run_type="chain", name="chunk_document")
def chunk_document(text: str, source: str, source_type: str,
                    chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    """Chunk a document's text and attach the metadata every chunk needs downstream
    (which source it came from, what type, and its position) - so embedding/storage
    doesn't have to reconstruct this later."""
    chunks = chunk_text(text, chunk_size, overlap)
    return [
        {
            "text": chunk,
            "source": source,
            "source_type": source_type,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]