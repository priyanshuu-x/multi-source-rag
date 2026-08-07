import os
from groq import Groq
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

_client = None
DEFAULT_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context. "
    "If the answer isn't contained in the context, say you don't know - never make up information. "
    "When you use information from the context, mention which source it came from using the "
    "[Source: name] label shown with that piece of context."
)


def get_groq_client(client=None):
    """Returns the shared Groq client. Pass client to override (used by tests
    to avoid needing a real API key / network access).
    Note: the Groq SDK retries connection errors, timeouts, 429s, and 5xxs
    automatically (max_retries=2 by default) - no extra retry logic needed here."""
    global _client
    if client is not None:
        return client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key."
            )
        _client = Groq(api_key=api_key)
    return _client


def _build_context(chunks: list[dict]) -> str:
    """Formats reranked chunks into a labeled context block the LLM can cite from."""
    parts = []
    for c in chunks:
        source = c.get("metadata", {}).get("source", "unknown")
        parts.append(f"[Source: {source}]\n{c['text']}")
    return "\n\n".join(parts)


@traceable(run_type="llm", name="generate_answer")
def generate_answer(query: str, context_chunks: list[dict], model: str = DEFAULT_MODEL, client=None) -> dict:
    """Generates an answer grounded in context_chunks (reranker.rerank()'s output).
    Returns {"answer": str, "sources": [source names used as context]}."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if not context_chunks:
        raise ValueError("No context chunks provided - retrieve and rerank chunks before calling the LLM")

    context = _build_context(context_chunks)
    groq_client = get_groq_client(client)

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {e}")

    sources = sorted({c.get("metadata", {}).get("source", "unknown") for c in context_chunks})
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
    }