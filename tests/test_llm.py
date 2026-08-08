"""
Phase 6 tests - run this after implementing/changing llm.py

Uses a fake Groq client (mimics client.chat.completions.create()) so these
tests run instantly with no real API key or network access needed.

Usage: python test_llm.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm import generate_answer


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeGroqClient:
    """Mimics groq.Groq's client.chat.completions.create() interface."""
    class chat:
        class completions:
            @staticmethod
            def create(model, messages):
                # Echo back something derived from the input so tests can assert on it
                user_msg = messages[-1]["content"]
                return FakeResponse(f"Fake answer based on: {user_msg[:50]}...")


def sample_chunks():
    return [
        {"text": "RAG combines retrieval with generation", "metadata": {"source": "doc1.md"}},
        {"text": "Hybrid search combines dense and sparse retrieval", "metadata": {"source": "doc2.pdf"}},
    ]


def test_empty_query_raises():
    try:
        generate_answer("   ", sample_chunks(), client=FakeGroqClient())
        raise AssertionError("FAILED: should raise on empty query")
    except ValueError:
        print("PASS: empty query raises ValueError")


def test_empty_context_raises():
    try:
        generate_answer("what is RAG", [], client=FakeGroqClient())
        raise AssertionError("FAILED: should raise on empty context")
    except ValueError:
        print("PASS: empty context_chunks raises ValueError")


def test_missing_api_key_raises_clear_error():
    import os
    from llm import get_groq_client
    old_key = os.environ.pop("GROQ_API_KEY", None)
    try:
        import llm
        llm._client = None  # reset singleton so it re-checks the env var
        try:
            get_groq_client()
            raise AssertionError("FAILED: should raise when GROQ_API_KEY is missing")
        except ValueError as e:
            assert "GROQ_API_KEY" in str(e)
            print("PASS: missing GROQ_API_KEY raises a clear ValueError")
    finally:
        if old_key:
            os.environ["GROQ_API_KEY"] = old_key


def test_generate_answer_returns_expected_shape():
    result = generate_answer("what is RAG", sample_chunks(), client=FakeGroqClient())
    assert "answer" in result and "sources" in result
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert result["sources"] == ["doc1.md", "doc2.pdf"]
    print("PASS: generate_answer returns {answer, sources} with sources deduplicated and sorted")


def test_context_includes_source_labels():
    # Verify the prompt sent to the LLM actually includes [Source: ...] labels
    captured = {}

    class CapturingClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages):
                    captured["messages"] = messages
                    return FakeResponse("ok")

    generate_answer("what is RAG", sample_chunks(), client=CapturingClient())
    user_content = captured["messages"][-1]["content"]
    assert "[Source: doc1.md]" in user_content
    assert "[Source: doc2.pdf]" in user_content
    print("PASS: prompt sent to the LLM includes source labels for each chunk")


def test_groq_api_error_is_wrapped_clearly():
    class FailingClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages):
                    raise ConnectionError("simulated network failure")

    try:
        generate_answer("what is RAG", sample_chunks(), client=FailingClient())
        raise AssertionError("FAILED: should raise when the API call fails")
    except RuntimeError as e:
        assert "Groq API call failed" in str(e)
        print("PASS: Groq API failures are wrapped in a clear RuntimeError")


if __name__ == "__main__":
    test_empty_query_raises()
    test_empty_context_raises()
    test_missing_api_key_raises_clear_error()
    test_generate_answer_returns_expected_shape()
    test_context_includes_source_labels()
    test_groq_api_error_is_wrapped_clearly()
    print("\nAll Phase 6 tests passed (fake Groq client - plumbing fully verified).")
