"""
Phase 9 tests - run this after implementing/changing eval.py

Uses a fake Groq client (returns canned scores) and mocks the pipeline
functions (hybrid_search, rerank, generate_answer), so these tests run
instantly with no real models, API keys, or network access needed.

Usage: python test_eval.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch
from eval import (
    _parse_score, judge_faithfulness, evaluate_question, run_evaluation,
)


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeJudgeClient:
    """Always returns a fixed score - configurable per test."""
    def __init__(self, score_text="0.8"):
        self.chat = self._Chat(score_text)

    class _Chat:
        def __init__(self, score_text):
            self.completions = self
            self.score_text = score_text

        def create(self, model, messages):
            return FakeResponse(self.score_text)


def test_parse_score_plain_decimal():
    assert _parse_score("0.8") == 0.8
    print("PASS: parses a plain decimal score")


def test_parse_score_with_surrounding_text():
    assert _parse_score("The score is 0.75 based on the context.") == 0.75
    print("PASS: parses a score embedded in explanatory text")


def test_parse_score_out_of_ten():
    assert _parse_score("8") == 0.8
    print("PASS: rescales an out-of-10 style score (e.g. judge says '8' meaning 8/10)")


def test_parse_score_out_of_hundred():
    assert _parse_score("80") == 0.8
    print("PASS: rescales an out-of-100 style score")


def test_parse_score_unparseable_raises():
    try:
        _parse_score("I cannot determine a score")
        raise AssertionError("FAILED: should raise ValueError when no number is present")
    except ValueError:
        print("PASS: unparseable judge response raises a clear ValueError")


def test_judge_faithfulness_with_fake_client():
    score = judge_faithfulness("some answer", "some context", client=FakeJudgeClient("0.9"))
    assert score == 0.9
    print("PASS: judge_faithfulness returns the parsed score from the judge LLM")


@patch("eval.generate_answer")
@patch("eval.rerank")
@patch("eval.hybrid_search")
def test_evaluate_question_without_ground_truth_skips_context_recall(mock_hybrid, mock_rerank, mock_generate):
    mock_hybrid.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}}]
    mock_rerank.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}, "rerank_score": 0.9}]
    mock_generate.return_value = {"answer": "some answer", "sources": ["doc1"]}

    result = evaluate_question("what is RAG?", client=FakeJudgeClient("0.8"))
    assert "context_recall" not in result
    assert result["faithfulness"] == 0.8
    assert result["answer_relevancy"] == 0.8
    assert result["context_precision"] == 0.8
    print("PASS: evaluate_question skips context_recall when no ground_truth is given")


@patch("eval.generate_answer")
@patch("eval.rerank")
@patch("eval.hybrid_search")
def test_evaluate_question_with_ground_truth_includes_context_recall(mock_hybrid, mock_rerank, mock_generate):
    mock_hybrid.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}}]
    mock_rerank.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}, "rerank_score": 0.9}]
    mock_generate.return_value = {"answer": "some answer", "sources": ["doc1"]}

    result = evaluate_question("what is RAG?", ground_truth="RAG combines retrieval and generation",
                                client=FakeJudgeClient("0.7"))
    assert "context_recall" in result and result["context_recall"] == 0.7
    print("PASS: evaluate_question includes context_recall when ground_truth is given")


def test_run_evaluation_empty_set_raises():
    try:
        run_evaluation([])
        raise AssertionError("FAILED: should raise ValueError for empty eval set")
    except ValueError:
        print("PASS: run_evaluation raises ValueError for an empty eval set")


@patch("eval.generate_answer")
@patch("eval.rerank")
@patch("eval.hybrid_search")
def test_run_evaluation_computes_averages(mock_hybrid, mock_rerank, mock_generate):
    mock_hybrid.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}}]
    mock_rerank.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}, "rerank_score": 0.9}]
    mock_generate.return_value = {"answer": "some answer", "sources": ["doc1"]}

    eval_set = [{"question": "q1"}, {"question": "q2"}]
    report = run_evaluation(eval_set, client=FakeJudgeClient("0.6"))
    assert report["averages"]["faithfulness"] == 0.6
    assert report["errored_questions"] == 0
    assert len(report["results"]) == 2
    print("PASS: run_evaluation computes averages correctly across multiple questions")


@patch("eval.generate_answer")
@patch("eval.rerank")
@patch("eval.hybrid_search")
def test_run_evaluation_survives_one_bad_question(mock_hybrid, mock_rerank, mock_generate):
    # First question's hybrid_search raises; second question succeeds normally
    mock_hybrid.side_effect = [ValueError("No documents indexed"), [{"text": "chunk", "metadata": {"source": "doc1"}}]]
    mock_rerank.return_value = [{"text": "chunk", "metadata": {"source": "doc1"}, "rerank_score": 0.9}]
    mock_generate.return_value = {"answer": "some answer", "sources": ["doc1"]}

    eval_set = [{"question": "bad question"}, {"question": "good question"}]
    report = run_evaluation(eval_set, client=FakeJudgeClient("0.5"))
    assert report["errored_questions"] == 1
    assert len(report["results"]) == 2  # both questions still appear in results
    assert "error" in report["results"][0]
    assert "faithfulness" in report["results"][1]
    print("PASS: run_evaluation records one question's failure without losing the rest of the batch")


if __name__ == "__main__":
    test_parse_score_plain_decimal()
    test_parse_score_with_surrounding_text()
    test_parse_score_out_of_ten()
    test_parse_score_out_of_hundred()
    test_parse_score_unparseable_raises()
    test_judge_faithfulness_with_fake_client()
    test_evaluate_question_without_ground_truth_skips_context_recall()
    test_evaluate_question_with_ground_truth_includes_context_recall()
    test_run_evaluation_empty_set_raises()
    test_run_evaluation_computes_averages()
    test_run_evaluation_survives_one_bad_question()
    print("\nAll Phase 9 tests passed.")
