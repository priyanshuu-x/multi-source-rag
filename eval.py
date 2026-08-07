import re
from langsmith import traceable
from llm import get_groq_client, generate_answer, DEFAULT_MODEL
from retrieval import hybrid_search
from reranker import rerank

JUDGE_MODEL = DEFAULT_MODEL  # reuse the same Groq model for judging - no new dependency needed


def _call_judge(prompt: str, client=None) -> str:
    groq_client = get_groq_client(client)
    try:
        response = groq_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise RuntimeError(f"Judge LLM call failed: {e}")
    return response.choices[0].message.content


def _parse_score(text: str) -> float:
    """Extracts a 0-1 score from the judge's response. Handles a judge answering
    '0.8', '8/10', or '80' (out of 100) - not just a bare decimal - since LLMs
    don't always follow formatting instructions exactly."""
    match = re.search(r"(\d*\.?\d+)", text)
    if not match:
        raise ValueError(f"Could not parse a numeric score from judge response: {text!r}")
    score = float(match.group(1))
    if score > 1:
        score = score / 10 if score <= 10 else score / 100
    return max(0.0, min(1.0, score))


@traceable(run_type="tool", name="judge_faithfulness")
def judge_faithfulness(answer: str, context: str, client=None) -> float:
    """Is the answer grounded in the given context (not hallucinated)?"""
    prompt = (
        "You are evaluating whether an AI-generated answer is faithful to the given context "
        "(i.e. it does not state anything unsupported by the context).\n\n"
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Respond with ONLY a number between 0 and 1, where 1 means fully grounded in the "
        "context and 0 means entirely unsupported/hallucinated. No explanation, just the number."
    )
    return _parse_score(_call_judge(prompt, client))


@traceable(run_type="tool", name="judge_answer_relevancy")
def judge_answer_relevancy(question: str, answer: str, client=None) -> float:
    """Does the answer actually address the question asked?"""
    prompt = (
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        "On a scale of 0 to 1, how relevant is this answer to the question "
        "(1 = directly and fully addresses it, 0 = completely irrelevant)? "
        "Respond with ONLY the number."
    )
    return _parse_score(_call_judge(prompt, client))


@traceable(run_type="tool", name="judge_context_precision")
def judge_context_precision(question: str, context: str, client=None) -> float:
    """Of the retrieved context, how much is actually relevant to the question?"""
    prompt = (
        f"Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
        "On a scale of 0 to 1, what proportion of this retrieved context is actually "
        "relevant/useful for answering the question? Respond with ONLY the number."
    )
    return _parse_score(_call_judge(prompt, client))


@traceable(run_type="tool", name="judge_context_recall")
def judge_context_recall(question: str, context: str, ground_truth: str, client=None) -> float:
    """Does the retrieved context contain what's needed to produce the known ground-truth answer?"""
    prompt = (
        f"Question:\n{question}\n\nGround truth answer:\n{ground_truth}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "On a scale of 0 to 1, how much of the information needed to produce the ground truth "
        "answer is present in the retrieved context (1 = all of it, 0 = none of it)? "
        "Respond with ONLY the number."
    )
    return _parse_score(_call_judge(prompt, client))


@traceable(run_type="chain", name="evaluate_question")
def evaluate_question(question: str, ground_truth: str = None, top_k: int = 5, client=None) -> dict:
    """Runs the full pipeline for one question and scores it on all 4 metrics.
    ground_truth is optional - context_recall is skipped without it."""
    candidates = hybrid_search(question, top_k=top_k * 2)
    reranked = rerank(question, candidates, top_k=top_k)
    result = generate_answer(question, reranked)
    context = "\n\n".join(c["text"] for c in reranked)

    scores = {
        "question": question,
        "answer": result["answer"],
        "faithfulness": judge_faithfulness(result["answer"], context, client),
        "answer_relevancy": judge_answer_relevancy(question, result["answer"], client),
        "context_precision": judge_context_precision(question, context, client),
    }
    if ground_truth:
        scores["context_recall"] = judge_context_recall(question, context, ground_truth, client)
    return scores


@traceable(run_type="chain", name="run_evaluation")
def run_evaluation(eval_set: list[dict], top_k: int = 5, client=None) -> dict:
    """eval_set: list of {"question": str, "ground_truth": str (optional)}.
    Returns per-question results plus averaged scores. A single bad question
    (e.g. judge output that fails to parse) is recorded as an error and does NOT
    crash the whole batch - one flaky LLM response shouldn't lose the rest of the run."""
    if not eval_set:
        raise ValueError("eval_set cannot be empty")

    results = []
    for item in eval_set:
        try:
            result = evaluate_question(item["question"], item.get("ground_truth"), top_k, client)
        except Exception as e:
            result = {"question": item["question"], "error": str(e)}
        results.append(result)

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {}
    for metric in metric_names:
        values = [r[metric] for r in results if metric in r]
        if values:
            averages[metric] = sum(values) / len(values)

    return {
        "results": results,
        "averages": averages,
        "errored_questions": sum(1 for r in results if "error" in r),
    }


if __name__ == "__main__":
    import json

    with open("eval_dataset.json") as f:
        eval_set = json.load(f)

    report = run_evaluation(eval_set)

    print("\n=== Evaluation Report ===")
    for metric, avg in report["averages"].items():
        print(f"{metric}: {avg:.2f}")
    if report["errored_questions"]:
        print(f"\n{report['errored_questions']} question(s) errored - see eval_results.json for details")

    with open("eval_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull results saved to eval_results.json")