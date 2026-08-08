"""
Runs every test_*.py file in this folder and reports a summary.
Each file already exits non-zero on failure, so this just aggregates that.
Works no matter where you run it from (resolves paths relative to this file).

Usage: python tests/run_all_tests.py   (from the project root)
   or: python run_all_tests.py         (from inside tests/)
"""
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent

TEST_FILES = [
    "test_loaders.py",
    "test_chunking.py",
    "test_embedding.py",
    "test_retrieval.py",
    "test_reranker.py",
    "test_llm.py",
    "test_main.py",
    "test_frontend.py",
    "test_eval.py",
]


def run_all():
    failed = []
    for test_file in TEST_FILES:
        print(f"\n{'=' * 60}\nRunning {test_file}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(TEST_DIR / test_file)])
        if result.returncode != 0:
            failed.append(test_file)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {len(failed)} test file(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(TEST_FILES)} test suites passed.")


if __name__ == "__main__":
    run_all()
