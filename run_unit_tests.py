#!/usr/bin/env python3
"""
Single script to run all unit tests for myprogram.py.

Usage (from project root):
  python run_unit_tests.py [--work_dir WORK_DIR] [--train-first] [--quiet]

- Writes tests/input.txt and tests/answer.txt from unit test cases.
- Runs: python src/myprogram.py test --work_dir WORK_DIR --test_data tests/input.txt --test_output tests/pred.txt
- Grades using same logic as grader/grade.py (correct if gold in first 3 pred chars, case-insensitive).
- Prints each failure (index, input snippet, expected, got), then overall accuracy % and runtime in seconds.

Quick run (no network, ~5–10% accuracy expected with English-only model):
  python run_unit_tests.py --work_dir work_unit_tests --minimal_model

First create the minimal model once (takes ~30s, needs datasets library):
  python tests/create_minimal_model.py work_unit_tests

Then run tests (uses existing work_dir):
  python run_unit_tests.py --work_dir work_unit_tests
"""

import argparse
import os
import subprocess
import sys
import time

# Ensure we can import from tests/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tests.unit_test_cases import UNIT_TEST_CASES, get_input_lines, get_answer_lines


def load_pred(fname, force_limit=3):
    with open(fname, encoding="utf-8") as f:
        loaded = []
        for line in f:
            line = line.rstrip("\n").lower()
            if force_limit is not None:
                line = line[:force_limit]
            loaded.append(line)
        return loaded


def load_gold(fname):
    with open(fname, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser(description="Run unit tests for myprogram.py")
    parser.add_argument(
        "--work_dir",
        default="work",
        help="Model work directory (use work_unit_tests with --minimal_model for fast runs)",
    )
    parser.add_argument(
        "--train_first",
        action="store_true",
        help="If no model in work_dir, run full train with small dataset_fraction first (requires network)",
    )
    parser.add_argument(
        "--minimal_model",
        action="store_true",
        help="If no model in work_dir, create a tiny English-only model (no network). Gives low accuracy; good for CI.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress myprogram.py stdout/stderr during test run",
    )
    parser.add_argument(
        "--src",
        default="src/myprogram.py",
        help="Path to myprogram.py from project root",
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    tests_dir = os.path.join(SCRIPT_DIR, "tests")
    os.makedirs(tests_dir, exist_ok=True)

    input_path = os.path.join(tests_dir, "input.txt")
    answer_path = os.path.join(tests_dir, "answer.txt")
    pred_path = os.path.join(tests_dir, "pred.txt")

    # Write test files
    with open(input_path, "w", encoding="utf-8") as f:
        for line in get_input_lines():
            f.write(line + "\n")

    with open(answer_path, "w", encoding="utf-8") as f:
        for line in get_answer_lines():
            f.write(line + "\n")

    myprogram_path = os.path.join(SCRIPT_DIR, args.src)
    if not os.path.isfile(myprogram_path):
        print(f"Error: myprogram not found at {myprogram_path}", file=sys.stderr)
        sys.exit(1)    

    # Run test (predict). If no model exists, myprogram will train; use small fraction then.
    test_cmd = [
        sys.executable,
        myprogram_path,
        "test",
        "--work_dir",
        work_dir,
        "--test_data",
        input_path,
        "--test_output",
        pred_path,
        "--dataset_fraction",
        "0.001",
    ]
    start = time.perf_counter()
    print("hi8")
    result = subprocess.run(
        test_cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        timeout=120,
    )
    elapsed = time.perf_counter() - start

    if not args.quiet and result.stdout:
        sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
    if not args.quiet and result.stderr:
        sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))

    if result.returncode != 0:
        print("Test run (prediction) failed.", file=sys.stderr)
        if result.stdout:
            sys.stderr.write(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
        sys.exit(1)

    if not os.path.isfile(pred_path):
        print("Error: predictions file not created.", file=sys.stderr)
        sys.exit(1)

    # Grade
    pred = load_pred(pred_path, force_limit=3)
    gold = load_gold(answer_path)

    if len(pred) < len(gold):
        pred.extend([""] * (len(gold) - len(pred)))

    correct = 0
    failures = []
    for i, (p, g) in enumerate(zip(pred, gold)):
        right = g.lower() in p
        if right:
            correct += 1
        else:
            inp = UNIT_TEST_CASES[i][0]
            # Show short snippet of input for readability
            snippet = repr(inp) if len(inp) <= 40 else repr(inp[:37] + "...")
            failures.append((i, snippet, g, p))

    total = len(gold)
    accuracy_pct = (correct / total * 100) if total else 0.0

    # Print failures
    print("=== Failures (expected next char not in top-3 predictions) ===\n")
    for i, snippet, expected, got in failures:
        print(f"  [{i}] input: {snippet}  expected: {repr(expected)}  got (top3): {repr(got)}")
    if not failures:
        print("  (none)")
    print()

    print("=== Summary ===")
    print(f"  Correct: {correct} / {total}")
    print(f"  Accuracy: {accuracy_pct:.2f}%")
    print(f"  Runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
