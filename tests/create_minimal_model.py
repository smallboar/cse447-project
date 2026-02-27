#!/usr/bin/env python3
"""Create a minimal n-gram model and save to work_dir. Called as subprocess by run_unit_tests.py."""
import os
import sys

# Run from project root; src must be on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from myprogram import Predictor, MyModel

def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else "work_unit_tests"
    os.makedirs(work_dir, exist_ok=True)
    engine = Predictor(max_context_length=6)
    sample = (
        "the quick brown fox jumps over the lazy dog. "
        "the cat sat on the mat. hello world. "
        "this is a test. one two three four five. "
    )
    engine.train(sample * 50)
    model = MyModel(engine=engine)
    model.save(work_dir)
    print("Minimal model saved to", work_dir, file=sys.stderr)

if __name__ == "__main__":
    main()
