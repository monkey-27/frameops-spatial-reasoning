from __future__ import annotations

from frameops.metrics import accuracy, summarize_predictions


def test_accuracy() -> None:
    assert accuracy(["yes", "no", "yes"], ["yes", "yes", "yes"]) == 2 / 3


def test_summary_by_task_and_difficulty() -> None:
    rows = [
        {"prediction": "yes", "answer": "yes", "task_type": "left", "difficulty": "standard"},
        {"prediction": "no", "answer": "yes", "task_type": "left", "difficulty": "hard"},
        {"prediction": "no", "answer": "no", "task_type": "depth", "difficulty": "hard"},
    ]
    summary = summarize_predictions(rows)
    assert summary["overall_accuracy"] == 2 / 3
    assert summary["accuracy_by_task_type"]["left"] == 0.5
    assert summary["accuracy_by_difficulty"]["hard"] == 0.5
