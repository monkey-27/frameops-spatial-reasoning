from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def normalize_answer(answer: Any) -> str:
    return str(answer).strip().lower()


def exact_match(prediction: Any, target: Any) -> float:
    return float(normalize_answer(prediction) == normalize_answer(target))


def accuracy(predictions: Iterable[Any], targets: Iterable[Any]) -> float:
    preds = list(predictions)
    gold = list(targets)
    if len(preds) != len(gold):
        raise ValueError("predictions and targets must have the same length")
    if not preds:
        return 0.0
    return sum(exact_match(p, t) for p, t in zip(preds, gold, strict=True)) / len(preds)


def accuracy_by_field(rows: Iterable[dict[str, Any]], field: str = "task_type") -> dict[str, float]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, float] = defaultdict(float)
    for row in rows:
        key = str(row.get(field, "unknown"))
        totals[key] += 1
        correct[key] += exact_match(row.get("prediction"), row.get("answer"))
    return {key: correct[key] / totals[key] for key in sorted(totals)}


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall_accuracy": accuracy([r.get("prediction") for r in rows], [r.get("answer") for r in rows]),
        "accuracy_by_task_type": accuracy_by_field(rows, "task_type"),
        "accuracy_by_difficulty": accuracy_by_field(rows, "difficulty"),
        "n": len(rows),
    }
