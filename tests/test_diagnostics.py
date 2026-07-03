from __future__ import annotations

from pathlib import Path

from datasets.synthetic import DIAGNOSTIC_TASKS, generate_diagnostic_dataset
from frameops.diagnostics import paired_delta_ci, summarize_prediction_rows, validate_diagnostic_split


def test_diagnostic_split_balanced_validator(tmp_path: Path) -> None:
    rows = generate_diagnostic_dataset(
        tmp_path / "diag",
        seed=123,
        image_size=64,
        train_per_task=2,
        val_per_task=1,
        test_per_task=2,
        enforce_min_test=False,
    )
    summary = validate_diagnostic_split(rows["test"], min_per_task=2)
    assert summary["n"] == len(DIAGNOSTIC_TASKS) * 2
    assert all(count == 2 for count in summary["task_counts"].values())
    assert (tmp_path / "diag" / "diagnostic_manifest.json").exists()


def test_summary_and_paired_delta() -> None:
    base = [
        {"id": "a", "correct": True, "task_family": "x", "hardness": "easy", "noise": 0.0},
        {"id": "b", "correct": False, "task_family": "x", "hardness": "hard", "noise": 0.0},
    ]
    better = [
        {"id": "a", "correct": True, "task_family": "x", "hardness": "easy", "noise": 0.0},
        {"id": "b", "correct": True, "task_family": "x", "hardness": "hard", "noise": 0.0},
    ]
    summary = summarize_prediction_rows(better, baseline_rows=base)
    assert summary["overall"]["mean"] == 1.0
    assert paired_delta_ci(better, base)["mean"] == 0.5
