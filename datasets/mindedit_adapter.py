from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.common import DataUnavailableError, common_example, limited, read_json_or_jsonl, require_root

TASKS = [
    "camera_motion",
    "cross_view_object_relationship",
    "cross_view_perspective_transformation",
    "cross_view_visibility",
    "single_view_perspective_transformation",
    "single_view_spatial_editing",
]
INSTRUCTIONS = (
    "MindEdit-Bench is public on Hugging Face as `ZODAOfficial/MindEdit-Bench`. "
    "Use a prepared export or pass root='hf' explicitly for small annotation sampling."
)


def _from_hf(limit: int | None) -> list[dict[str, Any]]:
    try:
        import importlib
        import sys

        local_pkg = sys.modules.get("datasets")
        repo_root = str(Path(__file__).resolve().parents[1])
        old_path = list(sys.path)
        sys.modules.pop("datasets", None)
        sys.path = [p for p in sys.path if p not in {"", repo_root}]
        hf_datasets = importlib.import_module("datasets")
    except Exception as exc:
        raise DataUnavailableError("Could not import Hugging Face `datasets`; use a local export.") from exc
    finally:
        sys.path = old_path
        if local_pkg is not None:
            sys.modules["datasets"] = local_pkg
    rows = []
    per_task = max(1, (limit or len(TASKS)) // len(TASKS))
    for task in TASKS:
        ds = hf_datasets.load_dataset("ZODAOfficial/MindEdit-Bench", task, split="test", streaming=True)
        for idx, row in enumerate(ds):
            rows.append(
                common_example(
                    dataset="mindedit_bench",
                    sample_id=str(row.get("id", f"{task}_{idx}")),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    metadata={
                        "task": row.get("task", task),
                        "room": row.get("room"),
                        "choices": row.get("choices"),
                        "views_present": [k for k in ("view1", "view2", "view3") if k in row],
                        "access_status": "public_hf_annotations_streaming",
                    },
                )
            )
            if idx + 1 >= per_task:
                break
            if limit is not None and len(rows) >= limit:
                break
    return rows[:limit] if limit is not None else rows


def _from_local(root: Path, limit: int | None) -> list[dict[str, Any]]:
    files = sorted(root.rglob("*.jsonl")) + sorted(root.rglob("*.json"))
    if not files:
        raise DataUnavailableError(f"No MindEdit JSON/JSONL files found under {root}. {INSTRUCTIONS}")
    rows = []
    for path in files:
        data = read_json_or_jsonl(path)
        if isinstance(data, dict):
            data = data.get("data", data.get("rows", []))
        for row in data:
            rows.append(
                common_example(
                    dataset="mindedit_bench",
                    sample_id=str(row.get("id", len(rows))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    metadata={"task": row.get("task"), "choices": row.get("choices"), "access_status": "prepared_local_root"},
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict[str, Any]]:
    if root == "hf":
        return _from_hf(limit if smoke else limit)
    return _from_local(require_root(root, "MindEdit-Bench", INSTRUCTIONS), limit)
