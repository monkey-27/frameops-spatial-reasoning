from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets.common import DataUnavailableError, common_example, limited, read_json_or_jsonl, require_root

INSTRUCTIONS = (
    "VSI-Bench-Debiased is a config of `nyu-visionx/VSI-Bench` on Hugging Face. "
    "Do not download video zips by default. Use a prepared JSON/JSONL export or pass root='hf' "
    "for explicit streaming annotation access."
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
        raise DataUnavailableError(
            "Could not import Hugging Face `datasets` because this repo also has a local package named `datasets`. "
            "Use a prepared JSON/JSONL export if the import shim fails."
        ) from exc
    finally:
        sys.path = old_path
        if local_pkg is not None:
            sys.modules["datasets"] = local_pkg
    ds = hf_datasets.load_dataset("nyu-visionx/VSI-Bench", "debiased", split="test", streaming=True)
    rows = []
    for row in ds:
        rows.append(
            common_example(
                dataset="vsi_bench_debiased",
                sample_id=str(row.get("id", len(rows))),
                question=str(row.get("question", "")),
                answer=str(row.get("ground_truth", row.get("answer", ""))),
                metadata={
                    "scene_name": row.get("scene_name"),
                    "question_type": row.get("question_type"),
                    "options": row.get("options"),
                    "dataset": row.get("dataset"),
                    "access_status": "public_hf_annotations_streaming",
                    "media_status": "video_optional_not_downloaded",
                },
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _from_local(root: Path, limit: int | None) -> list[dict[str, Any]]:
    candidates = sorted(root.rglob("*debiased*.jsonl")) + sorted(root.rglob("*debiased*.json"))
    if not candidates:
        raise DataUnavailableError(f"No VSI debiased JSON/JSONL export found under {root}. {INSTRUCTIONS}")
    data = read_json_or_jsonl(candidates[0])
    if isinstance(data, dict):
        data = data.get("data", data.get("rows", []))
    rows = []
    for row in data:
        if isinstance(row, str):
            row = json.loads(row)
        rows.append(
            common_example(
                dataset="vsi_bench_debiased",
                sample_id=str(row.get("id", len(rows))),
                question=str(row.get("question", "")),
                answer=str(row.get("ground_truth", row.get("answer", ""))),
                metadata={
                    "scene_name": row.get("scene_name"),
                    "question_type": row.get("question_type"),
                    "options": row.get("options"),
                    "access_status": "prepared_local_root",
                },
            )
        )
    return limited(rows, limit)


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict[str, Any]]:
    if root == "hf":
        return _from_hf(limit if smoke else limit)
    return _from_local(require_root(root, "VSI-Bench-Debiased", INSTRUCTIONS), limit)
