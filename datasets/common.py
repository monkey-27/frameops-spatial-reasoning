from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class DataUnavailableError(RuntimeError):
    pass


def require_root(root: str | Path | None, dataset_name: str, instructions: str) -> Path:
    if root is None:
        raise DataUnavailableError(
            f"{dataset_name} requires a prepared local root. {instructions}"
        )
    path = Path(root)
    if not path.exists():
        raise DataUnavailableError(f"{dataset_name} root does not exist: {path}. {instructions}")
    return path


def limited(rows: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def read_json_or_jsonl(path: Path) -> Any:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def common_example(
    *,
    dataset: str,
    sample_id: str,
    question: str,
    answer: str,
    metadata: dict[str, Any],
    image_path: str | None = None,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "image_path": image_path,
        "question": question,
        "answer": answer,
        "program": [],
        "objects": [],
        "frames": [],
        "camera": {},
        "metadata": {"source_dataset": dataset, **metadata},
    }


def unavailable_stub(dataset: str, reason: str, instructions: str) -> list[dict[str, Any]]:
    raise DataUnavailableError(f"{dataset} unavailable: {reason}. {instructions}")
