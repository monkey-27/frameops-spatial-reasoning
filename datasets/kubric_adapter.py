from __future__ import annotations

from pathlib import Path

from datasets.common import DataUnavailableError, common_example, limited, read_json_or_jsonl, require_root

INSTRUCTIONS = (
    "Kubric/MOVi data is available through official Kubric/TFDS/GCS paths but can "
    "be heavy and Blender-based for generation. This adapter consumes a prepared "
    "small JSON/JSONL conversion and never starts Blender or downloads TFDS by default."
)


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict]:
    root_path = require_root(root, "Kubric", INSTRUCTIONS)
    files = sorted(root_path.rglob("*.jsonl")) + sorted(root_path.rglob("*.json"))
    if not files:
        raise DataUnavailableError(f"No Kubric conversion JSON/JSONL found under {root_path}. {INSTRUCTIONS}")
    rows = []
    for path in files:
        data = read_json_or_jsonl(path)
        records = data if isinstance(data, list) else data.get("data", data.get("examples", [])) if isinstance(data, dict) else []
        for record in records:
            rows.append(
                common_example(
                    dataset="kubric",
                    sample_id=str(record.get("id", len(rows))),
                    question=str(record.get("question", "")),
                    answer=str(record.get("answer", "")),
                    image_path=record.get("image_path"),
                    metadata={
                        "access_status": "prepared_local_root",
                        "kubric_metadata": record.get("metadata", {}),
                    },
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return limited(rows, limit if smoke else limit)
