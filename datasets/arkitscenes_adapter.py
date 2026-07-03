from __future__ import annotations

from pathlib import Path

from datasets.common import DataUnavailableError, common_example, limited, read_json_or_jsonl, require_root

INSTRUCTIONS = (
    "ARKitScenes is very large and governed by Apple's license. Use Apple's "
    "official downloader for a tiny selected video/asset subset, then pass that "
    "prepared root here. This adapter does not auto-download assets."
)


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict]:
    root_path = require_root(root, "ARKitScenes", INSTRUCTIONS)
    annotation_files = sorted(root_path.rglob("*.json"))
    if not annotation_files:
        raise DataUnavailableError(f"No ARKitScenes JSON metadata found under {root_path}. {INSTRUCTIONS}")
    rows = []
    for path in annotation_files:
        try:
            data = read_json_or_jsonl(path)
        except Exception:
            continue
        records = data if isinstance(data, list) else data.get("data", data.get("annotations", [])) if isinstance(data, dict) else []
        if not isinstance(records, list):
            continue
        for record in records:
            rows.append(
                common_example(
                    dataset="arkitscenes",
                    sample_id=str(record.get("id", record.get("video_id", len(rows)))),
                    question=str(record.get("question", "ARKitScenes scene metadata placeholder")),
                    answer=str(record.get("answer", "")),
                    metadata={
                        "video_id": record.get("video_id"),
                        "access_status": "prepared_local_root",
                        "adapter_note": "ARKitScenes has geometry assets but no native QA labels; use converted subsets only.",
                    },
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows
    if not rows:
        raise DataUnavailableError(f"ARKitScenes root contained JSON but no converted QA records. {INSTRUCTIONS}")
    return limited(rows, limit if smoke else limit)
