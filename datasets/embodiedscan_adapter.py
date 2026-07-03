from __future__ import annotations

from pathlib import Path

from datasets.common import DataUnavailableError, common_example, limited, read_json_or_jsonl, require_root

INSTRUCTIONS = (
    "EmbodiedScan requires a manual access form plus prepared ScanNet/3RScan/"
    "Matterport3D/ARKitScenes roots. This adapter never downloads it. Expected "
    "annotation files include `embodiedscan_*_vg*.json` or mini variants."
)


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict]:
    root_path = require_root(root, "EmbodiedScan", INSTRUCTIONS)
    files = sorted(root_path.rglob("embodiedscan_*mini*vg*.json")) + sorted(root_path.rglob("embodiedscan_*_vg*.json"))
    if not files:
        raise DataUnavailableError(f"No EmbodiedScan VG annotation JSON found under {root_path}. {INSTRUCTIONS}")
    data = read_json_or_jsonl(files[0])
    if isinstance(data, dict):
        data = data.get("data", data.get("annotations", []))
    rows = []
    for item in data:
        question = item.get("question", item.get("utterance", ""))
        answer = item.get("answer", item.get("answers", [""])[0] if isinstance(item.get("answers"), list) else "")
        rows.append(
            common_example(
                dataset="embodiedscan",
                sample_id=str(item.get("id", item.get("sample_id", len(rows)))),
                question=str(question),
                answer=str(answer),
                metadata={
                    "scene_id": item.get("scene_id", item.get("scan_id")),
                    "access_status": "prepared_local_root",
                    "media_status": "raw_rgbd_occupancy_optional_not_loaded",
                },
            )
        )
    return limited(rows, limit if smoke else limit)
