from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.common import common_example, limited, read_json_or_jsonl, require_root

INSTRUCTIONS = (
    "Download SQA3D annotations from the official project/Zenodo and place "
    "`v1_balanced_questions_*_scannetv2.json`, "
    "`v1_balanced_sqa_annotations_*_scannetv2.json`, and `answer_dict.json` "
    "under the provided root. ScanNet scene assets remain separate/manual."
)


def _find(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[0]


def _records(root: Path) -> list[dict[str, Any]]:
    q_path = _find(root, "v1_balanced_questions_*_scannetv2.json")
    a_path = _find(root, "v1_balanced_sqa_annotations_*_scannetv2.json")
    questions = read_json_or_jsonl(q_path)
    annotations = read_json_or_jsonl(a_path)
    q_rows = questions.get("questions", questions) if isinstance(questions, dict) else questions
    a_rows = annotations.get("annotations", annotations) if isinstance(annotations, dict) else annotations
    anns = {str(row.get("question_id", row.get("id"))): row for row in a_rows}
    out = []
    for q in q_rows:
        qid = str(q.get("question_id", q.get("id", len(out))))
        ann = anns.get(qid, {})
        answer = ann.get("answer", ann.get("answers", [""])[0] if isinstance(ann.get("answers"), list) else "")
        scene_id = q.get("scene_id", ann.get("scene_id", "unknown"))
        out.append(
            common_example(
                dataset="sqa3d",
                sample_id=f"sqa3d_{qid}",
                question=str(q.get("question", q.get("situation", ""))).strip(),
                answer=str(answer),
                image_path=None,
                metadata={
                    "scene_id": scene_id,
                    "question_id": qid,
                    "annotation_only": True,
                    "access_status": "prepared_local_root",
                    "scene_asset_status": "missing_unless_scannet_root_is_provided",
                    "position": ann.get("position", q.get("position")),
                    "rotation": ann.get("rotation", q.get("rotation")),
                },
            )
        )
    return out


def load_examples(root: str | None = None, limit: int | None = 8, smoke: bool = True) -> list[dict[str, Any]]:
    root_path = require_root(root, "SQA3D", INSTRUCTIONS)
    try:
        return limited(_records(root_path), limit if smoke else limit)
    except FileNotFoundError as exc:
        raise RuntimeError(f"SQA3D expected file missing under {root_path}: {exc}. {INSTRUCTIONS}") from exc
