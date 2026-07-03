from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from frameops.controller import FrameOpsController
from frameops.model import (
    ANSWER_LABELS,
    FRAME_LABELS,
    ID_TO_ANSWER,
    execute_program,
    frame_label_from_program,
    selected_object_ids,
    task_from_sample,
)
from frameops.slots import SpatialBatch
from frameops.utils import read_jsonl, write_jsonl
from scripts.train_frameops import (
    _encode_question,
    _slot_feature,
)

DIAGNOSTIC_TASKS = [
    "left_right_camera",
    "front_back_depth",
    "above_below",
    "distance_comparison",
    "object_centered_frame",
    "agent_centered_frame",
    "vector_composition",
    "occlusion_line_of_sight",
    "counterfactual_translation",
    "counterfactual_viewer_rotation",
    "perspective_taking",
    "distractor_heavy_relation",
]
HARDNESS_BUCKETS = ["easy", "medium", "hard"]
NOISE_LEVELS = [0.05, 0.10, 0.20]


def canonical_operator_label(task: str) -> str:
    return "distance_comparison" if task == "distractor_heavy_relation" else task


def diagnostic_timestamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def ensure_out_dir(out_dir: str | Path | None) -> Path:
    if out_dir is None:
        out_dir = Path("results") / "diagnostic_latest"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def accuracy_ci(values: list[bool], n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    acc = sum(values) / len(values)
    rng = random.Random(seed)
    samples = []
    n = len(values)
    for _ in range(n_boot):
        samples.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    samples.sort()
    return {
        "mean": acc,
        "low": samples[int(0.025 * (n_boot - 1))],
        "high": samples[int(0.975 * (n_boot - 1))],
        "n": n,
    }


def paired_delta_ci(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict[str, float]:
    a = {row["id"]: bool(row["correct"]) for row in rows_a}
    b = {row["id"]: bool(row["correct"]) for row in rows_b}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    diffs = [float(a[i]) - float(b[i]) for i in ids]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        samples.append(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs))
    samples.sort()
    return {
        "mean": mean,
        "low": samples[int(0.025 * (n_boot - 1))],
        "high": samples[int(0.975 * (n_boot - 1))],
        "n": len(ids),
    }


def mcnemar_counts(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> dict[str, int]:
    a = {row["id"]: bool(row["correct"]) for row in rows_a}
    b = {row["id"]: bool(row["correct"]) for row in rows_b}
    counts = Counter()
    for sample_id in sorted(set(a) & set(b)):
        counts[f"a_{int(a[sample_id])}_b_{int(b[sample_id])}"] += 1
    return dict(counts)


def summarize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    correct = [bool(row["correct"]) for row in rows]
    summary: dict[str, Any] = {
        "overall": accuracy_ci(correct),
        "n": len(rows),
        "by_task_family": {},
        "by_hardness": {},
        "by_noise": {},
    }
    for field, out_key in [("task_family", "by_task_family"), ("hardness", "by_hardness"), ("noise", "by_noise")]:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(field, "none"))].append(bool(row["correct"]))
        summary[out_key] = {key: accuracy_ci(vals) for key, vals in sorted(buckets.items())}
    if baseline_rows is not None:
        summary["paired_delta_vs_baseline"] = paired_delta_ci(rows, baseline_rows)
        summary["mcnemar_vs_baseline"] = mcnemar_counts(rows, baseline_rows)
    return summary


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_controller(checkpoint: str | Path) -> tuple[FrameOpsController, dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = FrameOpsController(
        vocab_size=len(ckpt["vocab"]),
        slot_dim=ckpt["slot_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_operators=len(ckpt["task_vocab"]),
        num_frames=len(FRAME_LABELS),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@dataclass
class ControllerRouting:
    answer: str
    answer_confidence: float
    predicted_operator: str
    predicted_frame: str
    predicted_objects: list[str]
    object_scores: dict[str, float]
    operator_scores: dict[str, float]
    frame_scores: dict[str, float]
    object_top1_hit: bool
    object_top3_all: bool
    frame_correct: bool
    operator_correct: bool
    all_routing_correct: bool


def predict_routing(sample: dict[str, Any], model: FrameOpsController, ckpt: dict[str, Any]) -> ControllerRouting:
    question = torch.tensor([_encode_question(sample["question"], ckpt["vocab"])], dtype=torch.long)
    slot_rows = [_slot_feature(obj, ckpt["label_vocab"]) for obj in sample["objects"]]
    slots = torch.stack(slot_rows, dim=0).unsqueeze(0)
    slot_mask = torch.ones(1, len(slot_rows), dtype=torch.bool)
    with torch.no_grad():
        out = model(question, slots, slot_mask)
    answer_probs = torch.softmax(out.answer_logits[0], dim=-1)
    op_probs = torch.softmax(out.operator_logits[0], dim=-1)
    frame_probs = torch.softmax(out.frame_logits[0], dim=-1)
    object_probs = torch.sigmoid(out.object_logits[0, : len(sample["objects"])])

    inv_task = {idx: task for task, idx in ckpt["task_vocab"].items()}
    inv_frame = {idx: frame for frame, idx in FRAME_LABELS.items()}
    pred_op = inv_task[int(op_probs.argmax())]
    pred_frame = inv_frame[int(frame_probs.argmax())]
    ranked_indices = torch.argsort(object_probs, descending=True).tolist()
    object_ids = [obj["id"] for obj in sample["objects"]]
    gold_objects = selected_object_ids(sample)
    n_select = max(1, len(gold_objects))
    pred_objects = [object_ids[i] for i in ranked_indices[:n_select]]
    top3 = {object_ids[i] for i in ranked_indices[: min(3, len(object_ids))]}
    frame_gold = frame_label_from_program(sample)
    op_gold = canonical_operator_label(task_from_sample(sample))
    return ControllerRouting(
        answer=ID_TO_ANSWER[int(answer_probs.argmax())],
        answer_confidence=float(answer_probs.max()),
        predicted_operator=pred_op,
        predicted_frame=pred_frame,
        predicted_objects=pred_objects,
        object_scores={object_ids[i]: float(object_probs[i]) for i in range(len(object_ids))},
        operator_scores={inv_task[i]: float(op_probs[i]) for i in range(len(inv_task))},
        frame_scores={inv_frame[i]: float(frame_probs[i]) for i in range(len(inv_frame))},
        object_top1_hit=bool(object_ids[ranked_indices[0]] in gold_objects) if gold_objects else True,
        object_top3_all=bool(gold_objects.issubset(top3)) if gold_objects else True,
        frame_correct=pred_frame == frame_gold,
        operator_correct=pred_op == op_gold,
        all_routing_correct=set(pred_objects) == gold_objects and pred_frame == frame_gold and pred_op == op_gold,
    )


def replace_program_objects(sample: dict[str, Any], predicted_objects: list[str]) -> list[dict[str, Any]]:
    gold_order = []
    for step in sample.get("program", []):
        for key in ("a", "b", "c", "ref", "slot", "target", "viewer"):
            val = step.get(key)
            if isinstance(val, str) and val.startswith("obj_") and val not in gold_order:
                gold_order.append(val)
    mapping = {
        gold: predicted_objects[min(i, len(predicted_objects) - 1)]
        for i, gold in enumerate(gold_order)
        if predicted_objects
    }
    new_program = []
    for step in sample.get("program", []):
        new_step = dict(step)
        for key, val in list(new_step.items()):
            if isinstance(val, str) and val in mapping:
                new_step[key] = mapping[val]
        new_program.append(new_step)
    return new_program


def replace_program_frame(program: list[dict[str, Any]], frame_label: str, object_id: str | None = None) -> list[dict[str, Any]]:
    new_program = []
    frame_value = {
        "world": "world",
        "camera": "camera",
        "agent": "agent",
        "query": "camera",
        "object": f"object:{object_id or 'obj_0'}",
    }.get(frame_label, "world")
    for step in program:
        new_step = dict(step)
        if new_step.get("op") == "compare_axis":
            new_step["frame"] = frame_value
        new_program.append(new_step)
    return new_program


def program_for_operator(sample: dict[str, Any], operator_label: str, object_ids: list[str], frame_label: str) -> list[dict[str, Any]]:
    ids = object_ids or [obj["id"] for obj in sample["objects"][:3]]
    while len(ids) < 3:
        ids.append(ids[-1])
    a, b, c = ids[:3]
    if operator_label == "front_back_depth":
        return [{"op": "compare_axis", "frame": "camera", "a": b, "b": a, "axis": "y", "relation": "front"}]
    if operator_label == "above_below":
        return [{"op": "compare_axis", "frame": "world", "a": b, "b": a, "axis": "z", "relation": "above"}]
    if operator_label == "distance_comparison":
        return [{"op": "closer_than", "a": a, "b": b, "ref": c}]
    if operator_label == "object_centered_frame":
        return [{"op": "compare_axis", "frame": f"object:{a}", "a": a, "b": b, "axis": "x", "relation": "left"}]
    if operator_label == "agent_centered_frame":
        return [{"op": "compare_axis", "frame": "agent", "a": b, "b": a, "axis": "x", "relation": "right"}]
    if operator_label == "vector_composition":
        return [{"op": "compose_vectors", "a": a, "b": b, "c": c, "axis": "z", "relation": "positive"}]
    if operator_label == "occlusion_line_of_sight":
        return [{"op": "line_of_sight", "viewer": "camera", "target": a}]
    if operator_label == "counterfactual_translation":
        return [{"op": "counterfactual_translate", "slot": a, "delta": [1.0, 0.0, 0.0]}, {"op": "compare_axis", "frame": "camera", "a": b, "b": a, "axis": "x", "relation": "right"}]
    if operator_label == "counterfactual_viewer_rotation":
        return [{"op": "counterfactual_rotate_frame", "frame": "camera", "yaw_delta": math.pi / 2}, {"op": "compare_axis", "a": b, "b": a, "axis": "x", "relation": "left"}]
    if operator_label == "perspective_taking":
        return [{"op": "perspective_taking", "viewer": c}, {"op": "compare_axis", "frame": f"object:{c}", "a": b, "b": a, "axis": "x", "relation": "left"}]
    return [{"op": "compare_axis", "frame": "camera", "a": b, "b": a, "axis": "x", "relation": "left"}]


def execute_with_program(sample: dict[str, Any], program: list[dict[str, Any]], slots: SpatialBatch | None = None) -> dict[str, Any]:
    patched = dict(sample)
    patched["program"] = program
    pred = execute_program(patched, slots=slots)
    return {"answer": pred.answer, "logit": pred.logit, "trace": pred.trace}


def rows_by_mode(predictions_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(predictions_path):
        grouped[row["mode"]].append(row)
    return grouped


def validate_diagnostic_split(rows: list[dict[str, Any]], min_per_task: int = 50) -> dict[str, Any]:
    counts = Counter(task_from_sample(row) for row in rows)
    missing = {task: counts.get(task, 0) for task in DIAGNOSTIC_TASKS if counts.get(task, 0) < min_per_task}
    if missing:
        raise ValueError(f"Diagnostic split is underfilled: {missing}")
    hardness = Counter(row.get("metadata", {}).get("hardness", row.get("metadata", {}).get("difficulty", "unknown")) for row in rows)
    return {"task_counts": dict(sorted(counts.items())), "hardness_counts": dict(sorted(hardness.items())), "n": len(rows)}
