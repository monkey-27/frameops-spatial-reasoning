from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from frameops.operators import line_of_sight
from frameops.slots import SpatialBatch, SpatialSlot
from frameops.utils import write_jsonl

COLORS = {
    "red": (220, 64, 54),
    "blue": (52, 112, 220),
    "green": (70, 170, 95),
    "yellow": (230, 190, 70),
    "purple": (150, 90, 210),
    "cyan": (70, 190, 200),
    "orange": (230, 130, 60),
    "gray": (145, 145, 150),
}
SHAPES = ["cube", "sphere", "cylinder", "cone"]
TASKS = [
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
]
DIAGNOSTIC_TASKS = TASKS + ["distractor_heavy_relation"]
DIAGNOSTIC_HARDNESS = ["easy", "medium", "hard"]


@dataclass
class ObjectSpec:
    id: str
    shape: str
    color: str
    label: str
    center: list[float]
    extent: list[float]
    yaw: float
    confidence: float = 1.0
    visibility: float = 1.0
    bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shape": self.shape,
            "color": self.color,
            "label": self.label,
            "center": [round(float(v), 4) for v in self.center],
            "extent": [round(float(v), 4) for v in self.extent],
            "yaw": round(float(self.yaw), 4),
            "confidence": self.confidence,
            "visibility": round(float(self.visibility), 4),
            "bbox": None if self.bbox is None else [round(float(v), 2) for v in self.bbox],
        }


def describe(obj: ObjectSpec | dict[str, Any]) -> str:
    if isinstance(obj, dict):
        return f"{obj['color']} {obj['shape']} ({obj['id']})"
    return f"{obj.color} {obj.shape} ({obj.id})"


def _project(center: list[float], image_size: int) -> tuple[float, float, float]:
    x, y, z = center
    focal = image_size * 0.62
    depth = max(y, 0.2)
    u = image_size * 0.5 + focal * x / depth
    v = image_size * 0.72 - focal * z / depth
    return u, v, depth


def _render_scene(objects: list[ObjectSpec], image_path: Path, image_size: int) -> None:
    image = Image.new("RGB", (image_size, image_size), (238, 240, 244))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([0, int(image_size * 0.72), image_size, image_size], fill=(216, 220, 224, 255))
    draw.line([0, int(image_size * 0.72), image_size, int(image_size * 0.72)], fill=(180, 185, 190), width=2)

    projected: list[tuple[float, ObjectSpec, tuple[float, float, float], float]] = []
    for obj in objects:
        u, v, depth = _project(obj.center, image_size)
        scale = image_size * 0.42 / depth
        radius = max(7.0, scale * max(obj.extent[0], obj.extent[2]))
        bbox = [u - radius, v - radius, u + radius, v + radius]
        obj.bbox = bbox
        projected.append((depth, obj, (u, v, depth), radius))

    occupancy = np.zeros((image_size, image_size), dtype=np.float32)
    for _, obj, _, _ in sorted(projected, key=lambda row: row[0], reverse=True):
        assert obj.bbox is not None
        x0, y0, x1, y1 = [int(round(v)) for v in obj.bbox]
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(image_size - 1, x1), min(image_size - 1, y1)
        if x1c <= x0c or y1c <= y0c:
            obj.visibility = 0.0
            continue
        prior = occupancy[y0c:y1c, x0c:x1c].mean() if x1c > x0c and y1c > y0c else 0.0
        obj.visibility = float(max(0.0, 1.0 - prior))
        occupancy[y0c:y1c, x0c:x1c] = 1.0

        color = COLORS[obj.color]
        fill = (*color, 230)
        outline = (35, 35, 40, 255)
        if obj.shape == "sphere":
            draw.ellipse(obj.bbox, fill=fill, outline=outline, width=2)
        elif obj.shape == "cone":
            u = (obj.bbox[0] + obj.bbox[2]) * 0.5
            draw.polygon([(u, obj.bbox[1]), (obj.bbox[2], obj.bbox[3]), (obj.bbox[0], obj.bbox[3])], fill=fill, outline=outline)
        elif obj.shape == "cylinder":
            box_w = max(1.0, obj.bbox[2] - obj.bbox[0])
            box_h = max(1.0, obj.bbox[3] - obj.bbox[1])
            radius_px = int(max(1.0, min(6.0, box_w * 0.25, box_h * 0.25)))
            draw.rounded_rectangle(obj.bbox, radius=radius_px, fill=fill, outline=outline, width=2)
            draw.ellipse([obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[1] + (obj.bbox[3] - obj.bbox[1]) * 0.28], fill=(*color, 245), outline=outline, width=1)
        else:
            draw.rectangle(obj.bbox, fill=fill, outline=outline, width=2)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def _sample_objects(rng: random.Random, n_objects: int, *, hard: bool = False) -> list[ObjectSpec]:
    objects: list[ObjectSpec] = []
    colors = list(COLORS)
    rng.shuffle(colors)
    for idx in range(n_objects):
        if hard and idx > 0 and rng.random() < 0.45:
            base = objects[rng.randrange(len(objects))]
            x = base.center[0] + rng.uniform(-0.55, 0.55)
            y = base.center[1] + rng.uniform(-0.45, 0.45)
        else:
            x = rng.uniform(-2.8, 2.8)
            y = rng.uniform(2.3, 7.8)
        z = rng.uniform(0.25, 2.6)
        extent = [rng.uniform(0.35, 0.85), rng.uniform(0.35, 0.85), rng.uniform(0.35, 0.95)]
        shape = rng.choice(SHAPES)
        color = colors[idx % len(colors)]
        obj_id = f"obj_{idx}"
        objects.append(
            ObjectSpec(
                id=obj_id,
                shape=shape,
                color=color,
                label=f"{color}_{shape}",
                center=[x, y, z],
                extent=extent,
                yaw=rng.uniform(-math.pi, math.pi),
            )
        )
    return objects


def _rotated_x(point: list[float], origin: list[float], yaw: float) -> float:
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    return math.cos(yaw) * dx + math.sin(yaw) * dy


def _dist(a: ObjectSpec, b: ObjectSpec) -> float:
    return float(np.linalg.norm(np.asarray(a.center) - np.asarray(b.center)))


def _simple_los(viewer: list[float], target: ObjectSpec, objects: list[ObjectSpec]) -> bool:
    occluders = [
        SpatialSlot(obj.id, obj.label, obj.center, obj.extent, obj.yaw, obj.confidence)
        for obj in objects
        if obj.id != target.id
    ]
    if not occluders:
        return True
    logit = line_of_sight(
        torch.tensor(viewer, dtype=torch.float32),
        torch.tensor(target.center, dtype=torch.float32),
        SpatialBatch(occluders),
    )
    return bool(float(logit) > 0.0)


def _make_question(rng: random.Random, objects: list[ObjectSpec], task: str, difficulty: str) -> dict[str, Any]:
    a, b = rng.sample(objects, 2)
    camera = {"position": [0.0, 0.0, 1.2], "yaw": 0.0, "image_width": 256, "image_height": 256, "focal": 158.0}
    frames = [
        {"id": "world", "type": "world", "origin": [0.0, 0.0, 0.0], "basis": "identity"},
        {"id": "camera", "type": "camera", "origin": camera["position"], "yaw": 0.0},
    ]

    if task == "left_right_camera":
        answer = "yes" if a.center[0] < b.center[0] else "no"
        question = f"From the camera view, is the {describe(a)} left of the {describe(b)}?"
        program = [{"op": "compare_axis", "frame": "camera", "a": b.id, "b": a.id, "axis": "x", "relation": "left"}]
    elif task == "front_back_depth":
        answer = "yes" if a.center[1] < b.center[1] else "no"
        question = f"Is the {describe(a)} in front of the {describe(b)}, meaning closer to the camera?"
        program = [{"op": "compare_axis", "frame": "camera", "a": b.id, "b": a.id, "axis": "y", "relation": "front"}]
    elif task == "above_below":
        answer = "yes" if a.center[2] > b.center[2] else "no"
        question = f"Is the {describe(a)} above the {describe(b)}?"
        program = [{"op": "compare_axis", "frame": "world", "a": b.id, "b": a.id, "axis": "z", "relation": "above"}]
    elif task == "distance_comparison":
        c = rng.choice([obj for obj in objects if obj.id not in {a.id, b.id}])
        answer = "yes" if _dist(a, c) < _dist(b, c) else "no"
        question = f"Is the {describe(a)} closer to the {describe(c)} than the {describe(b)} is?"
        program = [{"op": "closer_than", "a": a.id, "b": b.id, "ref": c.id}]
    elif task == "object_centered_frame":
        frames.append({"id": f"object:{a.id}", "type": "object", "origin": a.center, "yaw": a.yaw})
        answer = "yes" if _rotated_x(b.center, a.center, a.yaw) < 0 else "no"
        question = f"From the {describe(a)}'s own facing direction, is the {describe(b)} on its left?"
        program = [{"op": "compare_axis", "frame": f"object:{a.id}", "a": a.id, "b": b.id, "axis": "x", "relation": "left"}]
    elif task == "agent_centered_frame":
        agent_yaw = rng.choice([0.0, math.pi / 2, -math.pi / 2, math.pi])
        frames.append({"id": "agent", "type": "agent", "origin": [0.0, 0.0, 1.2], "yaw": agent_yaw})
        answer = "yes" if _rotated_x(a.center, [0.0, 0.0, 1.2], agent_yaw) > _rotated_x(b.center, [0.0, 0.0, 1.2], agent_yaw) else "no"
        question = f"From an agent facing yaw {agent_yaw:.2f}, is the {describe(a)} to the right of the {describe(b)}?"
        program = [{"op": "compare_axis", "frame": "agent", "a": b.id, "b": a.id, "axis": "x", "relation": "right"}]
    elif task == "vector_composition":
        c = rng.choice([obj for obj in objects if obj.id not in {a.id, b.id}])
        dz = c.center[2] - a.center[2]
        answer = "yes" if dz > 0 else "no"
        question = f"If you compose the vector from {describe(a)} to {describe(b)} with the vector from {describe(b)} to {describe(c)}, does the result point upward?"
        program = [{"op": "compose_vectors", "a": a.id, "b": b.id, "c": c.id, "axis": "z", "relation": "positive"}]
    elif task == "occlusion_line_of_sight":
        visible = _simple_los(camera["position"], a, objects)
        answer = "yes" if visible else "no"
        question = f"Is the {describe(a)} visible from the camera without another object blocking the line of sight?"
        program = [{"op": "line_of_sight", "viewer": "camera", "target": a.id}]
    elif task == "counterfactual_translation":
        delta = [1.0, 0.0, 0.0]
        moved_x = a.center[0] + delta[0]
        answer = "yes" if moved_x > b.center[0] else "no"
        question = f"If the {describe(a)} moved 1 meter to the right, would it be right of the {describe(b)}?"
        program = [{"op": "counterfactual_translate", "slot": a.id, "delta": delta}, {"op": "compare_axis", "frame": "camera", "a": b.id, "b": a.id, "axis": "x"}]
    elif task == "counterfactual_viewer_rotation":
        yaw_delta = math.pi / 2
        ax = _rotated_x(a.center, camera["position"], yaw_delta)
        bx = _rotated_x(b.center, camera["position"], yaw_delta)
        answer = "yes" if ax < bx else "no"
        question = f"If the viewer rotated 90 degrees left, would the {describe(a)} appear left of the {describe(b)}?"
        program = [{"op": "counterfactual_rotate_frame", "frame": "camera", "yaw_delta": yaw_delta}, {"op": "compare_axis", "a": b.id, "b": a.id, "axis": "x", "relation": "left"}]
    elif task == "perspective_taking":
        viewer = rng.choice([obj for obj in objects if obj.id not in {a.id, b.id}])
        frames.append({"id": f"object:{viewer.id}", "type": "object", "origin": viewer.center, "yaw": viewer.yaw})
        ax = _rotated_x(a.center, viewer.center, viewer.yaw)
        bx = _rotated_x(b.center, viewer.center, viewer.yaw)
        answer = "yes" if ax < bx else "no"
        question = f"From the {describe(viewer)}'s perspective, is the {describe(a)} left of the {describe(b)}?"
        program = [{"op": "perspective_taking", "viewer": viewer.id}, {"op": "compare_axis", "frame": f"object:{viewer.id}", "a": b.id, "b": a.id, "axis": "x", "relation": "left"}]
    elif task == "distractor_heavy_relation":
        c = rng.choice([obj for obj in objects if obj.id not in {a.id, b.id}])
        distractor = max(objects, key=lambda obj: obj.extent[0] * obj.extent[1] * obj.extent[2])
        answer = "yes" if _dist(a, c) < _dist(b, c) else "no"
        question = (
            f"Ignore the visually salient {describe(distractor)}. "
            f"Using 3D positions, is the {describe(a)} closer to the {describe(c)} "
            f"than the {describe(b)} is?"
        )
        program = [{"op": "closer_than", "a": a.id, "b": b.id, "ref": c.id}]
    else:
        raise ValueError(task)

    return {
        "question": question,
        "answer": answer,
        "program": program,
        "frames": frames,
        "camera": camera,
        "metadata": {"task_type": task, "difficulty": difficulty},
    }


def generate_split(
    output_dir: str | Path,
    split: str,
    n_scenes: int,
    *,
    image_size: int,
    min_objects: int,
    max_objects: int,
    seed: int,
    hard: bool = False,
) -> list[dict[str, Any]]:
    out = Path(output_dir)
    image_dir = out / "images" / split
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(n_scenes):
        n_objects = rng.randint(max(min_objects, 5 if hard else min_objects), max_objects + (1 if hard else 0))
        objects = _sample_objects(rng, n_objects, hard=hard)
        task = TASKS[idx % len(TASKS)] if not hard else TASKS[(idx * 3 + 1) % len(TASKS)]
        image_path = image_dir / f"{split}_{idx:05d}.png"
        _render_scene(objects, image_path, image_size)
        qa = _make_question(rng, objects, task, "hard" if hard else "standard")
        row = {
            "id": f"{split}_{idx:05d}",
            "image_path": str(image_path),
            "question": qa["question"],
            "answer": qa["answer"],
            "program": qa["program"],
            "objects": [obj.to_dict() for obj in objects],
            "frames": qa["frames"],
            "camera": qa["camera"] | {"image_width": image_size, "image_height": image_size},
            "metadata": qa["metadata"] | {"num_objects": n_objects},
        }
        rows.append(row)
    write_jsonl(out / f"{split}.jsonl", rows)
    return rows


def generate_dataset(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    seed = int(config.get("seed", 0))
    synth = config.get("synthetic", {})
    out = Path(config.get("output_dir", "data/synthetic/oracle_v0"))
    out.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "train": int(synth.get("train_scenes", 240)),
        "val": int(synth.get("val_scenes", 60)),
        "test": int(synth.get("test_scenes", 80)),
    }
    rows = {}
    for offset, (split, n_scenes) in enumerate(split_specs.items()):
        rows[split] = generate_split(
            out,
            split,
            n_scenes,
            image_size=int(synth.get("image_size", 256)),
            min_objects=int(synth.get("min_objects", 3)),
            max_objects=int(synth.get("max_objects", 7)),
            seed=seed + 1000 * offset,
            hard=split == "test" and bool(synth.get("heldout_test_hard", True)),
        )
    manifest = {
        "seed": seed,
        "splits": {split: len(split_rows) for split, split_rows in rows.items()},
        "task_types": TASKS,
        "notes": "Synthetic oracle v0. Test split biases toward more objects, close distractors, occlusion, and viewpoint changes.",
    }
    with (out / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return rows


def _diagnostic_object_count(rng: random.Random, hardness: str) -> int:
    if hardness == "easy":
        return rng.randint(3, 4)
    if hardness == "medium":
        return rng.randint(5, 6)
    return rng.randint(7, 8)


def _diagnostic_hard_flag(hardness: str) -> bool:
    return hardness in {"medium", "hard"}


def _camera_left(obj_a: ObjectSpec, obj_b: ObjectSpec) -> bool:
    return obj_a.center[0] < obj_b.center[0]


def _diagnostic_flags(row: dict[str, Any]) -> list[str]:
    task = row["metadata"]["task_type"]
    flags: list[str] = []
    objects = {obj["id"]: obj for obj in row["objects"]}
    if task in {"object_centered_frame", "perspective_taking"}:
        compare = next((step for step in row["program"] if step.get("op") == "compare_axis"), None)
        if compare:
            a = objects[compare["b"]]
            b = objects[compare["a"]]
            camera_answer = a["center"][0] < b["center"][0]
            if camera_answer != (row["answer"] == "yes"):
                flags.append("image_frame_differs_from_target_frame")
    if task == "occlusion_line_of_sight" and row["answer"] == "no":
        flags.append("semantically_obvious_but_geometrically_blocked")
    if task == "distractor_heavy_relation":
        flags.append("nearest_object_not_visually_saliencified")
    if task in {"counterfactual_translation", "counterfactual_viewer_rotation"}:
        flags.append("counterfactual_changes_relation")
    return flags


def _make_diagnostic_row(
    *,
    out: Path,
    split: str,
    idx: int,
    task: str,
    hardness: str,
    image_size: int,
    rng: random.Random,
) -> dict[str, Any]:
    hard = _diagnostic_hard_flag(hardness)
    for attempt in range(80):
        objects = _sample_objects(rng, _diagnostic_object_count(rng, hardness), hard=hard)
        if hardness == "hard":
            for obj in objects[1:]:
                if rng.random() < 0.25:
                    obj.center[0] = objects[0].center[0] + rng.uniform(-0.22, 0.22)
                    obj.center[1] = objects[0].center[1] + rng.uniform(-0.22, 0.22)
        qa = _make_question(rng, objects, task, hardness)
        row_id = f"{split}_{task}_{idx:05d}"
        image_path = out / "images" / split / f"{row_id}.png"
        _render_scene(objects, image_path, image_size)
        row = {
            "id": row_id,
            "image_path": str(image_path),
            "question": qa["question"],
            "answer": qa["answer"],
            "program": qa["program"],
            "objects": [obj.to_dict() for obj in objects],
            "frames": qa["frames"],
            "camera": qa["camera"] | {"image_width": image_size, "image_height": image_size},
            "metadata": qa["metadata"]
            | {
                "task_type": task,
                "task_family": task,
                "difficulty": hardness,
                "hardness": hardness,
                "num_objects": len(objects),
                "diagnostic": True,
                "attempt": attempt,
            },
        }
        flags = _diagnostic_flags(row)
        row["metadata"]["diagnostic_flags"] = flags
        if task in {"object_centered_frame", "perspective_taking"} and hardness != "easy":
            if "image_frame_differs_from_target_frame" not in flags:
                continue
        if task == "occlusion_line_of_sight" and hardness == "hard" and row["answer"] != "no":
            continue
        return row
    return row


def generate_diagnostic_dataset(
    output_dir: str | Path,
    *,
    seed: int = 20260703,
    image_size: int = 256,
    test_per_task: int = 50,
    train_per_task: int = 100,
    val_per_task: int = 20,
    enforce_min_test: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_specs = {"train": train_per_task, "val": val_per_task, "test": test_per_task}
    rows: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "seed": seed,
        "image_size": image_size,
        "tasks": DIAGNOSTIC_TASKS,
        "hardness_buckets": DIAGNOSTIC_HARDNESS,
        "splits": {},
        "notes": (
            "Balanced fixed diagnostic split. Test is intended for paired interface comparisons; "
            "do not tune on test after generating it."
        ),
    }
    for split_idx, (split, per_task) in enumerate(split_specs.items()):
        rng = random.Random(seed + 10000 * split_idx)
        split_rows: list[dict[str, Any]] = []
        for task in DIAGNOSTIC_TASKS:
            for task_idx in range(per_task):
                hardness = DIAGNOSTIC_HARDNESS[task_idx % len(DIAGNOSTIC_HARDNESS)]
                split_rows.append(
                    _make_diagnostic_row(
                        out=out,
                        split=split,
                        idx=len(split_rows),
                        task=task,
                        hardness=hardness,
                        image_size=image_size,
                        rng=rng,
                    )
                )
        rng.shuffle(split_rows)
        write_jsonl(out / f"diagnostic_{split}.jsonl", split_rows)
        write_jsonl(out / f"{split}.jsonl", split_rows)
        counts = {}
        for task in DIAGNOSTIC_TASKS:
            counts[task] = sum(1 for row in split_rows if row["metadata"]["task_type"] == task)
        manifest["splits"][split] = {
            "n": len(split_rows),
            "per_task": counts,
            "hardness": {
                bucket: sum(1 for row in split_rows if row["metadata"]["hardness"] == bucket)
                for bucket in DIAGNOSTIC_HARDNESS
            },
            "path": str(out / f"diagnostic_{split}.jsonl"),
        }
        rows[split] = split_rows
    min_test = min(manifest["splits"]["test"]["per_task"].values())
    if enforce_min_test and min_test < 50:
        raise ValueError(f"Diagnostic test split must have at least 50 per task; min={min_test}")
    with (out / "diagnostic_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return rows
