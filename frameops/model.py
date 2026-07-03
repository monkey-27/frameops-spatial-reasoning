from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from frameops.frames import ReferenceFrame, camera_frame, object_frame, query_frame, world_frame
from frameops.operators import (
    closer_than,
    compare_axis,
    compose_vectors,
    counterfactual_rotate_frame,
    counterfactual_translate,
    line_of_sight,
    relative_vector,
)
from frameops.slots import SpatialBatch, SpatialSlot

ANSWER_LABELS = {"no": 0, "yes": 1}
ID_TO_ANSWER = {v: k for k, v in ANSWER_LABELS.items()}
FRAME_LABELS = {"world": 0, "camera": 1, "object": 2, "agent": 3, "query": 4}


@dataclass
class ProgramExecution:
    answer: str
    logit: float
    features: torch.Tensor
    trace: list[dict[str, Any]]


def task_from_sample(sample: dict[str, Any]) -> str:
    return str(sample.get("metadata", {}).get("task_type", "unknown"))


def operator_from_program(sample: dict[str, Any]) -> str:
    program = sample.get("program", [])
    if not program:
        return "unknown"
    if program[0].get("op", "") == "counterfactual_translate":
        return "counterfactual_translation"
    if program[0].get("op", "") == "counterfactual_rotate_frame":
        return "counterfactual_viewer_rotation"
    if program[0].get("op", "") == "perspective_taking":
        return "perspective_taking"
    return str(program[0].get("op", "unknown"))


def frame_label_from_program(sample: dict[str, Any]) -> str:
    program = sample.get("program", [])
    for step in program:
        frame = str(step.get("frame", ""))
        if frame.startswith("object:"):
            return "object"
        if frame in {"world", "camera", "agent"}:
            return frame
    if operator_from_program(sample).startswith("counterfactual"):
        return "query"
    return "world"


def selected_object_ids(sample: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for step in sample.get("program", []):
        for key in ("a", "b", "c", "ref", "slot", "target", "viewer"):
            val = step.get(key)
            if isinstance(val, str) and val.startswith("obj_"):
                ids.add(val)
    return ids


def _get_frame(frame_id: str, slots: SpatialBatch, sample: dict[str, Any]) -> ReferenceFrame:
    if frame_id == "world":
        return world_frame()
    if frame_id == "camera":
        return camera_frame(sample["camera"])
    if frame_id == "agent":
        for frame in sample.get("frames", []):
            if frame.get("id") == "agent":
                return query_frame(frame["origin"], yaw=frame.get("yaw", 0.0), label="agent")
        return camera_frame(sample["camera"])
    if frame_id.startswith("object:"):
        return object_frame(slots.by_id(frame_id.split(":", 1)[1]))
    return world_frame()


def execute_program(sample: dict[str, Any], slots: SpatialBatch | None = None) -> ProgramExecution:
    slots = slots or SpatialBatch.from_objects(sample["objects"])
    trace: list[dict[str, Any]] = []
    last_logit = torch.tensor(0.0)
    feature_chunks: list[torch.Tensor] = []
    translated: dict[str, SpatialSlot] = {}
    current_frame: ReferenceFrame | None = None

    def slot(object_id: str) -> SpatialSlot:
        return translated.get(object_id, slots.by_id(object_id))

    for step in sample.get("program", []):
        op = step.get("op")
        if op == "compare_axis":
            frame = current_frame or _get_frame(str(step.get("frame", "world")), slots, sample)
            a = slot(step["a"]).center
            b = slot(step["b"]).center
            raw = compare_axis(a, b, frame, step.get("axis", "x"))
            relation = step.get("relation", "right")
            if relation in {"left", "front"}:
                raw = -raw
            last_logit = raw
            feature_chunks.append(raw.reshape(1))
            trace.append({"op": op, "frame": frame.label, "logit": float(raw)})
        elif op == "closer_than":
            raw = closer_than(slot(step["a"]).center, slot(step["b"]).center, slot(step["ref"]).center)
            last_logit = raw
            feature_chunks.append(raw.reshape(1))
            trace.append({"op": op, "logit": float(raw)})
        elif op == "compose_vectors":
            a = slot(step["a"]).center
            b = slot(step["b"]).center
            c = slot(step["c"]).center
            v = compose_vectors(relative_vector(a, b, world_frame()), relative_vector(b, c, world_frame()))
            raw = v[{"x": 0, "y": 1, "z": 2}[step.get("axis", "z")]]
            last_logit = raw
            feature_chunks.append(v.reshape(-1))
            trace.append({"op": op, "vector": [float(x) for x in v], "logit": float(raw)})
        elif op == "line_of_sight":
            viewer = torch.as_tensor(sample["camera"]["position"], dtype=torch.float32)
            target = slot(step["target"]).center
            occluders = SpatialBatch([s for s in slots if s.object_id != step["target"]])
            raw = line_of_sight(viewer, target, occluders)
            last_logit = raw
            feature_chunks.append(raw.reshape(1))
            trace.append({"op": op, "logit": float(raw)})
        elif op == "counterfactual_translate":
            translated[step["slot"]] = counterfactual_translate(slot(step["slot"]), step["delta"])
            feature_chunks.append(torch.as_tensor(step["delta"], dtype=torch.float32))
            trace.append({"op": op, "slot": step["slot"], "delta": step["delta"]})
        elif op == "counterfactual_rotate_frame":
            current_frame = counterfactual_rotate_frame(camera_frame(sample["camera"]), float(step.get("yaw_delta", math.pi / 2)))
            feature_chunks.append(torch.tensor([float(step.get("yaw_delta", 0.0))]))
            trace.append({"op": op, "yaw_delta": float(step.get("yaw_delta", 0.0))})
        elif op == "perspective_taking":
            current_frame = object_frame(slot(step["viewer"]))
            trace.append({"op": op, "viewer": step["viewer"]})
        else:
            trace.append({"op": op or "unknown", "warning": "not_executed"})

    answer = "yes" if float(last_logit) > 0 else "no"
    features = torch.cat(feature_chunks) if feature_chunks else torch.zeros(1)
    return ProgramExecution(answer=answer, logit=float(last_logit), features=features, trace=trace)
