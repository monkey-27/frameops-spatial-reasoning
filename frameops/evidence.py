from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from frameops.model import execute_program
from frameops.slots import SpatialBatch


@dataclass
class FrameOpsEvidence:
    answer: str
    confidence: float
    features: torch.Tensor
    trace: list[dict[str, Any]]
    selected_objects: list[str]
    frame: str

    def to_text(self) -> str:
        trace_bits = "; ".join(
            f"{step.get('op')}={step.get('logit', step.get('vector', step.get('delta', '')))}"
            for step in self.trace
        )
        return (
            "FrameOps evidence: "
            f"selected_objects={self.selected_objects}; frame={self.frame}; "
            f"predicted_answer={self.answer}; confidence={self.confidence:.3f}; trace={trace_bits}"
        )


def serialize_geometry_json(sample: dict[str, Any]) -> str:
    payload = {
        "objects": [
            {
                "id": obj["id"],
                "label": obj.get("label"),
                "center": obj.get("center"),
                "extent": obj.get("extent"),
                "yaw": obj.get("yaw"),
                "visibility": obj.get("visibility"),
            }
            for obj in sample.get("objects", [])
        ],
        "camera": sample.get("camera", {}),
        "frames": sample.get("frames", []),
    }
    return json.dumps(payload, sort_keys=True)


def serialize_geometry_nl(sample: dict[str, Any]) -> str:
    parts = []
    for obj in sample.get("objects", []):
        parts.append(
            f"{obj['id']} is a {obj.get('color', '')} {obj.get('shape', obj.get('label', 'object'))} "
            f"centered at {obj.get('center')} with extent {obj.get('extent')} and yaw {obj.get('yaw')}."
        )
    return " ".join(parts)


def build_evidence(sample: dict[str, Any], slots: SpatialBatch | None = None) -> FrameOpsEvidence:
    execution = execute_program(sample, slots=slots)
    selected = []
    for step in sample.get("program", []):
        for key in ("a", "b", "c", "ref", "slot", "target", "viewer"):
            val = step.get(key)
            if isinstance(val, str) and val.startswith("obj_") and val not in selected:
                selected.append(val)
    frame = "world"
    for step in sample.get("program", []):
        if "frame" in step:
            frame = str(step["frame"])
            break
    confidence = float(torch.sigmoid(torch.tensor(abs(execution.logit))).item())
    return FrameOpsEvidence(
        answer=execution.answer,
        confidence=confidence,
        features=execution.features,
        trace=execution.trace,
        selected_objects=selected,
        frame=frame,
    )


def serialize_frameops_text(sample: dict[str, Any], slots: SpatialBatch | None = None) -> str:
    return build_evidence(sample, slots=slots).to_text()


class EvidenceProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_tokens: int = 8):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_tokens * hidden_dim),
        )

    def forward(self, evidence_features: torch.Tensor) -> torch.Tensor:
        if evidence_features.ndim == 1:
            evidence_features = evidence_features.unsqueeze(0)
        projected = self.net(evidence_features)
        return projected.view(evidence_features.shape[0], self.num_tokens, self.hidden_dim)
