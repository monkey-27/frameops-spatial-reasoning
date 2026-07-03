from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from frameops.slots import SpatialSlot
from frameops.utils import ensure_tensor


@dataclass
class ReferenceFrame:
    origin: torch.Tensor
    basis: torch.Tensor
    frame_type: str
    label: str
    confidence: torch.Tensor

    def __post_init__(self) -> None:
        self.origin = ensure_tensor(self.origin).reshape(3)
        self.basis = ensure_tensor(self.basis).reshape(3, 3)
        self.confidence = ensure_tensor(self.confidence).reshape(())

    def to(self, device: torch.device | str) -> "ReferenceFrame":
        return replace(
            self,
            origin=self.origin.to(device),
            basis=self.basis.to(device),
            confidence=self.confidence.to(device),
        )


def yaw_to_basis(yaw: float | torch.Tensor) -> torch.Tensor:
    yaw_t = ensure_tensor(yaw)
    c = torch.cos(yaw_t)
    s = torch.sin(yaw_t)
    return torch.stack(
        [
            torch.stack([c, s, torch.zeros_like(c)]),
            torch.stack([-s, c, torch.zeros_like(c)]),
            torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
        ],
        dim=1,
    )


def world_frame() -> ReferenceFrame:
    return ReferenceFrame(
        origin=torch.zeros(3),
        basis=torch.eye(3),
        frame_type="world",
        label="world",
        confidence=torch.tensor(1.0),
    )


def camera_frame(camera_pose: dict[str, Any]) -> ReferenceFrame:
    origin = camera_pose.get("position", camera_pose.get("origin", [0.0, 0.0, 0.0]))
    if "basis" in camera_pose:
        basis = camera_pose["basis"]
    elif "rotation_matrix" in camera_pose:
        basis = camera_pose["rotation_matrix"]
    else:
        basis = yaw_to_basis(camera_pose.get("yaw", 0.0))
    return ReferenceFrame(
        origin=origin,
        basis=basis,
        frame_type="camera",
        label=camera_pose.get("label", "camera"),
        confidence=torch.tensor(float(camera_pose.get("confidence", 1.0))),
    )


def object_frame(slot: SpatialSlot) -> ReferenceFrame:
    return ReferenceFrame(
        origin=slot.center,
        basis=yaw_to_basis(slot.yaw),
        frame_type="object",
        label=f"object:{slot.object_id}",
        confidence=slot.confidence,
    )


def agent_frame(
    origin: list[float] | torch.Tensor,
    facing_target_or_yaw: list[float] | torch.Tensor | float,
    *,
    label: str = "agent",
    confidence: float = 1.0,
) -> ReferenceFrame:
    origin_t = ensure_tensor(origin).reshape(3)
    target = ensure_tensor(facing_target_or_yaw)
    if target.numel() == 1:
        yaw = target.reshape(())
    else:
        delta = target.reshape(3) - origin_t
        yaw = torch.atan2(delta[1], delta[0])
    return ReferenceFrame(
        origin=origin_t,
        basis=yaw_to_basis(yaw),
        frame_type="agent",
        label=label,
        confidence=torch.tensor(confidence),
    )


def query_frame(
    origin: list[float] | torch.Tensor,
    basis: list[list[float]] | torch.Tensor | None = None,
    *,
    yaw: float | torch.Tensor | None = None,
    label: str = "query",
    confidence: float = 1.0,
) -> ReferenceFrame:
    if basis is None:
        basis = yaw_to_basis(0.0 if yaw is None else yaw)
    return ReferenceFrame(
        origin=origin,
        basis=basis,
        frame_type="query-derived",
        label=label,
        confidence=torch.tensor(confidence),
    )
