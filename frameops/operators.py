from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from frameops.frames import ReferenceFrame, yaw_to_basis
from frameops.slots import SpatialBatch, SpatialSlot
from frameops.utils import ensure_tensor

AXIS_INDEX = {"x": 0, "right": 0, "left": 0, "y": 1, "front": 1, "depth": 1, "z": 2, "up": 2, "above": 2}


def _axis_idx(axis: int | str) -> int:
    if isinstance(axis, str):
        if axis not in AXIS_INDEX:
            raise ValueError(f"Unknown axis {axis!r}")
        return AXIS_INDEX[axis]
    if axis not in (0, 1, 2):
        raise ValueError(f"Axis must be 0, 1, or 2, got {axis}")
    return axis


def rebase(points: torch.Tensor, frame: ReferenceFrame) -> torch.Tensor:
    pts = ensure_tensor(points)
    original_shape = pts.shape
    pts = pts.reshape(-1, 3)
    coords = (pts - frame.origin.to(pts.device, pts.dtype)) @ frame.basis.to(pts.device, pts.dtype)
    return coords.reshape(original_shape)


def relative_vector(a: torch.Tensor, b: torch.Tensor, frame: ReferenceFrame) -> torch.Tensor:
    a_t = ensure_tensor(a)
    b_t = ensure_tensor(b)
    return (b_t - a_t) @ frame.basis.to(a_t.device, a_t.dtype)


def compare_axis(
    a: torch.Tensor,
    b: torch.Tensor,
    frame: ReferenceFrame,
    axis: int | str,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    rel = relative_vector(a, b, frame)
    return rel[..., _axis_idx(axis)] / temperature


def distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.norm(ensure_tensor(a) - ensure_tensor(b), dim=-1).clamp_min(eps)


def closer_than(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor, *, temperature: float = 1.0) -> torch.Tensor:
    return (distance(ref, b) - distance(ref, a)) / temperature


def compose_vectors(v_ab: torch.Tensor, v_bc: torch.Tensor) -> torch.Tensor:
    return ensure_tensor(v_ab) + ensure_tensor(v_bc)


def project_points(points: torch.Tensor, camera: dict, eps: float = 1e-4) -> torch.Tensor:
    pts = ensure_tensor(points)
    origin = ensure_tensor(camera.get("position", [0.0, 0.0, 0.0])).to(pts)
    yaw = camera.get("yaw", 0.0)
    basis = ensure_tensor(camera.get("basis", yaw_to_basis(yaw))).to(pts)
    local = (pts.reshape(-1, 3) - origin) @ basis
    depth = local[:, 1].clamp_min(eps)
    focal = float(camera.get("focal", camera.get("focal_length", 120.0)))
    width = float(camera.get("image_width", camera.get("width", 256)))
    height = float(camera.get("image_height", camera.get("height", 256)))
    u = width * 0.5 + focal * local[:, 0] / depth
    v = height * 0.72 - focal * local[:, 2] / depth
    return torch.stack([u, v, depth], dim=-1).reshape(*pts.shape[:-1], 3)


def _candidate_tensors(candidates: SpatialBatch | Sequence[SpatialSlot] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(candidates, SpatialBatch):
        return candidates.centers, candidates.extents
    if isinstance(candidates, torch.Tensor):
        centers = candidates.reshape(-1, 3)
        extents = torch.full_like(centers, 0.25)
        return centers, extents
    centers = torch.stack([slot.center for slot in candidates], dim=0)
    extents = torch.stack([slot.extent for slot in candidates], dim=0)
    return centers, extents


def line_of_sight(
    viewer: torch.Tensor,
    target: torch.Tensor,
    occluder_slots: SpatialBatch | Sequence[SpatialSlot] | torch.Tensor,
    *,
    softness: float = 0.08,
) -> torch.Tensor:
    viewer_t = ensure_tensor(viewer).reshape(3)
    target_t = ensure_tensor(target).reshape(3)
    centers, extents = _candidate_tensors(occluder_slots)
    centers = centers.to(viewer_t)
    extents = extents.to(viewer_t)
    ray = target_t - viewer_t
    ray_len_sq = torch.dot(ray, ray).clamp_min(1e-8)
    t = ((centers - viewer_t) @ ray) / ray_len_sq
    closest = viewer_t + t[:, None] * ray
    perp = torch.linalg.norm(centers - closest, dim=-1)
    radius = 0.5 * torch.linalg.norm(extents[:, [0, 2]], dim=-1)
    between = torch.sigmoid((t - 0.05) / softness) * torch.sigmoid((0.95 - t) / softness)
    clearance = perp - radius
    blockage = between * torch.sigmoid(-clearance / softness)
    clear_logit = -torch.logsumexp(torch.log(blockage.clamp_min(1e-6)), dim=0)
    return clear_logit


def occlusion_score(
    viewer: torch.Tensor,
    target: torch.Tensor,
    candidates: SpatialBatch | Sequence[SpatialSlot] | torch.Tensor,
) -> torch.Tensor:
    return torch.sigmoid(-line_of_sight(viewer, target, candidates))


def counterfactual_translate(slot: SpatialSlot, delta: torch.Tensor | list[float]) -> SpatialSlot:
    return replace(slot, center=slot.center + ensure_tensor(delta).reshape(3).to(slot.center))


def counterfactual_rotate_frame(frame: ReferenceFrame, yaw_delta: float | torch.Tensor) -> ReferenceFrame:
    rot = yaw_to_basis(yaw_delta).to(frame.basis)
    return replace(frame, basis=rot @ frame.basis)
