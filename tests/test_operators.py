from __future__ import annotations

import math

import torch

from frameops.frames import ReferenceFrame, object_frame, world_frame, yaw_to_basis
from frameops.operators import (
    closer_than,
    compare_axis,
    compose_vectors,
    counterfactual_rotate_frame,
    line_of_sight,
    rebase,
    relative_vector,
)
from frameops.slots import SpatialBatch, SpatialSlot


def test_rebase_world_identity() -> None:
    pts = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(rebase(pts, world_frame()), pts)


def test_left_right_comparison_under_rotated_frame() -> None:
    frame = ReferenceFrame(
        origin=torch.zeros(3),
        basis=yaw_to_basis(math.pi / 2),
        frame_type="agent",
        label="rotated",
        confidence=torch.tensor(1.0),
    )
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    assert compare_axis(a, b, frame, "x") > 0.9


def test_vector_composition() -> None:
    v_ab = torch.tensor([1.0, 2.0, 0.0])
    v_bc = torch.tensor([-0.5, 1.0, 3.0])
    assert torch.allclose(compose_vectors(v_ab, v_bc), torch.tensor([0.5, 3.0, 3.0]))


def test_relative_vector_and_closer_than() -> None:
    frame = world_frame()
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 2.0, 0.0])
    assert torch.allclose(relative_vector(a, b, frame), torch.tensor([0.0, 2.0, 0.0]))
    assert closer_than(a, b, torch.zeros(3)) > 0


def test_line_of_sight_simple_cases() -> None:
    viewer = torch.tensor([0.0, 0.0, 0.0])
    target = torch.tensor([0.0, 4.0, 0.0])
    blocking = SpatialSlot("occ", "gray_cube", [0.0, 2.0, 0.0], [0.8, 0.8, 0.8], 0.0, 1.0)
    off_ray = SpatialSlot("off", "blue_cube", [2.0, 2.0, 0.0], [0.4, 0.4, 0.4], 0.0, 1.0)
    assert line_of_sight(viewer, target, SpatialBatch([blocking])) < line_of_sight(viewer, target, SpatialBatch([off_ray]))


def test_counterfactual_rotate_frame_changes_axis() -> None:
    frame = world_frame()
    rotated = counterfactual_rotate_frame(frame, math.pi / 2)
    assert compare_axis(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]), rotated, "x") > 0.9


def test_object_frame_origin() -> None:
    slot = SpatialSlot("a", "red_cube", [2.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.0, 1.0)
    assert torch.allclose(rebase(slot.center, object_frame(slot)), torch.zeros(3))
