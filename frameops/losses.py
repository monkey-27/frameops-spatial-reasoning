from __future__ import annotations

import torch
from torch.nn import functional as F

from frameops.controller import ControllerOutput


def frameops_supervised_loss(
    output: ControllerOutput,
    *,
    answer_labels: torch.Tensor,
    operator_labels: torch.Tensor,
    object_labels: torch.Tensor | None = None,
    frame_labels: torch.Tensor | None = None,
    object_weight: float = 0.25,
    frame_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_answer = F.cross_entropy(output.answer_logits, answer_labels)
    loss_operator = F.cross_entropy(output.operator_logits, operator_labels)
    loss = loss_answer + loss_operator
    metrics = {
        "loss_answer": float(loss_answer.detach().cpu()),
        "loss_operator": float(loss_operator.detach().cpu()),
    }
    if object_labels is not None:
        loss_object = F.binary_cross_entropy_with_logits(output.object_logits, object_labels.float())
        loss = loss + object_weight * loss_object
        metrics["loss_object"] = float(loss_object.detach().cpu())
    if frame_labels is not None:
        loss_frame = F.cross_entropy(output.frame_logits, frame_labels)
        loss = loss + frame_weight * loss_frame
        metrics["loss_frame"] = float(loss_frame.detach().cpu())
    metrics["loss"] = float(loss.detach().cpu())
    return loss, metrics
