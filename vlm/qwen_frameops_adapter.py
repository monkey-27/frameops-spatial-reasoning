from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch
from torch import nn


@dataclass
class AdapterState:
    target: Any
    original_get_image_features: Callable[..., Any]


class QwenFrameOpsConnectorAdapter(nn.Module):
    """Experimental hidden-evidence adapter for Qwen3-VL visual tokens.

    This does not fake decoder prefix tokens. It wraps Qwen3-VL's image-feature
    extraction, projects FrameOps evidence into the visual hidden size, and adds
    a gated summary to the returned visual-token features while preserving token
    counts, masks, grids, and M-RoPE handling.
    """

    def __init__(self, evidence_dim: int, hidden_size: int, *, gate_init: float = -4.0):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(evidence_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self._evidence: torch.Tensor | None = None
        self._state: AdapterState | None = None

    def set_evidence(self, evidence_features: torch.Tensor | None) -> None:
        self._evidence = evidence_features

    def attach(self, model: Any) -> None:
        target = getattr(model, "model", model)
        if not hasattr(target, "get_image_features"):
            raise RuntimeError(
                "Qwen hidden evidence adapter expected `model.model.get_image_features` "
                "or `model.get_image_features`, but no such method was found."
            )
        original = target.get_image_features
        adapter = self

        def wrapped_get_image_features(this, *args, **kwargs):
            output = original(*args, **kwargs)
            if adapter._evidence is None:
                return output
            ev = adapter._evidence.to(next(adapter.parameters()).device)
            delta = adapter.project(ev)
            if delta.ndim == 2:
                delta = delta[:, None, :]
            delta_summary = delta.mean(dim=1)
            gate = torch.sigmoid(adapter.gate)

            if hasattr(output, "pooler_output"):
                pooled = output.pooler_output
                add = delta_summary.to(pooled.device, pooled.dtype)
                if add.shape[0] == 1 and pooled.shape[0] != 1:
                    add = add.expand(pooled.shape[0], -1)
                output.pooler_output = pooled + gate.to(pooled.device, pooled.dtype) * add[:, None, :]
                if getattr(output, "deepstack_features", None) is not None:
                    output.deepstack_features = [
                        feat + gate.to(feat.device, feat.dtype) * add.to(feat.device, feat.dtype)[:, None, :]
                        for feat in output.deepstack_features
                    ]
                return output
            if isinstance(output, torch.Tensor):
                add = delta_summary.to(output.device, output.dtype)
                if add.shape[0] == 1 and output.shape[0] != 1:
                    add = add.expand(output.shape[0], -1)
                return output + gate.to(output.device, output.dtype) * add[:, None, :]
            return output

        target.get_image_features = MethodType(wrapped_get_image_features, target)
        self._state = AdapterState(target=target, original_get_image_features=original)

    def detach(self) -> None:
        if self._state is not None:
            self._state.target.get_image_features = self._state.original_get_image_features
            self._state = None
