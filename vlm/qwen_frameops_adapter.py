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

    def _project_delta(self) -> torch.Tensor | None:
        if self._evidence is None:
            return None
        ev = self._evidence.to(next(self.parameters()).device)
        delta = self.project(ev)
        if delta.ndim == 2:
            delta = delta[:, None, :]
        return delta.mean(dim=1)

    def _add_delta(self, output: Any, delta_summary: torch.Tensor, gate: torch.Tensor) -> Any:
        if isinstance(output, torch.Tensor):
            add = delta_summary.to(output.device, output.dtype)
            gate_t = gate.to(output.device, output.dtype)
            if output.ndim == 2:
                add_vec = add[0] if add.ndim == 2 else add
                return output + gate_t * add_vec.unsqueeze(0)
            if output.ndim >= 3:
                if add.shape[0] == 1 and output.shape[0] != 1:
                    add = add.expand(output.shape[0], -1)
                return output + gate_t * add.view(add.shape[0], *([1] * (output.ndim - 2)), add.shape[-1])
            return output
        if isinstance(output, tuple):
            return tuple(self._add_delta(item, delta_summary, gate) for item in output)
        if isinstance(output, list):
            return [self._add_delta(item, delta_summary, gate) for item in output]
        return output

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
            delta_summary = adapter._project_delta()
            if delta_summary is None:
                return output
            gate = torch.sigmoid(adapter.gate)

            if hasattr(output, "pooler_output"):
                pooled = output.pooler_output
                add = delta_summary.to(pooled.device, pooled.dtype)
                if add.shape[0] == 1 and pooled.shape[0] != 1:
                    add = add.expand(pooled.shape[0], -1)
                output.pooler_output = pooled + gate.to(pooled.device, pooled.dtype) * add[:, None, :]
                if getattr(output, "deepstack_features", None) is not None:
                    output.deepstack_features = [
                        adapter._add_delta(feat, delta_summary, gate)
                        for feat in output.deepstack_features
                    ]
                return output
            return adapter._add_delta(output, delta_summary, gate)

        target.get_image_features = MethodType(wrapped_get_image_features, target)
        self._state = AdapterState(target=target, original_get_image_features=original)

    def detach(self) -> None:
        if self._state is not None:
            self._state.target.get_image_features = self._state.original_get_image_features
            self._state = None
