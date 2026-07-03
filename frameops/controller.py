from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ControllerOutput:
    answer_logits: torch.Tensor
    operator_logits: torch.Tensor
    object_logits: torch.Tensor
    frame_logits: torch.Tensor


class SimpleQuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)
        mask = (token_ids != self.pad_idx).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (emb * mask).sum(dim=1) / denom
        return self.proj(pooled)


class FrameOpsController(nn.Module):
    """Small v0 controller trained with synthetic program supervision.

    This is deliberately modest: it predicts the answer, operator class, object
    binding scores, and frame class from question tokens plus oracle/noisy slots.
    It is not meant to replace Qwen; it exists to exercise the operator interface
    and provide compact evidence for the VLM side.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        slot_dim: int,
        hidden_dim: int,
        num_operators: int,
        num_frames: int = 5,
        num_answers: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.question = SimpleQuestionEncoder(vocab_size, hidden_dim, pad_idx=pad_idx)
        self.slot_encoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.slot_context = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.answer_head = nn.Linear(hidden_dim, num_answers)
        self.operator_head = nn.Linear(hidden_dim, num_operators)
        self.object_head = nn.Linear(hidden_dim, 1)
        self.frame_head = nn.Linear(hidden_dim, num_frames)

    def forward(
        self,
        question_tokens: torch.Tensor,
        slot_features: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> ControllerOutput:
        question = self.question(question_tokens)
        slots = self.slot_encoder(slot_features)
        key_padding = ~slot_mask.bool()
        slots = self.slot_context(slots, src_key_padding_mask=key_padding)
        masked_slots = slots.masked_fill(~slot_mask.unsqueeze(-1).bool(), 0.0)
        denom = slot_mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked_slots.dtype)
        scene = masked_slots.sum(dim=1) / denom
        fused = self.fusion(torch.cat([question, scene], dim=-1))
        object_logits = self.object_head(slots).squeeze(-1).masked_fill(~slot_mask.bool(), -1e4)
        return ControllerOutput(
            answer_logits=self.answer_head(fused),
            operator_logits=self.operator_head(fused),
            object_logits=object_logits,
            frame_logits=self.frame_head(fused),
        )
