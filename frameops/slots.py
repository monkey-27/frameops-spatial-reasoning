from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import torch

from frameops.utils import ensure_tensor


@dataclass
class SpatialSlot:
    object_id: str
    semantic_label: str
    center: torch.Tensor
    extent: torch.Tensor
    yaw: torch.Tensor
    confidence: torch.Tensor
    semantic: torch.Tensor | None = None
    bbox: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.center = ensure_tensor(self.center).reshape(3)
        self.extent = ensure_tensor(self.extent).reshape(3).clamp_min(1e-4)
        self.yaw = ensure_tensor(self.yaw).reshape(())
        self.confidence = ensure_tensor(self.confidence).reshape(())
        if self.semantic is not None:
            self.semantic = ensure_tensor(self.semantic)
        if self.bbox is not None:
            self.bbox = ensure_tensor(self.bbox).reshape(4)
        if self.mask is not None and not isinstance(self.mask, torch.Tensor):
            self.mask = torch.as_tensor(self.mask)

    @classmethod
    def from_object_dict(cls, obj: dict[str, Any]) -> "SpatialSlot":
        return cls(
            object_id=str(obj["id"]),
            semantic_label=str(obj.get("label", obj.get("semantic_label", obj["id"]))),
            center=obj["center"],
            extent=obj["extent"],
            yaw=obj.get("yaw", 0.0),
            confidence=obj.get("confidence", 1.0),
            bbox=obj.get("bbox"),
            metadata=obj,
        )

    def to_feature(self, label_vocab: dict[str, int] | None = None) -> torch.Tensor:
        pieces = [self.center, self.extent, self.yaw[None], self.confidence[None]]
        if label_vocab is not None:
            one_hot = torch.zeros(len(label_vocab), dtype=self.center.dtype, device=self.center.device)
            if self.semantic_label in label_vocab:
                one_hot[label_vocab[self.semantic_label]] = 1.0
            pieces.append(one_hot)
        elif self.semantic is not None:
            pieces.append(self.semantic.to(device=self.center.device, dtype=self.center.dtype).flatten())
        return torch.cat(pieces)

    def with_noise(
        self,
        *,
        center_std: float = 0.0,
        extent_std: float = 0.0,
        yaw_std: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> "SpatialSlot":
        center = self.center + torch.randn(self.center.shape, generator=generator) * center_std
        extent_noise = torch.randn(self.extent.shape, generator=generator) * extent_std
        extent = (self.extent + extent_noise).clamp_min(1e-3)
        yaw = self.yaw + torch.randn((), generator=generator) * yaw_std
        return replace(self, center=center, extent=extent, yaw=yaw)

    def to(self, device: torch.device | str) -> "SpatialSlot":
        return replace(
            self,
            center=self.center.to(device),
            extent=self.extent.to(device),
            yaw=self.yaw.to(device),
            confidence=self.confidence.to(device),
            semantic=None if self.semantic is None else self.semantic.to(device),
            bbox=None if self.bbox is None else self.bbox.to(device),
            mask=None if self.mask is None else self.mask.to(device),
        )


class SpatialBatch:
    def __init__(self, slots: Iterable[SpatialSlot]):
        self.slots = list(slots)
        if not self.slots:
            raise ValueError("SpatialBatch requires at least one slot")

    @classmethod
    def from_objects(cls, objects: Iterable[dict[str, Any]]) -> "SpatialBatch":
        return cls(SpatialSlot.from_object_dict(obj) for obj in objects)

    def __len__(self) -> int:
        return len(self.slots)

    def __iter__(self):
        return iter(self.slots)

    def __getitem__(self, idx: int) -> SpatialSlot:
        return self.slots[idx]

    @property
    def centers(self) -> torch.Tensor:
        return torch.stack([slot.center for slot in self.slots], dim=0)

    @property
    def extents(self) -> torch.Tensor:
        return torch.stack([slot.extent for slot in self.slots], dim=0)

    @property
    def yaws(self) -> torch.Tensor:
        return torch.stack([slot.yaw for slot in self.slots], dim=0)

    @property
    def confidences(self) -> torch.Tensor:
        return torch.stack([slot.confidence for slot in self.slots], dim=0)

    @property
    def object_ids(self) -> list[str]:
        return [slot.object_id for slot in self.slots]

    @property
    def labels(self) -> list[str]:
        return [slot.semantic_label for slot in self.slots]

    def add_noise(
        self,
        *,
        center_std: float = 0.0,
        extent_std: float = 0.0,
        yaw_std: float = 0.0,
        seed: int | None = None,
    ) -> "SpatialBatch":
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return SpatialBatch(
            slot.with_noise(
                center_std=center_std,
                extent_std=extent_std,
                yaw_std=yaw_std,
                generator=generator,
            )
            for slot in self.slots
        )

    def to_feature_matrix(self, label_vocab: dict[str, int] | None = None) -> torch.Tensor:
        return torch.stack([slot.to_feature(label_vocab) for slot in self.slots], dim=0)

    def by_id(self, object_id: str) -> SpatialSlot:
        for slot in self.slots:
            if slot.object_id == object_id:
                return slot
        raise KeyError(object_id)

    def to(self, device: torch.device | str) -> "SpatialBatch":
        return SpatialBatch(slot.to(device) for slot in self.slots)
