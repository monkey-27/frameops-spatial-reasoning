from __future__ import annotations

import torch

from vlm.qwen_frameops_adapter import QwenFrameOpsConnectorAdapter


class FakeQwenVisual:
    def get_image_features(self, *args, **kwargs):
        return (
            torch.zeros(1, 3, 4),
            [torch.zeros(1, 3, 4), torch.zeros(1, 2, 4)],
        )


class FakeQwen:
    def __init__(self) -> None:
        self.model = FakeQwenVisual()


def test_connector_adapter_changes_tuple_visual_features() -> None:
    fake = FakeQwen()
    before = fake.model.get_image_features()
    adapter = QwenFrameOpsConnectorAdapter(evidence_dim=2, hidden_size=4, gate_init=4.0)
    adapter.attach(fake)
    adapter.set_evidence(torch.tensor([[1.0, -0.5]]))
    after = fake.model.get_image_features()
    assert torch.max(torch.abs(after[0] - before[0])) > 0
    assert torch.max(torch.abs(after[1][0] - before[1][0])) > 0
    adapter.detach()
    restored = fake.model.get_image_features()
    assert torch.allclose(restored[0], before[0])
