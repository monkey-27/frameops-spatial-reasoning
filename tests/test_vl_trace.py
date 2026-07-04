from __future__ import annotations

import torch

from frameops.evidence import serialize_frameops_evidence_text
from vlm.vl_trace import image_related_keys, tensor_summary


def test_image_related_keys_detect_qwen_vl_inputs() -> None:
    keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
    assert image_related_keys(keys) == ["pixel_values", "image_grid_thw"]


def test_tensor_summary_handles_nested_qwen_shapes() -> None:
    summary = tensor_summary((torch.zeros(2, 3), [torch.ones(1, 4)]))
    assert summary["kind"] == "tuple"
    assert summary["items"][0]["shape"] == [2, 3]
    assert summary["items"][1]["items"][0]["shape"] == [1, 4]


def test_oracle_frameops_evidence_text_does_not_leak_answer() -> None:
    sample = {
        "question": "Is obj_0 left of obj_1?",
        "answer": "yes",
        "objects": [
            {"id": "obj_0", "label": "red_cube", "center": [0.0, 3.0, 1.0], "extent": [1, 1, 1], "yaw": 0.0},
            {"id": "obj_1", "label": "blue_cube", "center": [1.0, 3.0, 1.0], "extent": [1, 1, 1], "yaw": 0.0},
        ],
        "camera": {"position": [0.0, 0.0, 1.2], "yaw": 0.0},
        "program": [{"op": "compare_axis", "frame": "camera", "a": "obj_1", "b": "obj_0", "axis": "x", "relation": "left"}],
    }
    text = serialize_frameops_evidence_text(sample, include_answer=False)
    assert "predicted_answer" not in text
    assert "therefore" not in text.lower()
