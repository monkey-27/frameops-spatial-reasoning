from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class QwenBundle:
    model: Any
    processor: Any
    model_name: str


def assert_exact_qwen_available(model_name: str = QWEN3_VL_MODEL) -> None:
    if model_name != QWEN3_VL_MODEL:
        raise ValueError(
            f"FrameOps pilot must use exactly {QWEN3_VL_MODEL}; got {model_name}. "
            "Do not silently switch base VLMs."
        )
    try:
        from huggingface_hub import model_info

        model_info(model_name)
    except Exception as exc:  # pragma: no cover - network/model access dependent
        raise RuntimeError(
            f"Could not confirm availability of exact base VLM {model_name}. "
            "Stop and document this blocker instead of switching models."
        ) from exc


def load_qwen3_vl(
    model_name: str = QWEN3_VL_MODEL,
    *,
    dtype: str | torch.dtype = "auto",
    device_map: str | dict[str, int] | None = "auto",
    attn_implementation: str = "sdpa",
) -> QwenBundle:
    assert_exact_qwen_available(model_name)
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - depends on installed transformers
        raise RuntimeError(
            "Qwen3-VL requires a recent Transformers build exposing "
            "`Qwen3VLForConditionalGeneration`. Upgrade transformers rather than "
            "using trust_remote_code or switching models."
        ) from exc

    torch_dtype: str | torch.dtype
    if isinstance(dtype, str) and dtype != "auto":
        torch_dtype = getattr(torch, dtype)
    else:
        torch_dtype = dtype
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return QwenBundle(model=model, processor=processor, model_name=model_name)


def set_deterministic_generation(model: Any) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
