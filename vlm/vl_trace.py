from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from vlm.prompts import SYSTEM_PROMPT
from vlm.qwen_loader import QwenBundle

IMAGE_KEY_MARKERS = ("pixel", "image", "video", "grid", "vision")


def image_related_keys(keys: list[str] | tuple[str, ...]) -> list[str]:
    return [key for key in keys if any(marker in key.lower() for marker in IMAGE_KEY_MARKERS)]


def hash_image_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_input_device(model: Any) -> str:
    device = getattr(model, "device", None)
    if device is not None:
        return str(device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        if getattr(tensor, "device", None) is not None and str(tensor.device) != "meta":
            return str(tensor.device)
    return "cpu"


def tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "items": [tensor_summary(item) for item in value[:4]],
        }
    return {"kind": type(value).__name__, "repr": repr(value)[:160]}


def summarize_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {key: tensor_summary(value) for key, value in inputs.items()}


def _messages(prompt: str, image: Image.Image | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def prepare_qwen_vl_inputs(
    bundle: QwenBundle,
    prompt: str,
    *,
    image_path: str | Path | None = None,
    image: Image.Image | None = None,
    raw_image_path: str | Path | None = None,
    require_image_input: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    loaded_image = image
    image_loaded = False
    image_hash = None
    path_for_trace = raw_image_path or image_path
    if loaded_image is None and image_path is not None:
        loaded_image = Image.open(image_path).convert("RGB")
        image_loaded = True
        image_hash = hash_image_file(image_path)
    elif loaded_image is not None:
        image_loaded = True
        if image_path is not None and Path(image_path).exists():
            image_hash = hash_image_file(image_path)

    messages = _messages(prompt, loaded_image)
    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
    if loaded_image is not None:
        processor_kwargs["images"] = [loaded_image]
    processor_inputs = dict(bundle.processor(**processor_kwargs))
    processor_keys = list(processor_inputs.keys())
    image_keys = image_related_keys(processor_keys)
    if require_image_input and not image_keys:
        raise RuntimeError(
            "Qwen-VL image mode did not produce any image-related processor outputs. "
            f"Processor keys were: {processor_keys}"
        )

    target_device = torch.device(model_input_device(bundle.model))
    generation_inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in processor_inputs.items()
    }
    generation_image_keys = image_related_keys(list(generation_inputs.keys()))
    trace = {
        "prompt_text_length": len(prompt),
        "chat_template_length": len(text),
        "number_of_images": 1 if loaded_image is not None else 0,
        "raw_image_path": None if path_for_trace is None else str(path_for_trace),
        "image_loaded_successfully": image_loaded,
        "image_hash": image_hash,
        "processor_output_keys": processor_keys,
        "image_related_processor_keys": image_keys,
        "processor_inputs": summarize_inputs(processor_inputs),
        "generation_inputs": summarize_inputs(generation_inputs),
        "generation_received_image_related_tensors": bool(generation_image_keys),
        "generation_image_related_keys": generation_image_keys,
        "model_device": model_input_device(bundle.model),
        "model_class": type(bundle.model).__name__,
        "processor_class": type(bundle.processor).__name__,
    }
    if require_image_input and not trace["generation_received_image_related_tensors"]:
        raise RuntimeError("Generation inputs did not include image-related tensors in required image mode.")
    return generation_inputs, trace, text


def generate_with_trace(
    bundle: QwenBundle,
    prompt: str,
    *,
    image_path: str | Path | None = None,
    image: Image.Image | None = None,
    raw_image_path: str | Path | None = None,
    require_image_input: bool = False,
    max_new_tokens: int = 4,
) -> tuple[str, dict[str, Any]]:
    inputs, trace, _ = prepare_qwen_vl_inputs(
        bundle,
        prompt,
        image_path=image_path,
        image=image,
        raw_image_path=raw_image_path,
        require_image_input=require_image_input,
    )
    output_ids = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    raw = bundle.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    trace["generation_call_received_keys"] = list(inputs.keys())
    return raw, trace
