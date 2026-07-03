from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from vlm.prompts import SYSTEM_PROMPT, build_prompt
from vlm.qwen_loader import QwenBundle, set_deterministic_generation


@dataclass
class QwenPrediction:
    mode: str
    prediction: str
    raw_text: str


def _messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


class QwenBaselineRunner:
    def __init__(self, bundle: QwenBundle, *, max_new_tokens: int = 16):
        self.bundle = bundle
        self.max_new_tokens = max_new_tokens
        set_deterministic_generation(bundle.model)

    def predict(self, sample: dict, mode: str) -> QwenPrediction:
        image = Image.open(Path(sample["image_path"])).convert("RGB")
        prompt = build_prompt(sample, mode)
        messages = _messages(prompt, image)
        processor = self.bundle.processor
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.bundle.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        output_ids = self.bundle.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        norm = raw.lower().strip().split()[0].strip(".,:;") if raw.strip() else ""
        if norm not in {"yes", "no"}:
            norm = raw.strip()
        return QwenPrediction(mode=mode, prediction=norm, raw_text=raw)
