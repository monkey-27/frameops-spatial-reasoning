from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from frameops.diagnostics import ensure_out_dir, write_json
from frameops.progress import ProgressLogger
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation
from vlm.vl_trace import generate_with_trace, image_related_keys


def _has_visual_attr(model: Any) -> bool:
    candidates = [model, getattr(model, "model", None)]
    for obj in candidates:
        if obj is None:
            continue
        for name in ("visual", "vision_model", "vision_tower", "image_encoder"):
            if hasattr(obj, name):
                return True
    return False


def _probe_image(path: Path) -> None:
    image = Image.new("RGB", (96, 96), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.rectangle([16, 18, 44, 74], fill=(220, 40, 40), outline=(20, 20, 20))
    draw.ellipse([54, 24, 82, 70], fill=(40, 95, 220), outline=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run(model: str, out_dir: str | Path) -> dict[str, Any]:
    out = ensure_out_dir(out_dir)
    progress = ProgressLogger(out / "progress.jsonl")
    status: dict[str, Any] = {
        "requested_model_id": model,
        "actual_model_id": None,
        "model_class": None,
        "processor_class": None,
        "config_model_type": None,
        "has_vision_config": False,
        "has_visual_encoder": False,
        "has_image_processor": False,
        "supports_pixel_values_or_image_grid": False,
        "is_text_only_qwen": True,
        "image_mode_processor_keys": [],
        "image_mode_image_related_keys": [],
        "image_mode_generation_received_image_tensors": False,
        "text_only_generation_available": False,
        "pass": False,
        "failure_reason": "",
    }
    try:
        progress.log("identity", "started", f"loading {model}")
        bundle = load_qwen3_vl(model)
        set_deterministic_generation(bundle.model)
        config = bundle.model.config
        model_type = str(getattr(config, "model_type", "unknown"))
        actual = str(getattr(config, "_name_or_path", bundle.model_name))
        has_vision_config = getattr(config, "vision_config", None) is not None
        has_visual_encoder = _has_visual_attr(bundle.model)
        has_image_processor = getattr(bundle.processor, "image_processor", None) is not None
        status.update(
            {
                "actual_model_id": actual,
                "model_class": type(bundle.model).__name__,
                "processor_class": type(bundle.processor).__name__,
                "config_model_type": model_type,
                "has_vision_config": bool(has_vision_config),
                "has_visual_encoder": bool(has_visual_encoder),
                "has_image_processor": bool(has_image_processor),
                "is_text_only_qwen": ("vl" not in model_type.lower()) and not bool(has_vision_config),
            }
        )
        if model != QWEN3_VL_MODEL:
            raise RuntimeError(f"requested model must be exactly {QWEN3_VL_MODEL}; got {model}")
        if status["is_text_only_qwen"]:
            raise RuntimeError(f"loaded model appears text-only: model_type={model_type}")
        if not has_image_processor:
            raise RuntimeError("processor has no image_processor")

        probe = out / "identity_probe.png"
        _probe_image(probe)
        raw, trace = generate_with_trace(
            bundle,
            "Answer yes or no: is there an image attached?",
            image_path=probe,
            require_image_input=True,
            max_new_tokens=2,
        )
        status["image_probe_raw_output"] = raw
        status["image_mode_processor_keys"] = trace["processor_output_keys"]
        status["image_mode_image_related_keys"] = trace["image_related_processor_keys"]
        status["image_mode_generation_received_image_tensors"] = bool(trace["generation_received_image_related_tensors"])
        status["supports_pixel_values_or_image_grid"] = bool(
            {"pixel_values", "image_grid_thw"} & set(trace["processor_output_keys"])
            or image_related_keys(trace["processor_output_keys"])
        )
        try:
            generate_with_trace(
                bundle,
                "Answer yes or no: is this a text-only control?",
                require_image_input=False,
                max_new_tokens=2,
            )
            status["text_only_generation_available"] = True
        except Exception as exc:
            status["text_only_generation_error"] = str(exc)
        if not status["image_mode_generation_received_image_tensors"]:
            raise RuntimeError("image mode generation received no image-related tensors")
        if not status["supports_pixel_values_or_image_grid"]:
            raise RuntimeError("image mode processor output had no pixel/image grid support")
        status["pass"] = True
        status["failure_reason"] = ""
        progress.log("identity", "completed", "Qwen3-VL identity and image tensors verified")
    except Exception as exc:
        status["pass"] = False
        status["failure_reason"] = str(exc)
        status["traceback"] = traceback.format_exc()
        progress.log("identity", "failed", str(exc))
    write_json(out / "model_identity.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-check Qwen3-VL model identity and image input path.")
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    result = run(args.model, args.out_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
