from __future__ import annotations

import argparse
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import ensure_out_dir, validate_diagnostic_split, write_json
from frameops.evidence import build_evidence
from frameops.progress import ProgressLogger
from frameops.utils import load_yaml, read_jsonl
from scripts.check_qwen_vl_identity import run as run_identity
from vlm.qwen_frameops_adapter import QwenFrameOpsConnectorAdapter
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation
from vlm.vl_trace import generate_with_trace


def _pad_features(features: torch.Tensor, dim: int = 16) -> torch.Tensor:
    flat = features.detach().float().flatten()
    if flat.numel() >= dim:
        return flat[:dim]
    return torch.cat([flat, torch.zeros(dim - flat.numel())], dim=0)


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:
        return f"<signature unavailable: {exc}>"


def run(config: str, out_dir: str | Path, model: str, max_new_tokens: int) -> Path:
    out = ensure_out_dir(out_dir)
    progress = ProgressLogger(out / "progress.jsonl")
    status: dict[str, Any] = {
        "requested_model_id": model,
        "exact_model_required": QWEN3_VL_MODEL,
        "model_identity_passed": False,
        "model_class": None,
        "forward_signature": None,
        "generate_signature": None,
        "get_image_features_signature": None,
        "inputs_embeds_supported": False,
        "image_tokens_integrated_before_or_after_embedding_lookup": "unknown",
        "can_inject_prefix_embeddings_while_preserving_image_inputs": False,
        "tested_in_forward_pass": False,
        "tested_in_generation": False,
        "minimal_hidden_evidence_run_completed": False,
        "minimal_reproduction_script_path": "scripts/debug_hidden_connector_qwen_vl.py",
        "stack_trace": None,
        "blockers": [],
        "recommended_fix": "",
    }
    identity = run_identity(model, out)
    status["model_identity_passed"] = bool(identity.get("pass"))
    if not identity.get("pass"):
        status["blockers"].append(f"identity_check_failed: {identity.get('failure_reason')}")
        status["recommended_fix"] = "Fix exact Qwen3-VL availability/image processor path before hidden connector work."
        write_json(out / "hidden_connector_debug.json", status)
        return out

    try:
        progress.log("hidden_debug", "started", f"loading {model}")
        bundle = load_qwen3_vl(model)
        set_deterministic_generation(bundle.model)
        status["model_class"] = type(bundle.model).__name__
        status["forward_signature"] = _signature(bundle.model.forward)
        status["generate_signature"] = _signature(bundle.model.generate)
        status["inputs_embeds_supported"] = "inputs_embeds" in status["forward_signature"]
        target = getattr(bundle.model, "model", bundle.model)
        if hasattr(target, "get_image_features"):
            status["get_image_features_signature"] = _signature(target.get_image_features)
            status["image_tokens_integrated_before_or_after_embedding_lookup"] = (
                "visual features are obtained through get_image_features and fused by the Qwen3-VL model "
                "after text tokenization/embedding lookup using image-related processor tensors"
            )
        else:
            status["blockers"].append("model target has no get_image_features hook")

        cfg = load_yaml(config)
        data_dir = out / "data"
        if not (data_dir / "diagnostic_test.jsonl").exists():
            generate_diagnostic_dataset(
                data_dir,
                seed=int(cfg.get("diagnostic_seed", 20260703)),
                image_size=int(cfg.get("synthetic", {}).get("image_size", 256)),
            )
        rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
        validate_diagnostic_split(rows, min_per_task=50)
        sample = rows[0]
        hidden_size = getattr(getattr(bundle.model.config, "text_config", bundle.model.config), "hidden_size", None)
        hidden_size = hidden_size or getattr(bundle.model.config, "hidden_size")
        adapter = QwenFrameOpsConnectorAdapter(evidence_dim=16, hidden_size=int(hidden_size))
        adapter.to(bundle.model.device)
        adapter.attach(bundle.model)
        try:
            evidence = build_evidence(sample)
            adapter.set_evidence(_pad_features(evidence.features).unsqueeze(0).to(bundle.model.device))
            raw, trace = generate_with_trace(
                bundle,
                sample["question"],
                image_path=sample["image_path"],
                require_image_input=True,
                max_new_tokens=max_new_tokens,
            )
            status["tested_in_forward_pass"] = True
            status["tested_in_generation"] = True
            status["minimal_hidden_evidence_run_completed"] = True
            status["minimal_hidden_raw_output"] = raw
            status["minimal_hidden_trace_keys"] = trace["processor_output_keys"]
            status["can_inject_prefix_embeddings_while_preserving_image_inputs"] = False
            status["recommended_fix"] = (
                "Connector-level hidden evidence can run through generation in this environment; next verify "
                "shape-preserving visual feature deltas on >=120 paired samples before treating it as evidence."
            )
        finally:
            adapter.detach()
    except Exception as exc:
        status["stack_trace"] = traceback.format_exc()
        status["blockers"].append(f"hidden_generation_failed: {exc}")
        status["recommended_fix"] = (
            "Do not call this hidden evidence yet. Inspect the real get_image_features return contract and "
            "preserve the exact Qwen3-VL container structure; the prior tuple/device failure suggests the "
            "adapter returned a tuple where downstream generation expected a tensor or model output field."
        )
        progress.log("hidden_debug", "failed", str(exc))
    write_json(out / "hidden_connector_debug.json", status)
    progress.log("hidden_debug", "completed", f"wrote {out / 'hidden_connector_debug.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug hidden evidence connector against real Qwen3-VL.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    run(args.config, args.out_dir, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
