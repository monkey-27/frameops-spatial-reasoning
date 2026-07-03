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

import torch
from PIL import Image

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import DIAGNOSTIC_TASKS, ensure_out_dir, validate_diagnostic_split, write_json
from frameops.evidence import build_evidence, serialize_geometry_json
from frameops.diagnostics import mcnemar_counts, paired_delta_ci, summarize_prediction_rows
from frameops.utils import load_yaml, read_jsonl, write_jsonl
from vlm.prompts import SYSTEM_PROMPT
from vlm.qwen_frameops_adapter import QwenFrameOpsConnectorAdapter
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation


def _base_status() -> dict[str, Any]:
    status = {
        "can_access_inputs_embeds": False,
        "can_inject_prefix_embeddings": False,
        "can_access_visual_tokens": False,
        "can_modify_projector_or_connector": False,
        "tested_in_forward_pass": False,
        "tested_in_generation": False,
        "minimal_hidden_evidence_run_completed": False,
        "hidden_subset_n": 0,
        "hidden_vs_geometry_json": None,
        "blockers": [],
        "recommended_next_implementation": "",
    }
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401

        status["can_access_inputs_embeds"] = True
        status["can_inject_prefix_embeddings"] = False
        status["can_access_visual_tokens"] = True
        status["can_modify_projector_or_connector"] = True
        status["recommended_next_implementation"] = (
            "Continue with connector/projector-level adapter around get_image_features; "
            "avoid decoder prefix embeddings unless custom M-RoPE handling is added."
        )
    except Exception as exc:
        status["blockers"].append(f"transformers_qwen3_vl_import_failed: {exc}")
        status["recommended_next_implementation"] = "Install a Transformers build with Qwen3-VL support."
    return status


def _pad_features(features: torch.Tensor, dim: int = 16) -> torch.Tensor:
    flat = features.detach().float().flatten()
    if flat.numel() >= dim:
        return flat[:dim]
    return torch.cat([flat, torch.zeros(dim - flat.numel())], dim=0)


def _messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
    ]


def _parse_answer(text: str) -> str:
    stripped = text.strip().lower()
    if not stripped:
        return ""
    first = stripped.split()[0].strip(".,:;!?")
    if first in {"yes", "no"}:
        return first
    return first


def _generate(bundle: Any, sample: dict[str, Any], prompt: str | None = None, max_new_tokens: int = 4) -> str:
    image = Image.open(sample["image_path"]).convert("RGB")
    prompt = prompt or sample["question"]
    text = bundle.processor.apply_chat_template(
        _messages(prompt, image),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = bundle.processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(bundle.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    output = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[:, inputs["input_ids"].shape[1] :]
    return bundle.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_hidden_generation(status: dict[str, Any], *, out_dir: Path, config: str, n: int) -> dict[str, Any]:
    cfg = load_yaml(config)
    data_dir = out_dir / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        generate_diagnostic_dataset(
            data_dir,
            seed=int(cfg.get("diagnostic_seed", 20260703)),
            image_size=int(cfg.get("synthetic", {}).get("image_size", 256)),
        )
    rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
    validate_diagnostic_split(rows, min_per_task=50)
    per_task = max(1, n // len(DIAGNOSTIC_TASKS))
    subset = []
    for task in DIAGNOSTIC_TASKS:
        subset.extend([row for row in rows if row["metadata"]["task_type"] == task][:per_task])
    subset = subset[:n]
    bundle = load_qwen3_vl(QWEN3_VL_MODEL)
    set_deterministic_generation(bundle.model)
    hidden_size = getattr(getattr(bundle.model.config, "text_config", bundle.model.config), "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(bundle.model.config, "hidden_size")
    adapter = QwenFrameOpsConnectorAdapter(evidence_dim=16, hidden_size=int(hidden_size))
    adapter.to(bundle.model.device)
    adapter.attach(bundle.model)
    status["tested_in_forward_pass"] = True
    predictions = []
    try:
        for sample in subset:
            evidence = build_evidence(sample)
            adapter.set_evidence(_pad_features(evidence.features).unsqueeze(0).to(bundle.model.device))
            raw = _generate(bundle, sample)
            parsed = _parse_answer(raw)
            predictions.append(
                {
                    "id": sample["id"],
                    "mode": "hidden_connector_oracle_frameops",
                    "task_family": sample["metadata"]["task_type"],
                    "hardness": sample["metadata"].get("hardness", "unknown"),
                    "gold_answer": sample["answer"],
                    "prediction": raw,
                    "parsed_prediction": parsed,
                    "correct": parsed == sample["answer"],
                }
            )
            adapter.set_evidence(None)
            json_prompt = f"{sample['question']}\n\nOracle geometry JSON:\n{serialize_geometry_json(sample)}\nAnswer only yes or no."
            raw_json = _generate(bundle, sample, prompt=json_prompt)
            parsed_json = _parse_answer(raw_json)
            predictions.append(
                {
                    "id": sample["id"],
                    "mode": "geometry_json",
                    "task_family": sample["metadata"]["task_type"],
                    "hardness": sample["metadata"].get("hardness", "unknown"),
                    "gold_answer": sample["answer"],
                    "prediction": raw_json,
                    "parsed_prediction": parsed_json,
                    "correct": parsed_json == sample["answer"],
                }
            )
        status["tested_in_generation"] = True
        status["minimal_hidden_evidence_run_completed"] = bool(predictions)
        hidden_rows = [row for row in predictions if row["mode"] == "hidden_connector_oracle_frameops"]
        json_rows = [row for row in predictions if row["mode"] == "geometry_json"]
        status["hidden_subset_n"] = len(hidden_rows)
        status["hidden_accuracy"] = sum(row["correct"] for row in hidden_rows) / max(1, len(hidden_rows))
        status["hidden_vs_geometry_json"] = {
            "hidden": summarize_prediction_rows(hidden_rows),
            "geometry_json": summarize_prediction_rows(json_rows),
            "paired_delta_hidden_minus_geometry_json": paired_delta_ci(hidden_rows, json_rows),
            "mcnemar_counts": mcnemar_counts(hidden_rows, json_rows),
        }
        write_jsonl(out_dir / "hidden_connector_predictions.jsonl", predictions)
    finally:
        adapter.detach()
    return status


def run(config: str, out_dir: str | Path, n: int) -> Path:
    out = ensure_out_dir(out_dir)
    status = _base_status()
    if n > 0 and status["can_modify_projector_or_connector"]:
        try:
            status = run_hidden_generation(status, out_dir=out, config=config, n=n)
        except Exception as exc:
            status["blockers"].append(f"hidden_generation_failed: {exc}")
            status["blocker_traceback"] = traceback.format_exc(limit=8)
            status["minimal_hidden_evidence_run_completed"] = False
    if status["minimal_hidden_evidence_run_completed"] and status["hidden_subset_n"] < 120:
        status["blockers"].append("hidden_generation_only_tiny_subset_not_method_evidence")
    write_json(out / "hidden_connector_status.json", status)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and optionally exercise Qwen3-VL hidden evidence connector.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()
    out = run(args.config, args.out_dir, args.n)
    print(f"wrote hidden connector status to {out / 'hidden_connector_status.json'}")


if __name__ == "__main__":
    main()
