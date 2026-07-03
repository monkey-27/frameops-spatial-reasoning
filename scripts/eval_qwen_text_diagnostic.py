from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import (
    DIAGNOSTIC_TASKS,
    ensure_out_dir,
    execute_with_program,
    load_controller,
    mcnemar_counts,
    paired_delta_ci,
    predict_routing,
    program_for_operator,
    summarize_prediction_rows,
    validate_diagnostic_split,
    write_json,
)
from frameops.evidence import (
    serialize_frameops_evidence_text,
    serialize_geometry_json,
    serialize_geometry_nl,
)
from frameops.model import execute_program, frame_label_from_program, selected_object_ids, task_from_sample
from frameops.utils import load_yaml, read_jsonl, write_jsonl
from vlm.prompts import SYSTEM_PROMPT
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation

VALID_MODES = [
    "image_only",
    "geometry_json",
    "geometry_natural_language",
    "oracle_frameops_text",
    "learned_frameops_text",
    "gold_program_trace_text",
]


def _parse_answer(text: str) -> str:
    stripped = text.strip().lower()
    if not stripped:
        return ""
    first = stripped.split()[0].strip(".,:;!?")
    if first in {"yes", "no"}:
        return first
    if "yes" in stripped[:24] and "no" not in stripped[:24]:
        return "yes"
    if "no" in stripped[:24] and "yes" not in stripped[:24]:
        return "no"
    return first


def _balanced_subset(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    if n < 360:
        raise ValueError("Qwen diagnostic evidence requires n>=360. Use smoke scripts for tiny checks.")
    per_task = n // len(DIAGNOSTIC_TASKS)
    selected = []
    for task in DIAGNOSTIC_TASKS:
        task_rows = [row for row in rows if task_from_sample(row) == task]
        if len(task_rows) < per_task:
            raise ValueError(f"Need {per_task} rows for {task}, found {len(task_rows)}")
        selected.extend(task_rows[:per_task])
    return sorted(selected, key=lambda row: row["id"])


def _prompt_for_mode(
    sample: dict[str, Any],
    mode: str,
    *,
    controller: Any | None = None,
    ckpt: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    question = sample["question"]
    meta: dict[str, Any] = {}
    if mode == "image_only":
        return question, meta
    if mode == "geometry_json":
        return f"{question}\n\nOracle geometry JSON:\n{serialize_geometry_json(sample)}", meta
    if mode == "geometry_natural_language":
        return f"{question}\n\nOracle geometry description:\n{serialize_geometry_nl(sample)}", meta
    if mode == "oracle_frameops_text":
        evidence = serialize_frameops_evidence_text(sample, include_answer=False, label="Oracle FrameOps evidence")
        return f"{question}\n\n{evidence}\nUse the evidence but answer only yes or no.", meta
    if mode == "gold_program_trace_text":
        execution = execute_program(sample)
        payload = {
            "gold_program": sample["program"],
            "selected_objects": sorted(selected_object_ids(sample)),
            "selected_frame": frame_label_from_program(sample),
            "trace": execution.trace,
            "leakage_note": "gold program trace upper-bound diagnostic; no direct answer string included",
        }
        return f"{question}\n\nGold program trace diagnostic:\n{json.dumps(payload, sort_keys=True)}\nAnswer only yes or no.", meta
    if mode == "oracle_frameops_answer_leakage":
        execution = execute_program(sample)
        evidence = serialize_frameops_evidence_text(sample, include_answer=True, label="Answer leakage FrameOps evidence")
        return f"{question}\n\n{evidence}\nParser sanity mode: answer only yes or no.", {"leakage_answer": execution.answer}
    if mode == "learned_frameops_text":
        if controller is None or ckpt is None:
            raise RuntimeError("learned_frameops_text requires a controller checkpoint")
        routing = predict_routing(sample, controller, ckpt)
        program = program_for_operator(sample, routing.predicted_operator, routing.predicted_objects, routing.predicted_frame)
        execution = execute_with_program(sample, program)
        payload = {
            "selected_objects": routing.predicted_objects,
            "selected_frame": routing.predicted_frame,
            "selected_operator": routing.predicted_operator,
            "confidence": routing.answer_confidence,
            "trace": execution["trace"],
            "note": "learned FrameOps evidence; no direct answer string included",
        }
        meta = {
            "selected_objects": routing.predicted_objects,
            "selected_frame": routing.predicted_frame,
            "selected_operator": routing.predicted_operator,
        }
        return f"{question}\n\nLearned FrameOps evidence:\n{json.dumps(payload, sort_keys=True)}\nAnswer only yes or no.", meta
    raise ValueError(mode)


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


def _predict(bundle: Any, sample: dict[str, Any], prompt: str, max_new_tokens: int) -> str:
    image = Image.open(Path(sample["image_path"])).convert("RGB")
    messages = _messages(prompt, image)
    processor = bundle.processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(bundle.model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    output_ids = bundle.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run(
    *,
    config: str,
    out_dir: str | Path,
    n: int,
    checkpoint: str | Path,
    include_leakage: bool,
    max_new_tokens: int,
) -> Path:
    cfg = load_yaml(config)
    out = ensure_out_dir(out_dir)
    data_dir = out / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        generate_diagnostic_dataset(
            data_dir,
            seed=int(cfg.get("diagnostic_seed", 20260703)),
            image_size=int(cfg.get("synthetic", {}).get("image_size", 256)),
        )
    rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
    validate_diagnostic_split(rows, min_per_task=50)
    rows = _balanced_subset(rows, n)

    controller = None
    ckpt = None
    if Path(checkpoint).exists() or str(checkpoint).startswith("/vol/"):
        try:
            controller, ckpt = load_controller(checkpoint)
        except Exception as exc:
            print(f"warning: could not load controller for learned_frameops_text: {exc}")

    modes = list(VALID_MODES)
    if include_leakage:
        modes.append("oracle_frameops_answer_leakage")
    if controller is None:
        modes = [mode for mode in modes if mode != "learned_frameops_text"]

    bundle = load_qwen3_vl(QWEN3_VL_MODEL)
    set_deterministic_generation(bundle.model)
    predictions: list[dict[str, Any]] = []
    for mode in modes:
        for sample in rows:
            prompt, meta = _prompt_for_mode(sample, mode, controller=controller, ckpt=ckpt)
            raw = _predict(bundle, sample, prompt, max_new_tokens=max_new_tokens)
            parsed = _parse_answer(raw)
            predictions.append(
                {
                    "id": sample["id"],
                    "mode": mode,
                    "task_family": task_from_sample(sample),
                    "hardness": sample.get("metadata", {}).get("hardness", "unknown"),
                    "gold_answer": sample["answer"],
                    "prediction": raw,
                    "parsed_prediction": parsed,
                    "correct": parsed == str(sample["answer"]).lower(),
                    "parse_failed": parsed not in {"yes", "no"},
                    **meta,
                }
            )
        print(mode, summarize_prediction_rows([row for row in predictions if row["mode"] == mode])["overall"])

    pred_path = out / "qwen_text_baselines_predictions.jsonl"
    write_jsonl(pred_path, predictions)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["mode"]].append(row)
    geometry = grouped["geometry_json"]
    summary = {
        "model": QWEN3_VL_MODEL,
        "n": len(rows),
        "modes": {},
        "paired_examples": {},
        "notes": "All modes use the same paired diagnostic samples and oracle geometry where applicable.",
    }
    for mode, mode_rows in sorted(grouped.items()):
        mode_summary = summarize_prediction_rows(mode_rows, baseline_rows=None if mode == "geometry_json" else geometry)
        mode_summary["parse_failure_rate"] = sum(row["parse_failed"] for row in mode_rows) / max(1, len(mode_rows))
        summary["modes"][mode] = mode_summary
    for mode in [m for m in grouped if m != "geometry_json"]:
        g = {row["id"]: row for row in geometry}
        m = {row["id"]: row for row in grouped[mode]}
        categories = {"geometry_json_correct_frameops_wrong": [], "frameops_correct_geometry_json_wrong": [], "both_wrong": [], "both_correct": []}
        for sample_id in sorted(set(g) & set(m)):
            g_ok = bool(g[sample_id]["correct"])
            m_ok = bool(m[sample_id]["correct"])
            if g_ok and not m_ok:
                key = "geometry_json_correct_frameops_wrong"
            elif m_ok and not g_ok:
                key = "frameops_correct_geometry_json_wrong"
            elif not g_ok and not m_ok:
                key = "both_wrong"
            else:
                key = "both_correct"
            if len(categories[key]) < 10:
                categories[key].append({"id": sample_id, "task_family": g[sample_id]["task_family"], "gold": g[sample_id]["gold_answer"], "geometry_json": g[sample_id]["parsed_prediction"], mode: m[sample_id]["parsed_prediction"]})
        summary["paired_examples"][mode] = categories
    write_json(out / "qwen_text_baselines.json", summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL paired text-interface diagnostic.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--checkpoint", default="/vol/checkpoints/synthetic_oracle/frameops_controller.pt")
    parser.add_argument("--include-leakage", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    out = run(
        config=args.config,
        out_dir=args.out_dir,
        n=args.n,
        checkpoint=args.checkpoint,
        include_leakage=args.include_leakage,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"wrote qwen text diagnostic to {out}")


if __name__ == "__main__":
    main()
