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

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import (
    DIAGNOSTIC_TASKS,
    ensure_out_dir,
    execute_with_program,
    load_controller,
    paired_delta_ci,
    predict_routing,
    program_for_operator,
    summarize_prediction_rows,
    validate_diagnostic_split,
    write_json,
)
from frameops.evidence import serialize_frameops_evidence_text, serialize_geometry_json, serialize_geometry_nl
from frameops.model import execute_program, frame_label_from_program, selected_object_ids, task_from_sample
from frameops.progress import ProgressLogger
from frameops.utils import load_yaml, read_jsonl, write_jsonl
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation
from vlm.vl_trace import generate_with_trace, hash_image_file

IMAGE_MODES = [
    "image_only",
    "image_plus_geometry_json",
    "image_plus_geometry_nl",
    "image_plus_oracle_frameops_text",
    "image_plus_learned_frameops_text",
]
TEXT_ONLY_MODES = ["text_only_geometry_json", "text_only_oracle_frameops_text"]
VALID_MODES = IMAGE_MODES + TEXT_ONLY_MODES


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
        raise ValueError("Qwen-VL diagnostic evidence requires n>=360.")
    per_task = n // len(DIAGNOSTIC_TASKS)
    selected: list[dict[str, Any]] = []
    for task in DIAGNOSTIC_TASKS:
        task_rows = [row for row in rows if task_from_sample(row) == task]
        if len(task_rows) < per_task:
            raise ValueError(f"Need {per_task} rows for {task}, found {len(task_rows)}")
        selected.extend(task_rows[:per_task])
    return sorted(selected, key=lambda row: row["id"])[:n]


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
        return f"{question}\nAnswer only yes or no.", meta
    if mode in {"image_plus_geometry_json", "text_only_geometry_json"}:
        return f"{question}\n\nOracle geometry JSON:\n{serialize_geometry_json(sample)}\nAnswer only yes or no.", meta
    if mode == "image_plus_geometry_nl":
        return f"{question}\n\nOracle geometry description:\n{serialize_geometry_nl(sample)}\nAnswer only yes or no.", meta
    if mode in {"image_plus_oracle_frameops_text", "text_only_oracle_frameops_text"}:
        evidence = serialize_frameops_evidence_text(sample, include_answer=False, label="Oracle FrameOps evidence")
        return f"{question}\n\n{evidence}\nUse the evidence but answer only yes or no.", meta
    if mode == "image_plus_learned_frameops_text":
        if controller is None or ckpt is None:
            raise RuntimeError("image_plus_learned_frameops_text requires a controller checkpoint")
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
    if mode == "oracle_frameops_answer_leakage":
        execution = execute_program(sample)
        evidence = serialize_frameops_evidence_text(sample, include_answer=True, label="Answer leakage FrameOps evidence")
        return f"{question}\n\n{evidence}\nParser sanity mode: answer only yes or no.", {"leakage_answer": execution.answer}
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
    raise ValueError(mode)


def _paired_examples(
    grouped: dict[str, list[dict[str, Any]]],
    baseline_mode: str,
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    base = {row["id"]: row for row in grouped[baseline_mode]}
    other = {row["id"]: row for row in grouped[mode]}
    categories = {
        "geometry_json_correct_mode_wrong": [],
        "mode_correct_geometry_json_wrong": [],
        "both_wrong": [],
        "both_correct": [],
    }
    for sample_id in sorted(set(base) & set(other)):
        base_ok = bool(base[sample_id]["correct"])
        mode_ok = bool(other[sample_id]["correct"])
        if base_ok and not mode_ok:
            key = "geometry_json_correct_mode_wrong"
        elif mode_ok and not base_ok:
            key = "mode_correct_geometry_json_wrong"
        elif not base_ok and not mode_ok:
            key = "both_wrong"
        else:
            key = "both_correct"
        if len(categories[key]) < 10:
            categories[key].append(
                {
                    "id": sample_id,
                    "task_family": base[sample_id]["task_family"],
                    "hardness": base[sample_id]["hardness"],
                    "gold": base[sample_id]["gold_answer"],
                    baseline_mode: base[sample_id]["parsed_answer"],
                    mode: other[sample_id]["parsed_answer"],
                }
            )
    return categories


def run(
    *,
    config: str,
    out_dir: str | Path,
    model: str,
    n: int,
    checkpoint: str | Path,
    require_image_input: bool,
    max_new_tokens: int,
) -> Path:
    out = ensure_out_dir(out_dir)
    progress = ProgressLogger(out / "progress.jsonl")
    cfg = load_yaml(config)
    data_dir = out / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        progress.log("qwen_vl_baselines", "started", "generating diagnostic split")
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
    controller_error = None
    try:
        controller, ckpt = load_controller(checkpoint)
    except Exception as exc:
        controller_error = str(exc)
        progress.log("qwen_vl_baselines", "warning", f"learned FrameOps disabled: {exc}")
    modes = list(VALID_MODES)
    if controller is None:
        modes.remove("image_plus_learned_frameops_text")

    progress.log("qwen_vl_baselines", "started", f"loading {model}")
    bundle = load_qwen3_vl(model)
    set_deterministic_generation(bundle.model)
    predictions: list[dict[str, Any]] = []
    for mode in modes:
        progress.log("qwen_vl_baselines", "running", f"mode {mode} started")
        for idx, sample in enumerate(rows):
            prompt, meta = _prompt_for_mode(sample, mode, controller=controller, ckpt=ckpt)
            is_image_mode = mode in IMAGE_MODES
            image_path = sample["image_path"] if is_image_mode else None
            raw, trace = generate_with_trace(
                bundle,
                prompt,
                image_path=image_path,
                require_image_input=require_image_input and is_image_mode,
                max_new_tokens=max_new_tokens,
            )
            parsed = _parse_answer(raw)
            predictions.append(
                {
                    "id": sample["id"],
                    "task_family": task_from_sample(sample),
                    "hardness": sample.get("metadata", {}).get("hardness", "unknown"),
                    "mode": mode,
                    "prompt": prompt,
                    "image_path": image_path,
                    "image_hash": hash_image_file(image_path) if image_path is not None else None,
                    "processor_input_keys": trace["processor_output_keys"],
                    "image_related_processor_keys": trace["image_related_processor_keys"],
                    "generation_received_image_related_tensors": trace[
                        "generation_received_image_related_tensors"
                    ],
                    "gold_answer": sample["answer"],
                    "raw_model_output": raw,
                    "parsed_answer": parsed,
                    "correct": parsed == str(sample["answer"]).lower(),
                    "parse_failed": parsed not in {"yes", "no"},
                    **meta,
                }
            )
            if (idx + 1) % 30 == 0 or idx + 1 == len(rows):
                progress.log("qwen_vl_baselines", "running", f"{mode}: {idx + 1}/{len(rows)}")
    pred_path = out / "qwen_vl_baselines_predictions.jsonl"
    write_jsonl(pred_path, predictions)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["mode"]].append(row)
    baseline_rows = grouped["image_plus_geometry_json"]
    summary = {
        "model": model,
        "n": len(rows),
        "require_image_input": require_image_input,
        "controller_checkpoint": str(checkpoint),
        "controller_error": controller_error,
        "modes": {},
        "paired_examples_vs_image_plus_geometry_json": {},
        "notes": (
            "Image modes pass a PIL image through the Qwen3-VL processor and require image-related tensors. "
            "Text-only modes are diagnostic controls, not method baselines."
        ),
    }
    for mode, mode_rows in sorted(grouped.items()):
        mode_summary = summarize_prediction_rows(
            mode_rows,
            baseline_rows=None if mode == "image_plus_geometry_json" else baseline_rows,
        )
        mode_summary["parse_failure_rate"] = sum(row["parse_failed"] for row in mode_rows) / max(1, len(mode_rows))
        mode_summary["all_generation_calls_received_image_tensors"] = all(
            row["generation_received_image_related_tensors"] for row in mode_rows
        )
        summary["modes"][mode] = mode_summary
        if mode != "image_plus_geometry_json":
            summary["paired_examples_vs_image_plus_geometry_json"][mode] = _paired_examples(
                grouped, "image_plus_geometry_json", mode
            )
    write_json(out / "qwen_vl_baselines.json", summary)
    progress.log("qwen_vl_baselines", "completed", f"wrote {out / 'qwen_vl_baselines.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Qwen3-VL image/text interface diagnostic.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--checkpoint", default="/vol/checkpoints/synthetic_oracle/frameops_controller.pt")
    parser.add_argument("--require-image-input", default="true")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    require_image = str(args.require_image_input).lower() in {"1", "true", "yes"}
    run(
        config=args.config,
        out_dir=args.out_dir,
        model=args.model,
        n=args.n,
        checkpoint=args.checkpoint,
        require_image_input=require_image,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
