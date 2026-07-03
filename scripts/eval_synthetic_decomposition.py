from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import (
    DIAGNOSTIC_TASKS,
    NOISE_LEVELS,
    canonical_operator_label,
    diagnostic_timestamp,
    ensure_out_dir,
    execute_with_program,
    load_controller,
    paired_delta_ci,
    predict_routing,
    program_for_operator,
    replace_program_frame,
    replace_program_objects,
    summarize_prediction_rows,
    validate_diagnostic_split,
    write_json,
)
from frameops.model import execute_program, frame_label_from_program, selected_object_ids, task_from_sample
from frameops.slots import SpatialBatch
from frameops.utils import load_yaml, read_jsonl, write_jsonl
from scripts.train_frameops import train_from_config


def _sample_with_slots(sample: dict[str, Any], slots: SpatialBatch) -> dict[str, Any]:
    patched = dict(sample)
    patched["objects"] = []
    original = {obj["id"]: obj for obj in sample["objects"]}
    for slot in slots:
        obj = dict(original[slot.object_id])
        obj["center"] = [float(x) for x in slot.center]
        obj["extent"] = [float(x) for x in slot.extent]
        obj["yaw"] = float(slot.yaw)
        patched["objects"].append(obj)
    return patched


def _prediction_row(
    *,
    sample: dict[str, Any],
    mode: str,
    prediction: str,
    logit: float | None,
    trace: list[dict[str, Any]] | None,
    routing: Any | None,
    noise: float | None = None,
) -> dict[str, Any]:
    task = task_from_sample(sample)
    parsed = str(prediction).strip().lower()
    return {
        "id": sample["id"],
        "mode": mode,
        "task_family": task,
        "hardness": sample.get("metadata", {}).get("hardness", sample.get("metadata", {}).get("difficulty", "unknown")),
        "noise": noise,
        "gold_answer": sample["answer"],
        "prediction": prediction,
        "parsed_prediction": parsed,
        "correct": parsed == str(sample["answer"]).lower(),
        "selected_objects": None if routing is None else routing.predicted_objects,
        "selected_frame": None if routing is None else routing.predicted_frame,
        "selected_operator": None if routing is None else routing.predicted_operator,
        "gold_objects": sorted(selected_object_ids(sample)),
        "gold_frame": frame_label_from_program(sample),
        "gold_operator": canonical_operator_label(task),
        "logit": logit,
        "trace": trace or [],
    }


def _majority_by_task(train_rows: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    overall = Counter(str(row["answer"]).lower() for row in train_rows).most_common(1)[0][0]
    by_task = {}
    for task in DIAGNOSTIC_TASKS:
        rows = [row for row in train_rows if task_from_sample(row) == task]
        if rows:
            by_task[task] = Counter(str(row["answer"]).lower() for row in rows).most_common(1)[0][0]
        else:
            by_task[task] = overall
    return overall, by_task


def _confusion_add(matrix: dict[str, dict[str, int]], gold: str, pred: str) -> None:
    matrix.setdefault(gold, {})
    matrix[gold][pred] = matrix[gold].get(pred, 0) + 1


def _conditioned(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    selected = [row for row in rows if row.get(key)]
    if not selected:
        return {"n": 0, "accuracy": 0.0}
    return {"n": len(selected), "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected)}


def evaluate_checkpoint(
    *,
    checkpoint: str | Path,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    out_dir: Path,
    prefix: str = "existing",
    write_predictions: bool = True,
) -> dict[str, Any]:
    model, ckpt = load_controller(checkpoint)
    majority, by_task_majority = _majority_by_task(train_rows)
    rng = random.Random(123)
    predictions: list[dict[str, Any]] = []
    controller_rows: list[dict[str, Any]] = []
    frame_confusion: dict[str, dict[str, int]] = {}
    operator_confusion: dict[str, dict[str, int]] = {}
    task_failures: dict[str, Counter] = defaultdict(Counter)

    for idx, sample in enumerate(test_rows):
        slots = SpatialBatch.from_objects(sample["objects"])
        routing = predict_routing(sample, model, ckpt)
        gold_objects = []
        for step in sample["program"]:
            for key in ("a", "b", "c", "ref", "slot", "target", "viewer"):
                val = step.get(key)
                if isinstance(val, str) and val.startswith("obj_") and val not in gold_objects:
                    gold_objects.append(val)
        task = task_from_sample(sample)
        _confusion_add(frame_confusion, frame_label_from_program(sample), routing.predicted_frame)
        _confusion_add(operator_confusion, canonical_operator_label(task), routing.predicted_operator)

        oracle = execute_program(sample, slots=slots)
        predictions.append(
            _prediction_row(
                sample=sample,
                mode=f"{prefix}:oracle_program_oracle_slots",
                prediction=oracle.answer,
                logit=oracle.logit,
                trace=oracle.trace,
                routing=None,
                noise=0.0,
            )
        )
        for noise in NOISE_LEVELS:
            noisy = slots.add_noise(center_std=noise, extent_std=noise * 0.5, yaw_std=noise, seed=idx)
            pred = execute_program(sample, slots=noisy)
            predictions.append(
                _prediction_row(
                    sample=sample,
                    mode=f"{prefix}:oracle_program_noisy_slots",
                    prediction=pred.answer,
                    logit=pred.logit,
                    trace=pred.trace,
                    routing=None,
                    noise=noise,
                )
            )

        pred_program = program_for_operator(sample, routing.predicted_operator, gold_objects, routing.predicted_frame)
        pred_program = replace_program_frame(pred_program, routing.predicted_frame, gold_objects[0] if gold_objects else None)
        mixed = execute_with_program(sample, pred_program, slots=slots)
        predictions.append(
            _prediction_row(
                sample=sample,
                mode=f"{prefix}:oracle_objects_learned_frame_operator",
                prediction=mixed["answer"],
                logit=mixed["logit"],
                trace=mixed["trace"],
                routing=routing,
                noise=0.0,
            )
        )

        obj_program = replace_program_objects(sample, routing.predicted_objects)
        mixed = execute_with_program(sample, obj_program, slots=slots)
        predictions.append(
            _prediction_row(
                sample=sample,
                mode=f"{prefix}:oracle_frame_operator_learned_objects",
                prediction=mixed["answer"],
                logit=mixed["logit"],
                trace=mixed["trace"],
                routing=routing,
                noise=0.0,
            )
        )

        obj_frame_program = replace_program_frame(obj_program, routing.predicted_frame, routing.predicted_objects[0] if routing.predicted_objects else None)
        mixed = execute_with_program(sample, obj_frame_program, slots=slots)
        predictions.append(
            _prediction_row(
                sample=sample,
                mode=f"{prefix}:oracle_operator_learned_objects_frame",
                prediction=mixed["answer"],
                logit=mixed["logit"],
                trace=mixed["trace"],
                routing=routing,
                noise=0.0,
            )
        )

        learned = _prediction_row(
            sample=sample,
            mode=f"{prefix}:learned_controller_oracle_slots",
            prediction=routing.answer,
            logit=None,
            trace=[],
            routing=routing,
            noise=0.0,
        )
        learned["object_top1_hit"] = routing.object_top1_hit
        learned["object_top3_all"] = routing.object_top3_all
        learned["frame_correct"] = routing.frame_correct
        learned["operator_correct"] = routing.operator_correct
        learned["all_routing_correct"] = routing.all_routing_correct
        predictions.append(learned)
        controller_rows.append(learned)

        for noise in NOISE_LEVELS:
            noisy = slots.add_noise(center_std=noise, extent_std=noise * 0.5, yaw_std=noise, seed=1000 + idx)
            noisy_routing = predict_routing(_sample_with_slots(sample, noisy), model, ckpt)
            predictions.append(
                _prediction_row(
                    sample=sample,
                    mode=f"{prefix}:learned_controller_noisy_slots",
                    prediction=noisy_routing.answer,
                    logit=None,
                    trace=[],
                    routing=noisy_routing,
                    noise=noise,
                )
            )

        for mode, pred in [
            (f"{prefix}:majority_answer", majority),
            (f"{prefix}:random_answer_by_task", rng.choice(["yes", "no"])),
            (f"{prefix}:program_prior_only", by_task_majority[task]),
        ]:
            predictions.append(
                _prediction_row(
                    sample=sample,
                    mode=mode,
                    prediction=pred,
                    logit=None,
                    trace=[],
                    routing=None,
                    noise=0.0,
                )
            )

        if not routing.object_top3_all:
            task_failures[task]["object_binding"] += 1
        if not routing.frame_correct:
            task_failures[task]["frame_selection"] += 1
        if not routing.operator_correct:
            task_failures[task]["operator_selection"] += 1
        if routing.object_top3_all and routing.frame_correct and routing.operator_correct and not learned["correct"]:
            task_failures[task]["answer_head"] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["mode"]].append(row)
    baseline = grouped[f"{prefix}:learned_controller_oracle_slots"]
    summary = {
        "checkpoint": str(checkpoint),
        "modes": {
            mode: summarize_prediction_rows(rows, baseline_rows=None if mode == f"{prefix}:learned_controller_oracle_slots" else baseline)
            for mode, rows in sorted(grouped.items())
        },
        "controller_diagnostics": {
            "object_binding_top1_accuracy": sum(row.get("object_top1_hit", False) for row in controller_rows) / max(1, len(controller_rows)),
            "object_binding_top3_all_accuracy": sum(row.get("object_top3_all", False) for row in controller_rows) / max(1, len(controller_rows)),
            "frame_selection_accuracy": sum(row.get("frame_correct", False) for row in controller_rows) / max(1, len(controller_rows)),
            "operator_selection_accuracy": sum(row.get("operator_correct", False) for row in controller_rows) / max(1, len(controller_rows)),
            "answer_accuracy_conditioned_on_object_binding_correct": _conditioned(controller_rows, "object_top3_all"),
            "answer_accuracy_conditioned_on_frame_correct": _conditioned(controller_rows, "frame_correct"),
            "answer_accuracy_conditioned_on_operator_correct": _conditioned(controller_rows, "operator_correct"),
            "answer_accuracy_conditioned_on_all_routing_correct": _conditioned(controller_rows, "all_routing_correct"),
            "frame_confusion_matrix": frame_confusion,
            "operator_confusion_matrix": operator_confusion,
            "per_task_dominant_failure_mode": {
                task: (counts.most_common(1)[0][0] if counts else "none")
                for task, counts in sorted(task_failures.items())
            },
            "counterfactual_parameter_error": "not_predicted_in_v0_controller",
        },
    }
    if write_predictions:
        pred_path = out_dir / "synthetic_decomposition_predictions.jsonl"
        write_jsonl(pred_path, predictions)
    write_json(out_dir / f"controller_{prefix}.json", summary)
    return summary


def maybe_retrain(
    *,
    data_dir: Path,
    out_dir: Path,
    existing_summary: dict[str, Any],
    max_epochs: int,
) -> dict[str, Any]:
    learned_key = "existing:learned_controller_oracle_slots"
    existing_acc = existing_summary["modes"][learned_key]["overall"]["mean"]
    min_task = min(
        bucket["mean"]
        for bucket in existing_summary["modes"][learned_key]["by_task_family"].values()
    )
    if existing_acc >= 0.70 and min_task >= 0.65:
        result = {"retrained": False, "reason": "existing_controller_not_clearly_undertrained"}
        write_json(out_dir / "controller_retrained_summary.json", result)
        return result

    summaries = []
    checkpoints = []
    for seed in [0, 1, 2]:
        ckpt_dir = out_dir / f"checkpoints/retrained_seed{seed}"
        config = {
            "seed": seed,
            "output_dir": str(data_dir),
            "training": {
                "batch_size": 32,
                "epochs": max_epochs,
                "lr": 0.001,
                "hidden_dim": 128,
                "checkpoint_dir": str(ckpt_dir),
                "freeze_qwen": True,
                "wandb": False,
            },
        }
        config_path = out_dir / f"retrain_seed{seed}.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        checkpoint = train_from_config(config_path, epochs=max_epochs)
        checkpoints.append(str(checkpoint))
        train_rows = read_jsonl(data_dir / "train.jsonl")
        test_rows = read_jsonl(data_dir / "test.jsonl")
        summary = evaluate_checkpoint(
            checkpoint=checkpoint,
            train_rows=train_rows,
            test_rows=test_rows,
            out_dir=out_dir,
            prefix=f"retrained_seed{seed}",
            write_predictions=False,
        )
        write_json(out_dir / f"controller_retrained_seed{seed}.json", summary)
        summaries.append(summary)
    accs = [
        summary["modes"][f"retrained_seed{i}:learned_controller_oracle_slots"]["overall"]["mean"]
        for i, summary in enumerate(summaries)
    ]
    result = {
        "retrained": True,
        "reason": "existing_controller_below_diagnostic_threshold_or_task_floor",
        "seeds": [0, 1, 2],
        "answer_accuracy_mean": sum(accs) / len(accs),
        "answer_accuracy_by_seed": {str(i): acc for i, acc in enumerate(accs)},
        "best_seed": int(max(range(len(accs)), key=lambda idx: accs[idx])),
        "best_checkpoint": checkpoints[int(max(range(len(accs)), key=lambda idx: accs[idx]))],
    }
    write_json(out_dir / "controller_retrained_summary.json", result)
    return result


def run(config: str, checkpoint: str, out_dir: str | Path | None, retrain: bool, retrain_epochs: int) -> Path:
    config_data = load_yaml(config)
    out = ensure_out_dir(out_dir or (Path("results") / f"diagnostic_{diagnostic_timestamp()}"))
    data_dir = out / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        generate_diagnostic_dataset(
            data_dir,
            seed=int(config_data.get("diagnostic_seed", 20260703)),
            image_size=int(config_data.get("synthetic", {}).get("image_size", 256)),
            test_per_task=int(config_data.get("diagnostic", {}).get("test_per_task", 50)),
            train_per_task=int(config_data.get("diagnostic", {}).get("train_per_task", 100)),
            val_per_task=int(config_data.get("diagnostic", {}).get("val_per_task", 20)),
        )
    train_rows = read_jsonl(data_dir / "diagnostic_train.jsonl")
    test_rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
    validation = validate_diagnostic_split(test_rows, min_per_task=50)
    manifest = {
        "out_dir": str(out),
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "diagnostic_split": validation,
        "qwen_model": "Qwen/Qwen3-VL-8B-Instruct",
        "fixed_after_generation": True,
    }
    write_json(out / "manifest.json", manifest)
    existing = evaluate_checkpoint(checkpoint=checkpoint, train_rows=train_rows, test_rows=test_rows, out_dir=out, prefix="existing")
    write_json(out / "synthetic_decomposition.json", existing)
    shutil.copyfile(out / "controller_existing.json", out / "controller_confusions.json")
    if retrain:
        maybe_retrain(data_dir=data_dir, out_dir=out, existing_summary=existing, max_epochs=retrain_epochs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run balanced synthetic diagnostic decomposition.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--checkpoint", default="/vol/checkpoints/synthetic_oracle/frameops_controller.pt")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--retrain-epochs", type=int, default=16)
    args = parser.parse_args()
    out = run(args.config, args.checkpoint, args.out_dir, retrain=not args.no_retrain, retrain_epochs=args.retrain_epochs)
    print(f"wrote diagnostic synthetic outputs to {out}")


if __name__ == "__main__":
    main()
