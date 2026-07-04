from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import DIAGNOSTIC_TASKS, accuracy_ci, ensure_out_dir, validate_diagnostic_split, write_json
from frameops.model import task_from_sample
from frameops.progress import ProgressLogger
from frameops.utils import load_yaml, read_jsonl
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation
from vlm.vl_trace import generate_with_trace, hash_image_file


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
    per_task = max(1, n // len(DIAGNOSTIC_TASKS))
    selected: list[dict[str, Any]] = []
    for task in DIAGNOSTIC_TASKS:
        task_rows = [row for row in rows if task_from_sample(row) == task]
        if len(task_rows) < per_task:
            raise ValueError(f"Need {per_task} rows for {task}, found {len(task_rows)}")
        selected.extend(task_rows[:per_task])
    return selected[:n]


def _agreement(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]]) -> float:
    a = {row["id"]: row["parsed_prediction"] for row in rows_a}
    b = {row["id"]: row["parsed_prediction"] for row in rows_b}
    ids = sorted(set(a) & set(b))
    return sum(a[i] == b[i] for i in ids) / max(1, len(ids))


def run(config: str, out_dir: str | Path, n: int, model: str, max_new_tokens: int) -> Path:
    out = ensure_out_dir(out_dir)
    progress = ProgressLogger(out / "progress.jsonl")
    cfg = load_yaml(config)
    data_dir = out / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        progress.log("image_sensitivity", "started", "generating diagnostic split")
        generate_diagnostic_dataset(
            data_dir,
            seed=int(cfg.get("diagnostic_seed", 20260703)),
            image_size=int(cfg.get("synthetic", {}).get("image_size", 256)),
        )
    rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
    validate_diagnostic_split(rows, min_per_task=50)
    rows = _balanced_subset(rows, n)
    blank_path = out / "image_sensitivity_blank.png"
    Image.new("RGB", (256, 256), (238, 240, 244)).save(blank_path)
    progress.log("image_sensitivity", "started", f"loading {model}")
    bundle = load_qwen3_vl(model)
    set_deterministic_generation(bundle.model)
    predictions: list[dict[str, Any]] = []
    for idx, sample in enumerate(rows):
        wrong = rows[(idx + max(1, len(rows) // 3)) % len(rows)]
        variants = {
            "correct_image": sample["image_path"],
            "blank_image": str(blank_path),
            "shuffled_wrong_image": wrong["image_path"],
        }
        for mode, image_path in variants.items():
            raw, trace = generate_with_trace(
                bundle,
                sample["question"],
                image_path=image_path,
                require_image_input=True,
                max_new_tokens=max_new_tokens,
            )
            parsed = _parse_answer(raw)
            predictions.append(
                {
                    "id": sample["id"],
                    "task_family": task_from_sample(sample),
                    "hardness": sample.get("metadata", {}).get("hardness", "unknown"),
                    "mode": mode,
                    "prompt": sample["question"],
                    "image_path": image_path,
                    "image_hash": hash_image_file(image_path),
                    "wrong_image_source_id": wrong["id"] if mode == "shuffled_wrong_image" else None,
                    "processor_input_keys": trace["processor_output_keys"],
                    "generation_received_image_related_tensors": trace["generation_received_image_related_tensors"],
                    "gold_answer": sample["answer"],
                    "raw_model_output": raw,
                    "parsed_prediction": parsed,
                    "correct": parsed == str(sample["answer"]).lower(),
                }
            )
        if (idx + 1) % 5 == 0 or idx + 1 == len(rows):
            progress.log("image_sensitivity", "running", f"evaluated {idx + 1}/{len(rows)} samples")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["mode"]].append(row)
    correct_rows = grouped["correct_image"]
    blank_rows = grouped["blank_image"]
    wrong_rows = grouped["shuffled_wrong_image"]
    unchanged_blank = []
    unchanged_wrong = []
    correct_by_id = {row["id"]: row for row in correct_rows}
    for row in blank_rows:
        if correct_by_id[row["id"]]["parsed_prediction"] == row["parsed_prediction"] and len(unchanged_blank) < 20:
            unchanged_blank.append(
                {
                    "id": row["id"],
                    "task_family": row["task_family"],
                    "gold": row["gold_answer"],
                    "prediction": row["parsed_prediction"],
                }
            )
    for row in wrong_rows:
        if correct_by_id[row["id"]]["parsed_prediction"] == row["parsed_prediction"] and len(unchanged_wrong) < 20:
            unchanged_wrong.append(
                {
                    "id": row["id"],
                    "task_family": row["task_family"],
                    "gold": row["gold_answer"],
                    "prediction": row["parsed_prediction"],
                    "wrong_image_source_id": row["wrong_image_source_id"],
                }
            )
    payload = {
        "model": model,
        "n": len(rows),
        "modes": {
            mode: {
                "accuracy": accuracy_ci([bool(row["correct"]) for row in mode_rows]),
                "parse_failure_rate": sum(row["parsed_prediction"] not in {"yes", "no"} for row in mode_rows)
                / max(1, len(mode_rows)),
            }
            for mode, mode_rows in sorted(grouped.items())
        },
        "prediction_agreement_correct_vs_blank": _agreement(correct_rows, blank_rows),
        "prediction_agreement_correct_vs_wrong": _agreement(correct_rows, wrong_rows),
        "examples_unchanged_despite_blank": unchanged_blank,
        "examples_unchanged_despite_wrong_image": unchanged_wrong,
        "all_generation_calls_received_image_tensors": all(
            row["generation_received_image_related_tensors"] for row in predictions
        ),
        "predictions": predictions,
    }
    write_json(out / "image_sensitivity.json", payload)
    progress.log("image_sensitivity", "completed", f"wrote {out / 'image_sensitivity.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether Qwen3-VL predictions change when images are replaced.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    run(args.config, args.out_dir, args.n, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
