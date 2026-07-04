from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synthetic import generate_diagnostic_dataset
from frameops.diagnostics import DIAGNOSTIC_TASKS, ensure_out_dir, validate_diagnostic_split, write_json
from frameops.model import task_from_sample
from frameops.progress import ProgressLogger
from frameops.utils import load_yaml, read_jsonl
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl, set_deterministic_generation
from vlm.vl_trace import generate_with_trace


def _balanced_subset(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    per_task = max(1, n // len(DIAGNOSTIC_TASKS))
    selected: list[dict[str, Any]] = []
    for task in DIAGNOSTIC_TASKS:
        selected.extend([row for row in rows if task_from_sample(row) == task][:per_task])
    return selected[:n]


def run(config: str, out_dir: str | Path, n: int, model: str, max_new_tokens: int) -> Path:
    out = ensure_out_dir(out_dir)
    progress = ProgressLogger(out / "progress.jsonl")
    cfg = load_yaml(config)
    data_dir = out / "data"
    if not (data_dir / "diagnostic_test.jsonl").exists():
        progress.log("input_trace", "started", "generating diagnostic split")
        generate_diagnostic_dataset(
            data_dir,
            seed=int(cfg.get("diagnostic_seed", 20260703)),
            image_size=int(cfg.get("synthetic", {}).get("image_size", 256)),
        )
    rows = read_jsonl(data_dir / "diagnostic_test.jsonl")
    validate_diagnostic_split(rows, min_per_task=50)
    rows = _balanced_subset(rows, n)
    progress.log("input_trace", "started", f"loading {model}")
    bundle = load_qwen3_vl(model)
    set_deterministic_generation(bundle.model)
    traces = []
    for idx, sample in enumerate(rows):
        raw, trace = generate_with_trace(
            bundle,
            sample["question"],
            image_path=sample["image_path"],
            require_image_input=True,
            max_new_tokens=max_new_tokens,
        )
        trace.update(
            {
                "id": sample["id"],
                "task_family": task_from_sample(sample),
                "hardness": sample.get("metadata", {}).get("hardness", "unknown"),
                "raw_output": raw,
            }
        )
        traces.append(trace)
        if (idx + 1) % 4 == 0 or idx + 1 == len(rows):
            progress.log("input_trace", "running", f"traced {idx + 1}/{len(rows)} samples")
    payload = {
        "model": model,
        "n": len(traces),
        "all_generation_calls_received_image_tensors": all(
            row["generation_received_image_related_tensors"] for row in traces
        ),
        "processor_key_sets": sorted({tuple(row["processor_output_keys"]) for row in traces}),
        "image_related_key_sets": sorted({tuple(row["image_related_processor_keys"]) for row in traces}),
        "samples": traces,
    }
    write_json(out / "vlm_input_trace.json", payload)
    progress.log("input_trace", "completed", f"wrote {out / 'vlm_input_trace.json'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Qwen3-VL processor/model image inputs.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=QWEN3_VL_MODEL)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.out_dir, args.n, args.model, args.max_new_tokens)


if __name__ == "__main__":
    main()
