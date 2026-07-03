from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frameops.metrics import summarize_predictions
from frameops.model import task_from_sample
from frameops.utils import load_yaml, read_jsonl
from vlm.qwen_baselines import QwenBaselineRunner
from vlm.qwen_loader import QWEN3_VL_MODEL, load_qwen3_vl

BASELINE_MODES = ["image_only", "geometry_json", "geometry_nl", "frameops_text"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL baselines on synthetic samples.")
    parser.add_argument("--config", default="configs/qwen_frameops_lora.yaml")
    parser.add_argument("--data-config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--modes", nargs="+", default=BASELINE_MODES)
    parser.add_argument("--output", default="results/qwen_baselines.json")
    args = parser.parse_args()

    model_config = load_yaml(args.config).get("model", {})
    data_config = load_yaml(args.data_config)
    rows = read_jsonl(Path(data_config.get("output_dir", "data/synthetic/oracle_v0")) / f"{args.split}.jsonl")
    rows = rows[: args.limit] if args.limit else rows
    bundle = load_qwen3_vl(
        model_config.get("name", QWEN3_VL_MODEL),
        dtype=model_config.get("dtype", "auto"),
        device_map=model_config.get("device_map", "auto"),
    )
    runner = QwenBaselineRunner(bundle, max_new_tokens=int(model_config.get("max_new_tokens", 16)))
    result = {"model": bundle.model_name, "split": args.split, "limit": len(rows), "modes": {}}
    for mode in args.modes:
        preds = []
        for row in rows:
            pred = runner.predict(row, mode)
            preds.append(
                {
                    "id": row["id"],
                    "mode": mode,
                    "prediction": pred.prediction,
                    "raw_text": pred.raw_text,
                    "answer": row["answer"],
                    "task_type": task_from_sample(row),
                    "difficulty": row.get("metadata", {}).get("difficulty", "unknown"),
                }
            )
        result["modes"][mode] = {"summary": summarize_predictions(preds), "predictions": preds}
        print(mode, result["modes"][mode]["summary"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
