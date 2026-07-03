from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from frameops.controller import FrameOpsController
from frameops.metrics import summarize_predictions
from frameops.model import ID_TO_ANSWER, execute_program, task_from_sample
from frameops.slots import SpatialBatch
from frameops.utils import load_yaml, read_jsonl
from scripts.train_frameops import SyntheticProgramDataset, collate


def _oracle_predictions(rows: list[dict[str, Any]], noise: float = 0.0) -> list[dict[str, Any]]:
    out = []
    for idx, row in enumerate(rows):
        slots = SpatialBatch.from_objects(row["objects"])
        if noise:
            slots = slots.add_noise(center_std=noise, extent_std=noise * 0.5, yaw_std=noise, seed=idx)
        pred = execute_program(row, slots=slots)
        out.append(
            {
                "id": row["id"],
                "prediction": pred.answer,
                "answer": row["answer"],
                "task_type": task_from_sample(row),
                "difficulty": row.get("metadata", {}).get("difficulty", "unknown"),
                "logit": pred.logit,
                "trace": pred.trace,
                "method": "frameops_oracle_program",
                "noise": noise,
            }
        )
    return out


@torch.no_grad()
def _controller_predictions(rows: list[dict[str, Any]], checkpoint: Path) -> list[dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ds = SyntheticProgramDataset(
        rows,
        vocab=ckpt["vocab"],
        label_vocab=ckpt["label_vocab"],
        task_vocab=ckpt["task_vocab"],
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    model = FrameOpsController(
        vocab_size=len(ckpt["vocab"]),
        slot_dim=ckpt["slot_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_operators=len(ckpt["task_vocab"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    preds: list[str] = []
    for batch in loader:
        output = model(batch["question"], batch["slots"], batch["slot_mask"])
        preds.extend(ID_TO_ANSWER[int(i)] for i in output.answer_logits.argmax(dim=-1))
    return [
        {
            "id": row["id"],
            "prediction": pred,
            "answer": row["answer"],
            "task_type": task_from_sample(row),
            "difficulty": row.get("metadata", {}).get("difficulty", "unknown"),
            "method": "frameops_controller",
        }
        for row, pred in zip(rows, preds, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate synthetic FrameOps pilot outputs.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="results/synthetic_eval.json")
    args = parser.parse_args()
    config = load_yaml(args.config)
    data_dir = Path(config.get("output_dir", "data/synthetic/oracle_v0"))
    rows = read_jsonl(data_dir / f"{args.split}.jsonl")
    result: dict[str, Any] = {
        "split": args.split,
        "notes": "Local synthetic eval. Qwen image-only and geometry-text baselines are produced by scripts/eval_qwen_baselines.py, not faked here.",
        "qwen_baselines": None,
        "oracle_program": {},
    }
    all_predictions: list[dict[str, Any]] = []
    for noise in config.get("eval", {}).get("noise_levels", [0.0]):
        preds = _oracle_predictions(rows, noise=float(noise))
        all_predictions.extend(preds)
        result["oracle_program"][str(noise)] = summarize_predictions(preds)
    if args.checkpoint:
        controller_rows = _controller_predictions(rows, Path(args.checkpoint))
        all_predictions.extend(controller_rows)
        result["controller"] = summarize_predictions(controller_rows)
    result["predictions"] = all_predictions
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
