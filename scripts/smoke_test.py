from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.synthetic import generate_dataset
from frameops.utils import load_yaml
from scripts.eval_synthetic import _oracle_predictions
from scripts.train_frameops import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local no-Qwen FrameOps smoke test.")
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = generate_dataset(config)
    ckpt = train_from_config(args.config, max_train=16, max_val=8, epochs=1)
    oracle = _oracle_predictions(rows["test"], noise=0.0)
    summary = {
        "checkpoint": str(ckpt),
        "splits": {split: len(split_rows) for split, split_rows in rows.items()},
        "oracle_correct": sum(p["prediction"] == p["answer"] for p in oracle),
        "oracle_total": len(oracle),
    }
    out = Path("results/smoke_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
