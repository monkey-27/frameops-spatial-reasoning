from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import (
    arkitscenes_adapter,
    embodiedscan_adapter,
    kubric_adapter,
    mindedit_adapter,
    sqa3d_adapter,
    vsi_adapter,
)

ADAPTERS = {
    "sqa3d": sqa3d_adapter,
    "vsi": vsi_adapter,
    "mindedit": mindedit_adapter,
    "embodiedscan": embodiedscan_adapter,
    "arkitscenes": arkitscenes_adapter,
    "kubric": kubric_adapter,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or convert external datasets into FrameOps format.")
    parser.add_argument("--dataset", choices=ADAPTERS, required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", default="results/external_preview.json")
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset]
    examples = adapter.load_examples(root=args.root, limit=args.limit, smoke=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "examples": examples}, f, indent=2, sort_keys=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
