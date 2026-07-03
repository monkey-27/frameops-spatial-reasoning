from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_external import ADAPTERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an external dataset after manual download.")
    parser.add_argument("--dataset", choices=ADAPTERS, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="data/external/prepared.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    adapter = ADAPTERS[args.dataset]
    examples = adapter.load_examples(root=args.root, limit=args.limit, smoke=args.limit is not None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(examples)} examples to {out}")


if __name__ == "__main__":
    main()
