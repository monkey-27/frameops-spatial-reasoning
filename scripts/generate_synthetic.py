from __future__ import annotations

import argparse

from datasets.synthetic import generate_dataset
from frameops.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic oracle FrameOps data.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = generate_dataset(config)
    for split, split_rows in rows.items():
        print(f"{split}: {len(split_rows)} samples")
    print(f"wrote {config.get('output_dir', 'data/synthetic/oracle_v0')}")


if __name__ == "__main__":
    main()
