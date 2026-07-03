from __future__ import annotations

from pathlib import Path

from datasets.synthetic import generate_dataset


def test_tiny_generation_smoke(tmp_path: Path) -> None:
    config = {
        "seed": 5,
        "output_dir": str(tmp_path / "smoke"),
        "synthetic": {"image_size": 64, "train_scenes": 2, "val_scenes": 1, "test_scenes": 1},
    }
    rows = generate_dataset(config)
    assert sum(len(v) for v in rows.values()) == 4
