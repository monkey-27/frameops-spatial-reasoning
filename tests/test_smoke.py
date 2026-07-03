from __future__ import annotations

from pathlib import Path

import yaml

from datasets.synthetic import generate_dataset
from scripts.train_frameops import train_from_config


def test_tiny_generation_smoke(tmp_path: Path) -> None:
    config = {
        "seed": 5,
        "output_dir": str(tmp_path / "smoke"),
        "synthetic": {"image_size": 64, "train_scenes": 2, "val_scenes": 1, "test_scenes": 1},
    }
    rows = generate_dataset(config)
    assert sum(len(v) for v in rows.values()) == 4


def test_tiny_training_loop(tmp_path: Path) -> None:
    config = {
        "seed": 5,
        "output_dir": str(tmp_path / "train_smoke_data"),
        "synthetic": {"image_size": 64, "train_scenes": 8, "val_scenes": 4, "test_scenes": 2},
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "hidden_dim": 32,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "lr": 0.001,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ckpt = train_from_config(config_path, max_train=8, max_val=4, epochs=1)
    assert ckpt.exists()
