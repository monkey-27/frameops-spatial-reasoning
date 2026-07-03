from __future__ import annotations

from pathlib import Path

from datasets.synthetic import generate_dataset
from frameops.utils import read_jsonl


def test_synthetic_dataset_generation(tmp_path: Path) -> None:
    config = {
        "seed": 3,
        "output_dir": str(tmp_path / "synthetic"),
        "synthetic": {
            "image_size": 96,
            "train_scenes": 4,
            "val_scenes": 2,
            "test_scenes": 2,
            "min_objects": 3,
            "max_objects": 5,
            "heldout_test_hard": True,
        },
    }
    rows = generate_dataset(config)
    assert {split: len(items) for split, items in rows.items()} == {"train": 4, "val": 2, "test": 2}
    sample = read_jsonl(tmp_path / "synthetic" / "train.jsonl")[0]
    assert Path(sample["image_path"]).exists()
    assert sample["question"]
    assert sample["answer"] in {"yes", "no"}
    assert len(sample["objects"]) >= 3
    assert sample["program"]
    assert sample["camera"]
