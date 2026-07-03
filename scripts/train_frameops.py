from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.synthetic import generate_dataset
from frameops.controller import FrameOpsController
from frameops.losses import frameops_supervised_loss
from frameops.model import (
    ANSWER_LABELS,
    FRAME_LABELS,
    frame_label_from_program,
    selected_object_ids,
    task_from_sample,
)
from frameops.utils import load_yaml, read_jsonl, set_seed

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _ensure_data(config: dict[str, Any]) -> Path:
    out_dir = Path(config.get("output_dir", "data/synthetic/oracle_v0"))
    if not (out_dir / "train.jsonl").exists():
        generate_dataset(config)
    return out_dir


def _build_vocab(rows: list[dict[str, Any]], min_count: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for tok in _tokenize(row["question"]):
            counts[tok] = counts.get(tok, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, count in sorted(counts.items()):
        if count >= min_count:
            vocab[tok] = len(vocab)
    return vocab


def _label_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(obj.get("label", "object")) for row in rows for obj in row["objects"]})
    return {label: idx for idx, label in enumerate(labels)}


def _task_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    tasks = sorted({task_from_sample(row) for row in rows})
    return {task: idx for idx, task in enumerate(tasks)}


def _encode_question(text: str, vocab: dict[str, int], max_len: int = 48) -> list[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in _tokenize(text)[:max_len]]
    return ids or [vocab["<unk>"]]


def _slot_feature(obj: dict[str, Any], label_vocab: dict[str, int]) -> torch.Tensor:
    label = str(obj.get("label", "object"))
    one_hot = torch.zeros(len(label_vocab), dtype=torch.float32)
    if label in label_vocab:
        one_hot[label_vocab[label]] = 1.0
    base = torch.tensor(
        list(obj["center"]) + list(obj["extent"]) + [float(obj.get("yaw", 0.0)), float(obj.get("confidence", 1.0))],
        dtype=torch.float32,
    )
    return torch.cat([base, one_hot], dim=0)


class SyntheticProgramDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        vocab: dict[str, int],
        label_vocab: dict[str, int],
        task_vocab: dict[str, int],
    ):
        self.rows = rows
        self.vocab = vocab
        self.label_vocab = label_vocab
        self.task_vocab = task_vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        selected = selected_object_ids(row)
        return {
            "question": torch.tensor(_encode_question(row["question"], self.vocab), dtype=torch.long),
            "slots": [_slot_feature(obj, self.label_vocab) for obj in row["objects"]],
            "object_ids": [obj["id"] for obj in row["objects"]],
            "object_labels": [1.0 if obj["id"] in selected else 0.0 for obj in row["objects"]],
            "answer": torch.tensor(ANSWER_LABELS[str(row["answer"]).lower()], dtype=torch.long),
            "operator": torch.tensor(self.task_vocab[task_from_sample(row)], dtype=torch.long),
            "frame": torch.tensor(FRAME_LABELS.get(frame_label_from_program(row), FRAME_LABELS["world"]), dtype=torch.long),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_q = max(item["question"].numel() for item in batch)
    max_slots = max(len(item["slots"]) for item in batch)
    slot_dim = batch[0]["slots"][0].numel()
    q = torch.zeros(len(batch), max_q, dtype=torch.long)
    slots = torch.zeros(len(batch), max_slots, slot_dim)
    slot_mask = torch.zeros(len(batch), max_slots, dtype=torch.bool)
    object_labels = torch.zeros(len(batch), max_slots)
    for i, item in enumerate(batch):
        q[i, : item["question"].numel()] = item["question"]
        slot_tensor = torch.stack(item["slots"], dim=0)
        slots[i, : slot_tensor.shape[0]] = slot_tensor
        slot_mask[i, : slot_tensor.shape[0]] = True
        object_labels[i, : len(item["object_labels"])] = torch.tensor(item["object_labels"])
    return {
        "question": q,
        "slots": slots,
        "slot_mask": slot_mask,
        "object_labels": object_labels,
        "answer": torch.stack([item["answer"] for item in batch]),
        "operator": torch.stack([item["operator"] for item in batch]),
        "frame": torch.stack([item["frame"] for item in batch]),
    }


@torch.no_grad()
def evaluate_controller(model: FrameOpsController, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct_answer = 0
    correct_operator = 0
    correct_frame = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch["question"], batch["slots"], batch["slot_mask"])
        total += batch["answer"].numel()
        correct_answer += int((output.answer_logits.argmax(dim=-1) == batch["answer"]).sum().item())
        correct_operator += int((output.operator_logits.argmax(dim=-1) == batch["operator"]).sum().item())
        correct_frame += int((output.frame_logits.argmax(dim=-1) == batch["frame"]).sum().item())
    if total == 0:
        return {"answer_accuracy": 0.0, "operator_accuracy": 0.0, "frame_accuracy": 0.0}
    return {
        "answer_accuracy": correct_answer / total,
        "operator_accuracy": correct_operator / total,
        "frame_accuracy": correct_frame / total,
    }


def train_from_config(
    config_path: str | Path,
    *,
    max_train: int | None = None,
    max_val: int | None = None,
    epochs: int | None = None,
) -> Path:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 0)))
    data_dir = _ensure_data(config)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    if max_train is not None:
        train_rows = train_rows[:max_train]
    if max_val is not None:
        val_rows = val_rows[:max_val]

    vocab = _build_vocab(train_rows)
    label_vocab = _label_vocab(train_rows)
    task_vocab = _task_vocab(train_rows)
    train_ds = SyntheticProgramDataset(train_rows, vocab=vocab, label_vocab=label_vocab, task_vocab=task_vocab)
    val_ds = SyntheticProgramDataset(val_rows, vocab=vocab, label_vocab=label_vocab, task_vocab=task_vocab)
    batch_size = int(config.get("training", {}).get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    slot_dim = train_ds[0]["slots"][0].numel()
    hidden_dim = int(config.get("training", {}).get("hidden_dim", 128))
    model = FrameOpsController(
        vocab_size=len(vocab),
        slot_dim=slot_dim,
        hidden_dim=hidden_dim,
        num_operators=len(task_vocab),
        num_frames=len(FRAME_LABELS),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("training", {}).get("lr", 1e-3)))
    n_epochs = int(epochs if epochs is not None else config.get("training", {}).get("epochs", 8))

    ckpt_dir = Path(config.get("training", {}).get("checkpoint_dir", "checkpoints/synthetic_oracle"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "answer_accuracy", "operator_accuracy", "frame_accuracy"])
        writer.writeheader()
        for epoch in range(1, n_epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(batch["question"], batch["slots"], batch["slot_mask"])
                loss, _ = frameops_supervised_loss(
                    output,
                    answer_labels=batch["answer"],
                    operator_labels=batch["operator"],
                    object_labels=batch["object_labels"],
                    frame_labels=batch["frame"],
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics = evaluate_controller(model, val_loader, device)
            writer.writerow({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), **metrics})
            f.flush()
            print(f"epoch={epoch} loss={sum(losses) / max(1, len(losses)):.4f} val={metrics}")

    ckpt_path = ckpt_dir / "frameops_controller.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab,
            "label_vocab": label_vocab,
            "task_vocab": task_vocab,
            "slot_dim": slot_dim,
            "hidden_dim": hidden_dim,
            "config": config,
        },
        ckpt_path,
    )
    with (ckpt_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "task_vocab": task_vocab,
                "label_vocab": label_vocab,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small FrameOps controller on synthetic data.")
    parser.add_argument("--config", default="configs/synthetic_oracle.yaml")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    ckpt = train_from_config(args.config, max_train=args.max_train, max_val=args.max_val, epochs=args.epochs)
    print(f"saved checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
