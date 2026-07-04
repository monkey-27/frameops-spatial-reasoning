from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProgressLogger:
    """Append-only JSONL progress log for long Modal runs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: str, status: str, message: str = "", **extra: Any) -> None:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "status": status,
            "message": message,
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[{row['timestamp']}] {step} {status}: {message}", flush=True)
