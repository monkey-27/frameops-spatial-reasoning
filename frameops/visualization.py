from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_scene_topdown(sample: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for obj in sample.get("objects", []):
        x, y, _ = obj["center"]
        sx, sy, _ = obj["extent"]
        rect = plt.Rectangle((x - sx / 2, y - sy / 2), sx, sy, fill=False)
        ax.add_patch(rect)
        ax.text(x, y, obj["id"], ha="center", va="center", fontsize=8)
    ax.scatter([0], [0], marker="^", color="black", label="camera")
    ax.set_xlabel("x")
    ax.set_ylabel("depth y")
    ax.set_title(sample.get("id", "scene"))
    ax.axis("equal")
    ax.legend(loc="best")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
