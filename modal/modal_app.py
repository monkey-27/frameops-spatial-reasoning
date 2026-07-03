from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/root/frameops_pilot")
VOL = Path("/vol")
CACHE = Path("/cache")

app = modal.App("frameops-pilot")
volume = modal.Volume.from_name("frameops-pilot-vol", create_if_missing=True)
hf_cache = modal.Volume.from_name("frameops-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env(
        {
            "PYTHONPATH": str(REMOTE_ROOT),
            "HF_HOME": str(CACHE / "huggingface"),
            "HF_HUB_CACHE": str(CACHE / "huggingface" / "hub"),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .uv_pip_install(
        "torch>=2.2",
        "numpy>=1.24",
        "pillow>=10.0",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "pytest>=8.0",
        "transformers>=4.57.6",
        "accelerate>=0.33",
        "peft>=0.11",
        "datasets>=2.20",
        "huggingface_hub>=0.24",
        "qwen-vl-utils>=0.0.11",
        "safetensors>=0.4",
    )
    .add_local_dir(ROOT / "frameops", remote_path=str(REMOTE_ROOT / "frameops"))
    .add_local_dir(ROOT / "vlm", remote_path=str(REMOTE_ROOT / "vlm"))
    .add_local_dir(ROOT / "datasets", remote_path=str(REMOTE_ROOT / "datasets"))
    .add_local_dir(ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"))
    .add_local_file(ROOT / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    .add_local_file(ROOT / "modal" / "modal_app.py", remote_path="/root/modal_app.py")
)


def _run(args: list[str]) -> None:
    os.chdir(REMOTE_ROOT)
    subprocess.run([sys.executable, *args], check=True)


@app.function(image=image, timeout=30 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def smoke() -> dict[str, str]:
    _run(["scripts/smoke_test.py", "--config", "configs/modal_smoke.yaml"])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "summary": str(REMOTE_ROOT / "results/smoke_summary.json")}


@app.function(image=image, timeout=60 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def generate_synthetic(config: str = "configs/modal_smoke.yaml") -> dict[str, str]:
    _run(["scripts/generate_synthetic.py", "--config", config])
    volume.commit()
    return {"status": "ok", "config": config}


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def eval_baselines(
    data_config: str = "configs/modal_smoke.yaml",
    model_config: str = "configs/qwen_frameops_lora.yaml",
    split: str = "test",
    limit: int = 8,
) -> dict[str, str]:
    output = str(VOL / "results" / "qwen_baselines.json")
    _run(
        [
            "scripts/eval_qwen_baselines.py",
            "--config",
            model_config,
            "--data-config",
            data_config,
            "--split",
            split,
            "--limit",
            str(limit),
            "--output",
            output,
        ]
    )
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "output": output}
