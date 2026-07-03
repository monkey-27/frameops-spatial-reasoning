from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
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
        "torchvision>=0.17",
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


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _try_run(args: list[str], failure_path: str, label: str) -> bool:
    try:
        _run(args)
        return True
    except Exception as exc:
        Path(failure_path).parent.mkdir(parents=True, exist_ok=True)
        Path(failure_path).write_text(
            json.dumps({"status": "failed", "step": label, "error": str(exc)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return False


def _best_controller_checkpoint(out: str, fallback: str) -> str:
    summary_path = Path(out) / "controller_retrained_summary.json"
    if not summary_path.exists():
        return fallback
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return fallback
    if summary.get("retrained") and summary.get("best_checkpoint"):
        return str(summary["best_checkpoint"])
    return fallback


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


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def diagnostic_synthetic(
    config: str = "configs/synthetic_oracle.yaml",
    checkpoint: str = "/vol/checkpoints/synthetic_oracle/frameops_controller.pt",
    out_dir: str | None = None,
    retrain: bool = True,
    retrain_epochs: int = 16,
) -> dict[str, str]:
    out = out_dir or str(VOL / "results" / f"diagnostic_{_timestamp()}")
    args = [
        "scripts/eval_synthetic_decomposition.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--out_dir",
        out,
        "--retrain-epochs",
        str(retrain_epochs),
    ]
    if not retrain:
        args.append("--no-retrain")
    _run(args)
    volume.commit()
    return {"status": "ok", "out_dir": out}


@app.function(
    image=image,
    gpu="A10G",
    timeout=14 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_qwen_text(
    config: str = "configs/synthetic_oracle.yaml",
    out_dir: str | None = None,
    n: int = 360,
    checkpoint: str = "/vol/checkpoints/synthetic_oracle/frameops_controller.pt",
) -> dict[str, str]:
    out = out_dir or str(VOL / "results" / f"diagnostic_{_timestamp()}")
    _run(
        [
            "scripts/eval_qwen_text_diagnostic.py",
            "--config",
            config,
            "--out_dir",
            out,
            "--n",
            str(n),
            "--checkpoint",
            checkpoint,
        ]
    )
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "n": str(n)}


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_hidden_status(
    config: str = "configs/synthetic_oracle.yaml",
    out_dir: str | None = None,
    n: int = 1,
) -> dict[str, str]:
    out = out_dir or str(VOL / "results" / f"diagnostic_{_timestamp()}")
    _run(["scripts/hidden_connector_status.py", "--config", config, "--out_dir", out, "--n", str(n)])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "n": str(n)}


@app.function(image=image, timeout=30 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def diagnostic_report(out_dir: str) -> dict[str, str]:
    _run(["scripts/write_diagnostic_report.py", "--out_dir", out_dir])
    volume.commit()
    return {"status": "ok", "out_dir": out_dir, "report": str(Path(out_dir) / "diagnostic_report.md")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=20 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_all(
    config: str = "configs/synthetic_oracle.yaml",
    checkpoint: str = "/vol/checkpoints/synthetic_oracle/frameops_controller.pt",
    out_dir: str | None = None,
    qwen_n: int = 360,
    hidden_n: int = 1,
    retrain: bool = True,
    retrain_epochs: int = 16,
) -> dict[str, str]:
    out = out_dir or str(VOL / "results" / f"diagnostic_{_timestamp()}")
    synth_args = [
        "scripts/eval_synthetic_decomposition.py",
        "--config",
        config,
        "--checkpoint",
        checkpoint,
        "--out_dir",
        out,
        "--retrain-epochs",
        str(retrain_epochs),
    ]
    if not retrain:
        synth_args.append("--no-retrain")
    _run(synth_args)
    qwen_checkpoint = _best_controller_checkpoint(out, checkpoint)
    _try_run(
        [
            "scripts/eval_qwen_text_diagnostic.py",
            "--config",
            config,
            "--out_dir",
            out,
            "--n",
            str(qwen_n),
            "--checkpoint",
            qwen_checkpoint,
        ],
        str(Path(out) / "qwen_text_baselines.json"),
        "qwen_text_diagnostic",
    )
    _try_run(
        ["scripts/hidden_connector_status.py", "--config", config, "--out_dir", out, "--n", str(hidden_n)],
        str(Path(out) / "hidden_connector_status.json"),
        "hidden_connector_status",
    )
    _run(["scripts/write_diagnostic_report.py", "--out_dir", out])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "qwen_n": str(qwen_n), "hidden_n": str(hidden_n)}
