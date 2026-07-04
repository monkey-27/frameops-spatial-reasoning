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
        path = Path(failure_path)
        if path.exists():
            path = path.with_suffix(path.suffix + ".failure.json")
        path.write_text(
            json.dumps({"status": "failed", "step": label, "error": str(exc)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return False


def _vlm_out_dir(out_dir: str | None) -> str:
    return out_dir or str(VOL / "results" / f"vlm_pipeline_check_{_timestamp()}")


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


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def check_qwen_vl_identity(
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    _run(["scripts/check_qwen_vl_identity.py", "--model", model, "--out-dir", out])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "artifact": str(Path(out) / "model_identity.json")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_vlm_input_trace(
    config: str = "configs/synthetic_oracle.yaml",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
    n: int = 12,
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    _run(["scripts/trace_qwen_vl_inputs.py", "--config", config, "--model", model, "--out-dir", out, "--n", str(n)])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "artifact": str(Path(out) / "vlm_input_trace.json")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=6 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_qwen_vl_image_sensitivity(
    config: str = "configs/synthetic_oracle.yaml",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
    n: int = 60,
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    _run(
        [
            "scripts/eval_qwen_vl_image_sensitivity.py",
            "--config",
            config,
            "--model",
            model,
            "--out-dir",
            out,
            "--n",
            str(n),
        ]
    )
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "artifact": str(Path(out) / "image_sensitivity.json")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=16 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_qwen_vl_baselines(
    config: str = "configs/synthetic_oracle.yaml",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
    n: int = 360,
    checkpoint: str = "/vol/checkpoints/synthetic_oracle/frameops_controller.pt",
    require_image_input: bool = True,
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    _run(
        [
            "scripts/eval_qwen_vl_diagnostic.py",
            "--config",
            config,
            "--model",
            model,
            "--out-dir",
            out,
            "--n",
            str(n),
            "--checkpoint",
            checkpoint,
            "--require-image-input",
            str(require_image_input).lower(),
        ]
    )
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "artifact": str(Path(out) / "qwen_vl_baselines.json")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_qwen_vl_hidden_debug(
    config: str = "configs/synthetic_oracle.yaml",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    _run(["scripts/debug_hidden_connector_qwen_vl.py", "--config", config, "--model", model, "--out-dir", out])
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "out_dir": out, "artifact": str(Path(out) / "hidden_connector_debug.json")}


@app.function(image=image, timeout=30 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def diagnostic_qwen_vl_pipeline_report(out_dir: str) -> dict[str, str]:
    _run(["scripts/write_vlm_pipeline_report.py", "--out-dir", out_dir])
    volume.commit()
    return {"status": "ok", "out_dir": out_dir, "artifact": str(Path(out_dir) / "vlm_pipeline_report.md")}


@app.function(
    image=image,
    gpu="A10G",
    timeout=24 * 60 * 60,
    volumes={str(VOL): volume, str(CACHE): hf_cache},
    secrets=[hf_secret],
)
def diagnostic_qwen_vl_pipeline_all(
    config: str = "configs/synthetic_oracle.yaml",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    out_dir: str | None = None,
    n_baselines: int = 360,
    n_sensitivity: int = 60,
    n_trace: int = 12,
    checkpoint: str = "/vol/checkpoints/synthetic_oracle/frameops_controller.pt",
) -> dict[str, str]:
    out = _vlm_out_dir(out_dir)
    identity_ok = _try_run(
        ["scripts/check_qwen_vl_identity.py", "--model", model, "--out-dir", out],
        str(Path(out) / "model_identity.json"),
        "check_qwen_vl_identity",
    )
    if identity_ok:
        _try_run(
            ["scripts/trace_qwen_vl_inputs.py", "--config", config, "--model", model, "--out-dir", out, "--n", str(n_trace)],
            str(Path(out) / "vlm_input_trace.json"),
            "vlm_input_trace",
        )
        _try_run(
            [
                "scripts/eval_qwen_vl_image_sensitivity.py",
                "--config",
                config,
                "--model",
                model,
                "--out-dir",
                out,
                "--n",
                str(n_sensitivity),
            ],
            str(Path(out) / "image_sensitivity.json"),
            "qwen_vl_image_sensitivity",
        )
        _try_run(
            [
                "scripts/eval_qwen_vl_diagnostic.py",
                "--config",
                config,
                "--model",
                model,
                "--out-dir",
                out,
                "--n",
                str(n_baselines),
                "--checkpoint",
                checkpoint,
                "--require-image-input",
                "true",
            ],
            str(Path(out) / "qwen_vl_baselines.json"),
            "qwen_vl_baselines",
        )
        _try_run(
            ["scripts/debug_hidden_connector_qwen_vl.py", "--config", config, "--model", model, "--out-dir", out],
            str(Path(out) / "hidden_connector_debug.json"),
            "qwen_vl_hidden_debug",
        )
    _try_run(
        ["scripts/write_vlm_pipeline_report.py", "--out-dir", out],
        str(Path(out) / "vlm_pipeline_report.md"),
        "vlm_pipeline_report",
    )
    volume.commit()
    hf_cache.commit()
    return {"status": "ok" if identity_ok else "identity_failed", "out_dir": out}
