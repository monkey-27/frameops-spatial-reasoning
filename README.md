# FrameOps Pilot

FrameOps is a falsifiable research pilot for one question:

> Given the same base VLM and the same scene geometry, does an executable
> spatial-operator interface outperform passing geometry as text/JSON metadata
> for 3D spatial reasoning?

The base VLM is fixed to `Qwen/Qwen3-VL-8B-Instruct`. The code intentionally
raises if another model name is supplied. If the exact Qwen model cannot be
loaded or confirmed from Hugging Face, stop and document that blocker rather
than switching models.

## Hypothesis

The critical baseline is not image-only Qwen. The critical baseline is Qwen with
the same oracle or predicted geometry serialized as JSON or natural language.
FrameOps is only interesting if executable spatial operators beat
geometry-as-text under oracle geometry, then survive slot noise and at least one
real benchmark subset.

## What Is Implemented

- Pure Python synthetic 3D toy-scene generator with rendered 2D projections.
- Oracle programs and answers for camera left/right, depth, above/below,
  distance, object/agent frames, vector composition, occlusion, counterfactual
  translation, counterfactual viewer rotation, and perspective-taking.
- Differentiable PyTorch operators in `frameops/operators.py`.
- Spatial slots, reference frames, configurable noisy slots, and an oracle
  program executor.
- Small supervised FrameOps controller for smoke training.
- Evidence serialization as text plus an experimental Qwen3-VL connector
  adapter that adds projected FrameOps evidence to visual-token features.
- Qwen3-VL image-only, geometry-JSON, geometry-NL, and FrameOps-text baselines.
- Conservative external dataset adapters for SQA3D, VSI-Bench-Debiased,
  MindEdit-Bench, EmbodiedScan, ARKitScenes, and Kubric.
- Modal entrypoints for smoke, generation, training, synthetic eval, and Qwen
  baseline eval.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Local Smoke

No Qwen download is needed for these commands.

```bash
python3 -m pytest
python3 scripts/smoke_test.py
python3 scripts/generate_synthetic.py --config configs/local_smoke.yaml
python3 scripts/train_frameops.py --config configs/local_smoke.yaml
python3 scripts/eval_synthetic.py --config configs/local_smoke.yaml --checkpoint checkpoints/local_smoke/frameops_controller.pt
```

Synthetic data, checkpoints, and results are git-ignored.

## Qwen Baselines

These commands download and run `Qwen/Qwen3-VL-8B-Instruct`, so use Modal or a
GPU machine with Hugging Face access.

```bash
python3 scripts/eval_qwen_baselines.py \
  --data-config configs/synthetic_oracle.yaml \
  --config configs/qwen_frameops_lora.yaml \
  --split test \
  --limit 16
```

Modes are `image_only`, `geometry_json`, `geometry_nl`, and `frameops_text`.
Deterministic decoding is the default.

## Modal

Create a Modal secret named `huggingface-token` with `HF_TOKEN` before Qwen
baseline eval. Smoke and synthetic controller training do not require the token.

```bash
python3 -m modal run modal/modal_app.py::smoke
python3 -m modal run modal/modal_app.py::generate_synthetic --config configs/modal_smoke.yaml
python3 -m modal run modal/modal_train.py::train --config configs/modal_synthetic_oracle.yaml
python3 -m modal run modal/modal_eval.py::eval --config configs/modal_synthetic_oracle.yaml --checkpoint /vol/checkpoints/synthetic_oracle/frameops_controller.pt
python3 -m modal run modal/modal_app.py::eval_baselines --data-config configs/modal_smoke.yaml --limit 8
```

The Modal image uses persistent volumes for `/vol` outputs and `/cache`
Hugging Face cache.

## External Datasets

Adapters never silently download huge datasets.

```bash
python3 scripts/eval_external.py --dataset sqa3d --root /path/to/sqa3d --limit 8
python3 scripts/eval_external.py --dataset vsi --root hf --limit 8
python3 scripts/eval_external.py --dataset mindedit --root hf --limit 8
python3 scripts/prepare_external_dataset.py --dataset sqa3d --root /path/to/sqa3d --output data/external/sqa3d.jsonl
```

EmbodiedScan, ARKitScenes, and Kubric require prepared local roots. VSI and
MindEdit support explicit Hugging Face annotation sampling with `--root hf`.

## Hidden Evidence Status

Hidden evidence injection is implemented as an experimental Qwen3-VL connector
adapter in `vlm/qwen_frameops_adapter.py`. It wraps Qwen3-VL image-feature
extraction and adds a gated projection of FrameOps evidence to visual-token
features without changing token counts or M-RoPE bookkeeping.

This is a real integration path, but it has not been validated by downloading
and running Qwen in this local smoke. The fully reliable fallback is
`frameops_text`, which serializes FrameOps evidence into the prompt for
debugging and baseline comparisons.

## Go / No-Go

Continue only if:

1. FrameOps beats geometry-as-text on oracle synthetic tasks.
2. Gains survive moderate slot noise.
3. FrameOps improves at least one real/external spatial benchmark subset.

Kill or pivot if:

1. geometry-as-text matches FrameOps;
2. gains only appear on synthetic scenes;
3. gains disappear under realistic geometry noise;
4. improvements are limited to trivial left/right tasks.

## Known Limitations

- Synthetic rendering is deliberately simple and is not evidence by itself.
- The controller uses small supervised text/slot features; Qwen remains frozen
  for the intended pilot comparison.
- External adapters are schema validators/converters, not dataset mirrors.
- No results in this repo should be treated as scientific evidence until a real
  Qwen Modal run writes artifacts under `results/`.
