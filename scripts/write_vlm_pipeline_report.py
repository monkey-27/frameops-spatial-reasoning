from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _acc(summary: dict[str, Any] | None, mode: str) -> float | None:
    if not summary:
        return None
    try:
        mode_summary = summary["modes"][mode]
        if "overall" in mode_summary:
            return float(mode_summary["overall"]["mean"])
        if "accuracy" in mode_summary:
            return float(mode_summary["accuracy"]["mean"])
    except Exception:
        return None
    return None


def _fmt(value: float | None) -> str:
    return "missing" if value is None else f"{value:.3f}"


def run(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    identity = _load(out / "model_identity.json")
    trace = _load(out / "vlm_input_trace.json")
    sensitivity = _load(out / "image_sensitivity.json")
    baselines = _load(out / "qwen_vl_baselines.json")
    hidden = _load(out / "hidden_connector_debug.json")

    identity_pass = bool(identity and identity.get("pass"))
    trace_pass = bool(trace and trace.get("all_generation_calls_received_image_tensors"))
    correct = _acc(sensitivity, "correct_image")
    blank = _acc(sensitivity, "blank_image")
    wrong = _acc(sensitivity, "shuffled_wrong_image")
    img_json = _acc(baselines, "image_plus_geometry_json")
    img_nl = _acc(baselines, "image_plus_geometry_nl")
    img_oracle = _acc(baselines, "image_plus_oracle_frameops_text")
    img_learned = _acc(baselines, "image_plus_learned_frameops_text")
    text_json = _acc(baselines, "text_only_geometry_json")
    text_oracle = _acc(baselines, "text_only_oracle_frameops_text")

    if not identity_pass:
        decision = "STOP: the exact Qwen3-VL identity/image-input check failed."
        previous_validity = "Invalid for VLM claims until this identity failure is fixed."
    elif not trace_pass:
        decision = "STOP: generation traces did not prove image tensors reached Qwen3-VL."
        previous_validity = "Contaminated or unproven for VLM claims."
    elif correct is not None and blank is not None and wrong is not None and (
        correct <= blank + 0.05 or sensitivity.get("prediction_agreement_correct_vs_blank", 1.0) > 0.85
    ):
        decision = "PIVOT: real Qwen3-VL path, but image sensitivity is low or text/geometry leakage dominates."
        previous_validity = "Model path is VLM, but the task/prompt may not test vision strongly enough."
    elif img_oracle is not None and img_json is not None and img_oracle <= img_json + 0.02:
        decision = "KILL TEXT-INTERFACE FRAMEOPS: real image path, image-sensitive setup, but oracle FrameOps text does not beat geometry JSON."
        previous_validity = "Valid as a text-interface negative result if identity and trace passed."
    elif img_oracle is not None and img_json is not None and img_oracle > img_json + 0.05:
        decision = "CONTINUE: image+oracle FrameOps text appears to beat image+geometry JSON; rerun with replication."
        previous_validity = "Earlier negative conclusion should be revised for the VLM path."
    else:
        decision = "INCOMPLETE: missing enough Qwen-VL baseline evidence for a research decision."
        previous_validity = "Do not use as evidence until missing artifacts are present."

    hidden_feasible = bool(hidden and hidden.get("minimal_hidden_evidence_run_completed"))
    report = [
        "# Qwen3-VL Pipeline Check",
        "",
        "## Direct Answers",
        "",
        f"1. Actual loaded model was `Qwen/Qwen3-VL-8B-Instruct`: **{identity_pass}**.",
        f"2. Definitely not text-only Qwen3: **{bool(identity and not identity.get('is_text_only_qwen'))}**.",
        f"3. Processor emitted image-related tensors: **{bool(identity and identity.get('supports_pixel_values_or_image_grid'))}**.",
        f"4. Generation received image-related tensors: **{trace_pass or bool(identity and identity.get('image_mode_generation_received_image_tensors'))}**.",
        (
            "5. Image replacement changed behavior: "
            f"correct={_fmt(correct)}, blank={_fmt(blank)}, wrong={_fmt(wrong)}, "
            f"agreement_correct_blank={sensitivity.get('prediction_agreement_correct_vs_blank', 'missing') if sensitivity else 'missing'}, "
            f"agreement_correct_wrong={sensitivity.get('prediction_agreement_correct_vs_wrong', 'missing') if sensitivity else 'missing'}."
        ),
        (
            "6. Image+geometry vs text-only geometry: "
            f"image+JSON={_fmt(img_json)}, text-only JSON={_fmt(text_json)}."
        ),
        (
            "7. Image+oracle FrameOps text vs image+geometry: "
            f"oracle FrameOps={_fmt(img_oracle)}, geometry JSON={_fmt(img_json)}, geometry NL={_fmt(img_nl)}."
        ),
        f"8. Previous FrameOps diagnostic validity: **{previous_validity}**",
        f"9. Hidden evidence injection feasible in actual Qwen3-VL generation: **{hidden_feasible}**.",
        f"10. Decision: **{decision}**",
        "",
        "## Baseline Snapshot",
        "",
        f"- `image_only`: {_fmt(_acc(baselines, 'image_only'))}",
        f"- `image_plus_geometry_json`: {_fmt(img_json)}",
        f"- `image_plus_geometry_nl`: {_fmt(img_nl)}",
        f"- `image_plus_oracle_frameops_text`: {_fmt(img_oracle)}",
        f"- `image_plus_learned_frameops_text`: {_fmt(img_learned)}",
        f"- `text_only_geometry_json`: {_fmt(text_json)}",
        f"- `text_only_oracle_frameops_text`: {_fmt(text_oracle)}",
        "",
        "## Hidden Connector",
        "",
        f"- Completed minimal hidden generation: {hidden_feasible}",
        f"- Blockers: {hidden.get('blockers', []) if hidden else 'missing hidden_connector_debug.json'}",
        f"- Recommended fix: {hidden.get('recommended_fix', 'missing') if hidden else 'missing'}",
        "",
        "## Blunt Conclusion",
        "",
        decision,
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "vlm_pipeline_report.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the VLM pipeline diagnostic report.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    path = run(args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
