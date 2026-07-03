from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mode(summary: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not summary:
        return None
    return summary.get("modes", {}).get(key)


def _acc(mode: dict[str, Any] | None) -> str:
    if not mode:
        return "missing"
    overall = mode.get("overall", {})
    return f"{overall.get('mean', 0.0):.3f} [{overall.get('low', 0.0):.3f}, {overall.get('high', 0.0):.3f}] n={overall.get('n', 0)}"


def _plain_acc(mode: dict[str, Any] | None) -> float | None:
    if not mode:
        return None
    return mode.get("overall", {}).get("mean")


def _decision(synth: dict[str, Any] | None, qwen: dict[str, Any] | None, hidden: dict[str, Any] | None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    continue_hits = 0
    oracle_text_delta = None
    if qwen:
        oracle = qwen.get("modes", {}).get("oracle_frameops_text")
        geom = qwen.get("modes", {}).get("geometry_json")
        if oracle and geom:
            oracle_text_delta = oracle.get("paired_delta_vs_baseline", {}).get("mean")
            if oracle_text_delta is not None and oracle_text_delta >= 0.05 and oracle.get("paired_delta_vs_baseline", {}).get("low", -1) > -0.02:
                continue_hits += 1
                reasons.append("oracle FrameOps text clears +5 point paired-delta criterion")
            else:
                reasons.append("oracle FrameOps text does not clear +5 point criterion vs geometry JSON")
    if hidden and hidden.get("minimal_hidden_evidence_run_completed") and hidden.get("hidden_vs_geometry_json"):
        delta = hidden["hidden_vs_geometry_json"]["paired_delta_hidden_minus_geometry_json"]["mean"]
        if delta > 0:
            continue_hits += 1
            reasons.append("hidden evidence exercised and beats geometry JSON on hidden subset")
        else:
            reasons.append("hidden evidence exercised but does not beat geometry JSON")
    elif hidden:
        reasons.append("hidden evidence was not exercised as method evidence")
    learned_ok = False
    if synth:
        learned = synth.get("modes", {}).get("existing:learned_controller_oracle_slots")
        if learned:
            acc = learned["overall"]["mean"]
            min_task = min(v["mean"] for v in learned.get("by_task_family", {}).values())
            if acc >= 0.80 and min_task >= 0.65:
                continue_hits += 1
                learned_ok = True
                reasons.append("learned controller clears 80% overall and 65% task floor")
            else:
                reasons.append(f"learned controller below criterion: overall={acc:.3f}, task_floor={min_task:.3f}")
        noisy = synth.get("modes", {}).get("existing:oracle_program_noisy_slots")
        if noisy and noisy.get("by_noise", {}).get("0.1", {}).get("mean", 0.0) >= 0.90:
            continue_hits += 1
            reasons.append("oracle algebra gains survive noise 0.10")
    if continue_hits >= 2:
        return "CONTINUE cautiously", reasons
    if qwen and oracle_text_delta is not None and oracle_text_delta <= 0:
        return "KILL this exact text-interface implementation", reasons
    if synth and not learned_ok:
        return "PIVOT", reasons
    return "INCOMPLETE", reasons


def write_report(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    manifest = _load(out / "manifest.json")
    synth = _load(out / "synthetic_decomposition.json")
    qwen = _load(out / "qwen_text_baselines.json")
    hidden = _load(out / "hidden_connector_status.json")
    retrain = _load(out / "controller_retrained_summary.json")
    decision, reasons = _decision(synth, qwen, hidden)

    lines = [
        "# FrameOps Diagnostic Report",
        "",
        f"Result folder: `{out}`",
        f"Decision: **{decision}**",
        "",
        "## Direct Answers",
        "",
        "1. Does oracle FrameOps still solve the synthetic tasks?",
        f"   - {_acc(_mode(synth, 'existing:oracle_program_oracle_slots'))}",
        "2. Where does the learned controller fail?",
    ]
    if synth:
        diag = synth.get("controller_diagnostics", {})
        lines.extend(
            [
                f"   - object top1: {diag.get('object_binding_top1_accuracy', 0.0):.3f}",
                f"   - object top3-all: {diag.get('object_binding_top3_all_accuracy', 0.0):.3f}",
                f"   - frame selection: {diag.get('frame_selection_accuracy', 0.0):.3f}",
                f"   - operator selection: {diag.get('operator_selection_accuracy', 0.0):.3f}",
                f"   - dominant failures: `{json.dumps(diag.get('per_task_dominant_failure_mode', {}), sort_keys=True)}`",
            ]
        )
    else:
        lines.append("   - synthetic decomposition missing.")
    lines.extend(
        [
            "3. Does FrameOps text evidence beat geometry JSON/NL under oracle geometry?",
        ]
    )
    if qwen:
        lines.extend(
            [
                f"   - geometry JSON: {_acc(_mode(qwen, 'geometry_json'))}",
                f"   - geometry NL: {_acc(_mode(qwen, 'geometry_natural_language'))}",
                f"   - oracle FrameOps text: {_acc(_mode(qwen, 'oracle_frameops_text'))}",
                f"   - paired delta vs JSON: `{_mode(qwen, 'oracle_frameops_text').get('paired_delta_vs_baseline') if _mode(qwen, 'oracle_frameops_text') else 'missing'}`",
            ]
        )
    else:
        lines.append("   - Qwen text diagnostic missing or incomplete.")
    lines.append("4. Does learned FrameOps text evidence beat geometry JSON/NL?")
    if qwen and _mode(qwen, "learned_frameops_text"):
        lines.append(f"   - learned FrameOps text: {_acc(_mode(qwen, 'learned_frameops_text'))}")
    else:
        lines.append("   - learned FrameOps text missing or controller unavailable.")
    lines.append("5. Can hidden evidence actually be injected into Qwen3-VL?")
    if hidden:
        lines.extend(
            [
                f"   - tested in forward pass: `{hidden.get('tested_in_forward_pass')}`",
                f"   - tested in generation: `{hidden.get('tested_in_generation')}`",
                f"   - minimal hidden run completed: `{hidden.get('minimal_hidden_evidence_run_completed')}`",
                f"   - blockers: `{json.dumps(hidden.get('blockers', []))}`",
            ]
        )
    else:
        lines.append("   - hidden connector status missing.")
    lines.append("6. Are results consistent across task families?")
    if synth and _mode(synth, "existing:learned_controller_oracle_slots"):
        lines.append(f"   - learned controller task accuracies: `{json.dumps(_mode(synth, 'existing:learned_controller_oracle_slots').get('by_task_family', {}), sort_keys=True)}`")
    if qwen and _mode(qwen, "oracle_frameops_text"):
        lines.append(f"   - oracle FrameOps text task accuracies: `{json.dumps(_mode(qwen, 'oracle_frameops_text').get('by_task_family', {}), sort_keys=True)}`")
    lines.extend(
        [
            "7. Is there evidence that the interface matters beyond geometry availability?",
            "   - See decision reasons below; this report treats geometry JSON parity as negative evidence.",
            "8. Should we continue, pivot, or kill this exact direction?",
            f"   - **{decision}**",
            "",
            "## Decision Reasons",
        ]
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(["", "## Key Mode Accuracies", ""])
    if synth:
        for key in [
            "existing:oracle_program_oracle_slots",
            "existing:oracle_program_noisy_slots",
            "existing:oracle_objects_learned_frame_operator",
            "existing:oracle_frame_operator_learned_objects",
            "existing:oracle_operator_learned_objects_frame",
            "existing:learned_controller_oracle_slots",
            "existing:majority_answer",
            "existing:program_prior_only",
        ]:
            lines.append(f"- `{key}`: {_acc(_mode(synth, key))}")
    if qwen:
        lines.extend(["", "## Qwen Text Modes", ""])
        for key in sorted(qwen.get("modes", {})):
            lines.append(f"- `{key}`: {_acc(_mode(qwen, key))}")
    if retrain:
        lines.extend(["", "## Retraining Hygiene", "", f"`{json.dumps(retrain, sort_keys=True)}`"])
    lines.extend(["", "## Manifest", "", f"`{json.dumps(manifest or {}, sort_keys=True)}`"])
    report_path = out / "diagnostic_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write blunt diagnostic report from saved JSON artifacts.")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    path = write_report(args.out_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
