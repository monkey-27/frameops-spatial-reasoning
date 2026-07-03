from __future__ import annotations

from frameops.evidence import (
    serialize_frameops_text,
    serialize_geometry_json,
    serialize_geometry_nl,
)

SYSTEM_PROMPT = (
    "Answer the spatial question with a short answer. For yes/no questions, answer only yes or no."
)


def build_prompt(sample: dict, mode: str) -> str:
    question = sample["question"]
    if mode == "image_only":
        return question
    if mode == "geometry_json":
        return f"{question}\n\nOracle geometry JSON:\n{serialize_geometry_json(sample)}"
    if mode == "geometry_nl":
        return f"{question}\n\nGeometry description:\n{serialize_geometry_nl(sample)}"
    if mode == "frameops_text":
        return f"{question}\n\n{serialize_frameops_text(sample)}"
    raise ValueError(f"Unknown prompt mode {mode!r}")
