from __future__ import annotations

import json

from modal_app import CACHE, VOL, app, hf_cache, image, volume, _run


@app.function(image=image, gpu="L4", timeout=4 * 60 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def eval(
    config: str = "configs/synthetic_oracle.yaml",
    split: str = "test",
    checkpoint: str | None = None,
    output: str = "/vol/results/synthetic_eval.json",
) -> dict[str, str]:
    args = ["scripts/eval_synthetic.py", "--config", config, "--split", split, "--output", output]
    if checkpoint:
        args.extend(["--checkpoint", checkpoint])
    _run(args)
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "output": output}


@app.local_entrypoint()
def main(
    config: str = "configs/synthetic_oracle.yaml",
    split: str = "test",
    checkpoint: str | None = None,
) -> None:
    print(json.dumps(eval.remote(config=config, split=split, checkpoint=checkpoint), indent=2, sort_keys=True))
