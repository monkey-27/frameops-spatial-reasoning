from __future__ import annotations

import json

from modal_app import CACHE, REMOTE_ROOT, VOL, app, hf_cache, image, volume, _run


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60, volumes={str(VOL): volume, str(CACHE): hf_cache})
def train(config: str = "configs/synthetic_oracle.yaml", max_train: int | None = None) -> dict[str, str]:
    args = ["scripts/train_frameops.py", "--config", config]
    if max_train is not None:
        args.extend(["--max-train", str(max_train)])
    _run(args)
    volume.commit()
    hf_cache.commit()
    return {"status": "ok", "remote_root": str(REMOTE_ROOT), "config": config}


@app.local_entrypoint()
def main(config: str = "configs/synthetic_oracle.yaml", max_train: int | None = None) -> None:
    print(json.dumps(train.remote(config=config, max_train=max_train), indent=2, sort_keys=True))
