# Data

Generated synthetic data should live under `data/synthetic/` and is ignored by git.
External datasets should live under `data/external/` and are ignored by git.

No script in this repo silently downloads large external data. Use
`scripts/prepare_external_dataset.py` for validation and conversion after manual
download or after passing an explicit small-download flag where an adapter supports it.
