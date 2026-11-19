#!/usr/bin/env python3
"""Train a conventional synthetic-data model (Gaussian copula / CTGAN / TVAE) on REAL data.

    python train.py --config configs/ctgan.yaml
    python train.py --config configs/ctgan.yaml --model-dir models/diabetes_ctgan

Then generate any amount of data from the saved model, no real data needed:
    python run.py --config configs/ctgan.yaml        (uses conventional.model_dir)

Saves per-table models, schema.json, training_log.json (hyperparameters, device, loss
curves, held-out evaluation) and holdout_evaluation.md. Models are pickles: only load
models you trained yourself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import ConfigError  # noqa: E402
from backends.sdv_backend import train  # noqa: E402
from run import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    conv = cfg.get("conventional") or {}
    model_dir = Path(a.model_dir or conv.get("model_dir") or f"models/{conv.get('algorithm', 'gaussian_copula')}")
    try:
        log = train(cfg, model_dir.resolve())
    except ConfigError as e:
        sys.exit(f"error: {e}")
    log["loss"] = {t: {k: f"{len(v)} epochs, last={v[-1]}" for k, v in d.items() if v and k != 'Epoch'}
                   for t, d in log["loss"].items()}
    print(json.dumps(log, indent=2, default=str))
    print(f"\nModel saved to {model_dir.resolve()}")


if __name__ == "__main__":
    main()
