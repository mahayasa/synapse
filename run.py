#!/usr/bin/env python3
"""Config-driven runner: pick a backend (LLM or conventional), generate, validate, log.

    python run.py --config configs/rules.yaml
    python run.py --config configs/anthropic.yaml --out out/my_run

API keys: the config names an environment variable (api_key_env). The key itself
never goes in the config, logs, or output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends import BudgetExceeded, ConfigError, get_backend  # noqa: E402
from backends.common import run_validation  # noqa: E402


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError:
        sys.exit("YAML config needs PyYAML (pip install pyyaml), or use a .json config.")
    return yaml.safe_load(text)


def run(cfg: dict, out: Path | None = None) -> dict:
    backend = get_backend(cfg)
    out_dir = out or Path(cfg.get("out_dir") or
                          f"out/{backend.label()}_seed{cfg.get('seed', 42)}")
    out_dir = Path(out_dir).resolve()   # scripts run with cwd=out_dir, so paths must be absolute
    t0 = time.time()
    meta = backend.generate(out_dir)
    ok, report = run_validation(out_dir, cfg.get("validation"))
    log = {"backend": cfg["backend"], "seed": cfg.get("seed", 42), "rows": cfg.get("rows"),
           "domain": cfg.get("domain"), "out_dir": str(out_dir), **meta,
           "validation_passed": ok, "total_seconds": round(time.time() - t0, 1)}
    ev = cfg.get("evaluation")
    if ev and ev.get("real_data"):
        from evaluation import evaluate_dataset, to_markdown
        rep = evaluate_dataset(ev["real_data"], out_dir, ev, cfg.get("validation"))
        (out_dir / "evaluation_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        (out_dir / "evaluation_report.md").write_text(to_markdown(rep), encoding="utf-8")
        log["evaluation"] = {"composite": rep["composite"], **rep["groups"]}
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    (out_dir / "validation_report.txt").write_text(report, encoding="utf-8")
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", help="override output directory")
    a = ap.parse_args()
    try:
        log = run(load_config(Path(a.config)), Path(a.out) if a.out else None)
    except (ConfigError, BudgetExceeded) as e:
        sys.exit(f"error: {e}")
    print(json.dumps(log, indent=2, default=str))
    for w in log.get("warnings") or []:
        print(f"warning: {w}", file=sys.stderr)
    sys.exit(0 if log["validation_passed"] else 1)


if __name__ == "__main__":
    main()
