#!/usr/bin/env python3
"""Evaluate synthetic data against real data, or compare several runs.

    python evaluate.py --real data/real --synthetic out/run_a out/run_b --labels claude ctgan
    python evaluate.py --real data/real --synthetic out/run_a --utility orders:total:regression \
        --validation-config configs/rules.yaml

Writes evaluation_report.json and evaluation_report.md into each synthetic folder.
Needs REAL data: realism can only be measured against it. Scores are 0-1, higher is better.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation import compare, evaluate_dataset, to_markdown  # noqa: E402
from run import load_config  # noqa: E402


def save(report: dict, out_dir: Path, title: str) -> None:
    (out_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (out_dir / "evaluation_report.md").write_text(to_markdown(report, title), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="folder of real CSVs")
    ap.add_argument("--synthetic", nargs="+", required=True, help="one or more synthetic folders")
    ap.add_argument("--labels", nargs="*", help="names for the synthetic folders")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra columns to ignore (ids ending _id are automatic)")
    ap.add_argument("--utility", help="table:target[:classification|regression] for the TSTR test")
    ap.add_argument("--validation-config", help="run config whose `validation` section defines integrity checks")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = {"exclude_columns": a.exclude, "seed": a.seed}
    if a.utility:
        p = a.utility.split(":")
        cfg["utility"] = {"table": p[0], "target": p[1], **({"task": p[2]} if len(p) > 2 else {})}
    validation = load_config(Path(a.validation_config)).get("validation") if a.validation_config else None
    labels = a.labels or [Path(d).name for d in a.synthetic]
    if len(labels) != len(a.synthetic):
        sys.exit("--labels must match --synthetic in count")

    reports = {}
    for label, d in zip(labels, a.synthetic):
        reports[label] = evaluate_dataset(a.real, d, cfg, validation)
        save(reports[label], Path(d), f"Evaluation: {label}")
        if len(a.synthetic) == 1:
            print(to_markdown(reports[label], f"Evaluation: {label}"))
    if len(reports) > 1:
        print(compare(reports))


if __name__ == "__main__":
    main()
