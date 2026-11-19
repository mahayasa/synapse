"""Run all metrics on a (real, synthetic) pair of CSV folders and build reports."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M

# Heuristic composite weights; always read the component scores, not just the composite.
WEIGHTS = {"fidelity": 0.40, "utility": 0.20, "privacy": 0.15, "coverage_validity": 0.10, "integrity": 0.15}


def _mean(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.mean(xs)) if xs else None


def evaluate_dataset(real_dir, synth_dir, cfg: dict | None = None, validation: dict | None = None) -> dict:
    """cfg keys: exclude_columns, seed, utility: {table, target, task, features}, max_rows."""
    cfg = cfg or {}
    seed = int(cfg.get("seed", 0))
    real_dir, synth_dir = Path(real_dir), Path(synth_dir)
    real_files = {p.stem: p for p in real_dir.glob("*.csv")}
    synth_files = {p.stem: p for p in synth_dir.glob("*.csv")}
    tables = sorted(set(real_files) & set(synth_files))
    notes = []
    if not tables:
        raise ValueError(f"no tables in common between {real_dir} and {synth_dir}")
    for t in sorted(set(real_files) ^ set(synth_files)):
        notes.append(f"table {t!r} exists on only one side and was skipped")

    per_table = {}
    for t in tables:
        real, synth = pd.read_csv(real_files[t]), pd.read_csv(synth_files[t])
        r, s, kinds = M.prepare(real, synth, cfg.get("exclude_columns", ()))
        if len(real) < 500:
            notes.append(f"{t}: only {len(real)} real rows, scores are noisy")
        res = {"rows_real": len(real), "rows_synthetic": len(synth),
               "columns_evaluated": list(kinds),
               "column_shapes": M.column_shapes(r, s, kinds),
               "pair_trends": M.pair_trends(r, s, kinds),
               "c2st": M.c2st(r, s, kinds, seed, cfg.get("max_rows", 4000)),
               "coverage_validity": M.coverage_validity(r, s, kinds),
               "privacy": M.privacy(r, s, kinds, seed)}
        u = cfg.get("utility")
        if u and u.get("table") == t:
            res["utility"] = M.utility(r, s, kinds, u, seed)
        per_table[t] = res

    def grab(key, sub="score"):
        return [(v.get(key) or {}).get(sub) if isinstance(v.get(key), dict) else None
                for v in per_table.values()]

    groups = {
        "fidelity": _mean([_mean([a, b, c]) for a, b, c in zip(
            grab("column_shapes"), grab("pair_trends"), grab("c2st"))]),
        "utility": _mean(grab("utility")),
        "privacy": _mean(grab("privacy")),
        "coverage_validity": _mean(grab("coverage_validity")),
        "integrity": None,
    }
    integrity = None
    if validation:
        from backends.common import run_validation
        _, out = run_validation(synth_dir, validation)
        n_pass, n_fail = len(re.findall(r"^\s+PASS", out, re.M)), len(re.findall(r"^\s+FAIL", out, re.M))
        if n_pass + n_fail:
            groups["integrity"] = n_pass / (n_pass + n_fail)
        integrity = {"passed": n_pass, "failed": n_fail}
    avail = {k: v for k, v in groups.items() if v is not None}
    w = sum(WEIGHTS[k] for k in avail)
    composite = sum(WEIGHTS[k] * v for k, v in avail.items()) / w if w else None
    return {"composite": composite, "groups": groups, "integrity": integrity,
            "tables": per_table, "notes": notes, "weights": WEIGHTS}


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


def to_markdown(report: dict, title: str = "Synthetic data evaluation") -> str:
    lines = [f"# {title}", "", f"**Composite score: {_f(report['composite'])}** (heuristic weighted mean, 0-1, higher is better)", "",
             "| Group | Score | Weight |", "|---|---|---|"]
    for k, v in report["groups"].items():
        lines.append(f"| {k} | {_f(v)} | {report['weights'][k]} |")
    for t, res in report["tables"].items():
        lines += ["", f"## Table `{t}` ({res['rows_real']:,} real / {res['rows_synthetic']:,} synthetic rows)", "",
                  "| Metric | Score | Detail |", "|---|---|---|"]
        cs = res["column_shapes"]
        weak = sorted(cs["columns"].items(), key=lambda kv: kv[1])[:3]
        lines.append(f"| column_shapes | {_f(cs['score'])} | weakest: " +
                     ", ".join(f"{c} {v:.2f}" for c, v in weak) + " |")
        pt = res["pair_trends"]
        weak = sorted(pt["pairs"].items(), key=lambda kv: kv[1])[:3]
        lines.append(f"| pair_trends | {_f(pt['score'])} | weakest: " +
                     ", ".join(f"{c} {v:.2f}" for c, v in weak) + " |")
        if res["c2st"]:
            lines.append(f"| c2st | {_f(res['c2st']['score'])} | detector AUC {res['c2st']['auc']:.3f} (0.5 = indistinguishable) |")
        lines.append(f"| coverage_validity | {_f(res['coverage_validity']['score'])} | |")
        if res["privacy"]:
            p = res["privacy"]
            lines.append(f"| privacy | {_f(p['score'])} | exact-copy rate {p['exact_match_rate_synthetic']:.3f} "
                         f"(baseline {p['exact_match_rate_holdout_baseline']:.3f}), median DCR {p['median_dcr_synthetic']:.3f} "
                         f"vs holdout {p['median_dcr_holdout']:.3f} |")
        if res.get("utility"):
            u = res["utility"]
            lines.append(f"| utility (TSTR) | {_f(u.get('score'))} | {u.get('metric', '')}: synth {u.get('train_synth_test_real', 0):.3f}"
                         f" vs real {u.get('train_real_test_real', 0):.3f} |")
    if report.get("integrity"):
        lines += ["", f"Integrity checks: {report['integrity']['passed']} passed, {report['integrity']['failed']} failed."]
    if report["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in report["notes"]]
    lines += ["", "Interpretation: fidelity = looks like real; utility = trains useful models; privacy = no copying of real rows "
              "(relative to a real holdout); coverage_validity = values within real ranges and categories."]
    return "\n".join(lines)


def compare(reports: dict) -> str:
    """Markdown leaderboard of several evaluated synthetic datasets."""
    keys = list(next(iter(reports.values()))["groups"])
    lines = ["| Run | Composite | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for label, rep in sorted(reports.items(), key=lambda kv: -(kv[1]["composite"] or 0)):
        lines.append(f"| {label} | {_f(rep['composite'])} | " + " | ".join(_f(rep["groups"][k]) for k in keys) + " |")
    return "\n".join(lines)
