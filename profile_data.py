#!/usr/bin/env python3
"""Profile example data (CSV, TSV, pasted text, XLSX) as few-shot context for generation.

    python profile_data.py data/real/sample.xlsx
    python profile_data.py data/real/pasted.txt --exclude name email --mode rows --rows 5
    python profile_data.py data/real/sample.csv --export data/real/prepared --holdout 0.3

Prints a compact summary the agent (or you) uses to match formats, ranges, category mixes,
missingness, correlations, keys and derived columns, plus warnings about sample size and
possibly personal columns. `--export DIR` writes normalised CSVs (one per table / sheet) so
`evaluate.py` can use them; with `--holdout F` it writes DIR/train and DIR/holdout and
profiles only the train part, so the generator never sees the rows you evaluate on.

Nothing is sent anywhere: this runs locally. But whoever reads the output (including an AI
assistant in a chat) sees what it prints.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backends.base import ConfigError  # noqa: E402
from backends.dataio import read_tables  # noqa: E402
from backends.fewshot import summarize_columns  # noqa: E402

SENSITIVE_TOKENS = {
    "name", "firstname", "lastname", "surname", "fullname", "email", "phone", "mobile", "tel",
    "address", "street", "zip", "postcode", "postal", "ssn", "passport", "mrn", "dob",
    "birthdate", "birth", "ip", "latitude", "lat", "lon", "lng", "longitude", "gps",
}


def sensitive_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for c in df.columns:
        toks = set(re.split(r"[^a-z0-9]+", str(c).lower())) - {""}
        hit = sorted(toks & SENSITIVE_TOKENS)
        if hit:
            out.append((str(c), hit[0]))
    return out


def key_columns(df: pd.DataFrame) -> list[str]:
    if len(df) < 20:
        return []
    return [c for c in df.columns
            if df[c].notna().all() and df[c].is_unique
            and (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
                 or pd.api.types.is_object_dtype(df[c]))]      # pandas 3 uses a dedicated string dtype


def fk_candidates(tables: dict[str, pd.DataFrame]) -> list[str]:
    keys = {t: key_columns(df) for t, df in tables.items()}
    found = []
    for ct, cdf in tables.items():
        for c in cdf.columns:
            if cdf[c].is_unique:
                continue
            for pt, pks in keys.items():
                if pt == ct or c not in pks:
                    continue
                vals = cdf[c].dropna()
                if len(vals) and vals.isin(tables[pt][c]).mean() >= 0.95:
                    found.append(f"{ct}.{c} -> {pt}.{c}")
    return found


def _numeric_cols(df: pd.DataFrame, limit: int) -> list[str]:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and df[c].nunique() > 10][:limit]


def derived_relations(df: pd.DataFrame) -> list[tuple[str, float]]:
    """Columns that are (nearly) the product or sum of two others, e.g. total = qty * unit_price."""
    # operands may be small integers (e.g. qty); the derived column itself must vary more
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() > 1][:12]
    if len(cols) < 3 or len(df) < 20:
        return []
    s = df[cols].head(2000)
    found = []
    for a, b in itertools.combinations(cols, 2):
        for op, sym in ((np.multiply, "*"), (np.add, "+")):
            pred = pd.Series(op(s[a], s[b]), index=s.index, name="p")
            for c in cols:
                if c in (a, b) or s[c].nunique() <= 10:
                    continue
                ok = pd.concat([pred, s[[a, b, c]]], axis=1).dropna()
                if len(ok) < 20:
                    continue
                if any(np.isclose(ok[c], ok[x], rtol=0.01, atol=0.02).mean() >= 0.9 for x in (a, b)):
                    continue                      # explained by one operand alone, not a real relation
                share = float(np.isclose(ok["p"], ok[c], rtol=0.01, atol=0.02).mean())
                if share >= 0.9:
                    found.append((f"{c} = {a} {sym} {b}", share))
    return found


def strong_correlations(df: pd.DataFrame) -> list[tuple[str, float]]:
    cols = _numeric_cols(df, 20)
    if len(cols) < 2 or len(df) < 30:
        return []
    corr = df[cols].corr(method="spearman")
    pairs = [(f"{a} ~ {b}", float(corr.loc[a, b])) for a, b in itertools.combinations(cols, 2)
             if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= 0.4]
    return sorted(pairs, key=lambda p: -abs(p[1]))[:8]


def size_note(n: int) -> str:
    if n < 30:
        return ("very small sample: use it for column names, formats, ranges and category names only. "
                "Distributions and correlations are not supported by this few rows")
    if n < 100:
        return "small sample: summaries are rough and a held-out evaluation is not possible"
    if n < 500:
        return "modest sample: a holdout is possible but evaluation scores will be noisy"
    return "sample size is adequate for stable summaries"


def build_profile(tables: dict[str, pd.DataFrame]) -> dict:
    prof = {"tables": {}, "foreign_keys": fk_candidates(tables), "warnings": []}
    for name, df in tables.items():
        miss = {str(c): round(float(df[c].isna().mean()), 3) for c in df.columns if df[c].isna().mean() > 0.5}
        const = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
        dups = int(df.duplicated().sum())
        info = {
            "rows": len(df), "columns": [str(c) for c in df.columns], "size_note": size_note(len(df)),
            "keys": [str(c) for c in key_columns(df)],
            "sensitive": sensitive_columns(df),
            "derived": derived_relations(df), "correlations": strong_correlations(df),
        }
        prof["tables"][name] = info
        w = prof["warnings"]
        if info["sensitive"]:
            w.append(f"{name}: possible personal data in " + ", ".join(f"{c} ({t})" for c, t in info["sensitive"])
                     + ". Check, and remove identifiers with --exclude before sharing or generating")
        if miss:
            w.append(f"{name}: mostly empty columns: " + ", ".join(f"{c} {v:.0%}" for c, v in miss.items()))
        if const:
            w.append(f"{name}: constant columns (no variation): " + ", ".join(const))
        if dups:
            w.append(f"{name}: {dups} fully duplicated rows")
    return prof


def render_markdown(prof: dict, tables: dict[str, pd.DataFrame], mode: str, n_rows: int, seed: int,
                    source: str) -> str:
    out = [f"# Example data profile", f"Source: {source} ({len(tables)} table(s))"]
    for name, df in tables.items():
        info = prof["tables"][name]
        out += ["", f"## Table `{name}` ({len(df):,} rows, {len(df.columns)} columns)",
                f"Sample size: {info['size_note']}.", "", "Columns:"]
        out += summarize_columns(df, set())
        det = []
        if info["keys"]:
            det.append("Likely keys (unique, never empty): " + ", ".join(info["keys"]))
        if info["derived"]:
            det.append("Derived columns (recompute, do not sample): " +
                       "; ".join(f"{e} ({s:.0%} of rows)" for e, s in info["derived"]))
        if info["correlations"]:
            det.append("Strong correlations (Spearman): " +
                       ", ".join(f"{p} {r:+.2f}" for p, r in info["correlations"]))
        if det:
            out += ["", "Detected:"] + [f"- {d}" for d in det]
        if mode == "rows":
            ex = df.sample(min(n_rows, len(df)), random_state=seed)
            out += ["", "Example rows:", ex.to_csv(index=False).strip()]
    if prof["foreign_keys"]:
        out += ["", "Likely foreign keys: " + "; ".join(prof["foreign_keys"])]
    if prof["warnings"]:
        out += ["", "Warnings:"] + [f"- {w}" for w in prof["warnings"]]
    out += ["", "How to use this:",
            "- Treat it as few-shot guidance: match names, formats, ranges, category mixes, missingness, "
            "correlations, keys and derived columns.",
            "- Never copy or hard-code sample rows or real identifiers into the generator or the output.",
            "- Where it differs from a domain profile, the sample wins; list the differences in the hand-off.",
            "- Evaluate: python evaluate.py --real <prepared folder or holdout> --synthetic <output folder>"]
    return "\n".join(out)


def export_tables(tables: dict[str, pd.DataFrame], dest: Path, holdout: float, seed: int):
    """Write normalised CSVs. With a holdout, write dest/train and dest/holdout (tables with fewer
    than 100 rows are not split and go entirely to train). Returns the tables the profile should use."""
    dest = Path(dest)
    if not holdout:
        dest.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(dest / f"{name}.csv", index=False)
        return tables, {}
    (dest / "train").mkdir(parents=True, exist_ok=True)
    (dest / "holdout").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train, held = {}, {}
    for name, df in tables.items():
        if len(df) < 100:
            train[name] = df
        else:
            perm = rng.permutation(len(df))
            k = int(round(len(df) * holdout))
            held[name] = df.iloc[perm[:k]].reset_index(drop=True)
            train[name] = df.iloc[perm[k:]].reset_index(drop=True)
            held[name].to_csv(dest / "holdout" / f"{name}.csv", index=False)
        train[name].to_csv(dest / "train" / f"{name}.csv", index=False)
    return train, held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="CSV, TSV, XLSX, pasted-text file, or a folder of them")
    ap.add_argument("--mode", choices=["stats", "rows"], default="stats",
                    help="stats (default) prints summaries only; rows also prints example rows")
    ap.add_argument("--rows", type=int, default=5, help="example rows per table in rows mode")
    ap.add_argument("--exclude", nargs="*", default=[], help="columns to drop before profiling and exporting")
    ap.add_argument("--export", help="write normalised CSVs to this folder")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction of rows to hold out for evaluation (needs --export; e.g. 0.3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the profile as JSON")
    a = ap.parse_args()
    if a.holdout and not a.export:
        sys.exit("error: --holdout needs --export DIR")
    if a.holdout and not 0 < a.holdout < 0.9:
        sys.exit("error: --holdout must be between 0 and 0.9")
    try:
        tables = read_tables(a.path)
    except ConfigError as e:
        sys.exit(f"error: {e}")
    drop = set(a.exclude)
    tables = {n: df.drop(columns=[c for c in df.columns if c in drop]) for n, df in tables.items()}
    held: dict = {}
    if a.export:
        tables, held = export_tables(tables, Path(a.export), a.holdout, a.seed)
    prof = build_profile(tables)
    if a.json:
        print(json.dumps(prof, indent=2, default=str))
        return
    print(render_markdown(prof, tables, a.mode, a.rows, a.seed, a.path))
    if a.export:
        dest = Path(a.export)
        if a.holdout:
            print(f"\nExported: {dest / 'train'} (profiled above) and {dest / 'holdout'} "
                  f"({', '.join(f'{t}: {len(d)} rows' for t, d in held.items()) or 'no table had 100+ rows, nothing held out'})")
        else:
            print(f"\nExported normalised CSVs to {dest}")


if __name__ == "__main__":
    main()
