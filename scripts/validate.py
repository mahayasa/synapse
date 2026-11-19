#!/usr/bin/env python3
"""Generic validation for a directory of CSV tables.

Examples:
    python validate.py --dir out \
        --pk customers=customer_id orders=order_id \
        --fk orders.customer_id=customers.customer_id \
        --range orders.total=0:1000000 orders.qty=1:100 \
        --notnull orders.order_id orders.customer_id

Always reports: row counts, per-column null rates, fully duplicated rows.
Exit code 0 = all checks passed, 1 = at least one failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(dir_: Path) -> dict[str, pd.DataFrame]:
    tables = {p.stem: pd.read_csv(p) for p in sorted(dir_.glob("*.csv"))}
    if not tables:
        sys.exit(f"No CSV files found in {dir_}")
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pk", nargs="*", default=[], help="table=column[,column]")
    ap.add_argument("--fk", nargs="*", default=[], help="child.col=parent.col")
    ap.add_argument("--range", nargs="*", default=[], help="table.col=min:max (either side may be empty)")
    ap.add_argument("--notnull", nargs="*", default=[], help="table.col")
    ap.add_argument("--max-null-rate", type=float, default=0.10,
                    help="warn when an undeclared column exceeds this null rate")
    args = ap.parse_args()

    tables = load(Path(args.dir))
    failures: list[str] = []
    warns: list[str] = []

    def ok(msg): print(f"  PASS  {msg}")
    def bad(msg):
        failures.append(msg); print(f"  FAIL  {msg}")

    print("Row counts")
    for t, df in tables.items():
        print(f"  {t}: {len(df):,} rows, {len(df.columns)} cols")

    print("\nPrimary keys")
    for spec in args.pk:
        t, cols = spec.split("=")
        cols = cols.split(",")
        df = tables[t]
        dup = df.duplicated(subset=cols).sum()
        nulls = df[cols].isna().any(axis=1).sum()
        (ok if dup == 0 and nulls == 0 else bad)(f"{t}({','.join(cols)}) unique, {dup} dup, {nulls} null")

    print("\nForeign keys")
    for spec in args.fk:
        child, parent = spec.split("=")
        ct, cc = child.split(".")
        pt, pc = parent.split(".")
        vals = tables[ct][cc].dropna()
        orphans = (~vals.isin(tables[pt][pc])).sum()
        (ok if orphans == 0 else bad)(f"{child} -> {parent}: {orphans} orphans")

    print("\nNot-null")
    for spec in args.notnull:
        t, c = spec.split(".")
        n = tables[t][c].isna().sum()
        (ok if n == 0 else bad)(f"{spec}: {n} nulls")

    print("\nRanges")
    for spec in args.range:
        target, bounds = spec.split("=")
        t, c = target.split(".")
        lo, hi = bounds.split(":")
        s = pd.to_numeric(tables[t][c], errors="coerce").dropna()
        below = (s < float(lo)).sum() if lo else 0
        above = (s > float(hi)).sum() if hi else 0
        (ok if below == 0 and above == 0 else bad)(f"{target} in [{lo or '-inf'}, {hi or 'inf'}]: {below} below, {above} above")

    print("\nDuplicate rows / null rates")
    declared = {s for s in args.notnull}
    for t, df in tables.items():
        d = df.duplicated().sum()
        (ok if d == 0 else bad)(f"{t}: {d} fully duplicated rows")
        for c in df.columns:
            r = df[c].isna().mean()
            if r > args.max_null_rate and f"{t}.{c}" not in declared:
                warns.append(f"{t}.{c} null rate {r:.1%}")
    for w in warns:
        print(f"  WARN  {w}")

    print(f"\n{len(failures)} failed, {len(warns)} warnings")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
