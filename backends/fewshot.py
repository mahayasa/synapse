"""Few-shot context for LLM backends: show the model what the REAL data looks like.

Two modes (config `few_shot`):
  stats  (default) per-column summaries: types, ranges, quantiles, top categories.
  rows             a few real example rows per table.

`real_data` may be a folder or a single file; CSV, TSV, pasted-text and XLSX are supported
(see dataio.py). Each XLSX sheet becomes a table.

PRIVACY: whatever this builds is sent to the model provider. Aggregates still reveal
something about the real data; raw rows reveal much more. `rows` mode therefore
requires `acknowledge_external_data: true` unless the endpoint is local
(localhost / 127.0.0.1). Use `exclude_columns` to drop sensitive columns.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd

from .base import ConfigError
from .dataio import read_tables

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _is_local(cfg: dict) -> bool:
    url = cfg.get("base_url")
    return bool(url) and (urlparse(url).hostname or "") in LOCAL_HOSTS


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _mask(v: str) -> str:
    """Show an identifier's shape without revealing it: digits -> 9, letters -> a/A."""
    return re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", re.sub(r"\d", "9", v)))


def summarize_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """One line per column: type, range or top categories, missingness. Identifier-like text is masked."""
    lines = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        nulls = s.isna().mean()
        tag = f"nulls {nulls:.1%}" if nulls else "no nulls"
        if pd.api.types.is_bool_dtype(s):
            lines.append(f"- {c} (bool): true {s.mean():.1%}, {tag}")
        elif (pd.api.types.is_numeric_dtype(s) and 1 < s.nunique() <= 10
              and bool((s.dropna() % 1 == 0).all())):
            top = s.value_counts(normalize=True).sort_index()
            lines.append(f"- {c} (flag/small category, {s.nunique()} values): " +
                         ", ".join(f"{k:g}={v:.0%}" for k, v in top.items()) + f", {tag}")
        elif pd.api.types.is_numeric_dtype(s):
            q = s.dropna().quantile([0, .25, .5, .75, 1]).tolist() if s.notna().any() else []
            lines.append(f"- {c} (number): min/p25/median/p75/max = {' / '.join(_fmt(x) for x in q)}, {tag}")
        else:
            nun = s.nunique()
            if nun > 0.9 * max(len(s), 1) and len(s) > 20:
                ex = _mask(str(s.dropna().iloc[0])) if s.notna().any() else ""
                lines.append(f"- {c} (identifier-like text, masked format {ex}), {tag}")
            else:
                top = s.value_counts(normalize=True).head(8)
                lines.append(f"- {c} (category, {nun} values): " +
                             ", ".join(f"{k}={v:.0%}" for k, v in top.items()) + f", {tag}")
    return lines


def build_few_shot(cfg: dict) -> str:
    fs = cfg.get("few_shot")
    if not fs:
        return ""
    real = fs.get("real_data")
    if not real:
        raise ConfigError("few_shot.real_data must point to a folder or file of example data")
    mode = fs.get("mode", "stats")
    if mode not in ("stats", "rows"):
        raise ConfigError("few_shot.mode must be 'stats' or 'rows'")
    if mode == "rows" and not (fs.get("acknowledge_external_data") or _is_local(cfg)):
        raise ConfigError("few_shot.mode 'rows' sends real rows to the model provider. Set "
                          "few_shot.acknowledge_external_data: true to confirm, or use a local base_url.")
    tables = read_tables(real)
    wanted = fs.get("tables")
    if wanted:
        tables = {k: v for k, v in tables.items() if k in wanted}
    if not tables:
        raise ConfigError(f"no tables found in {real}")
    exclude = set(fs.get("exclude_columns", []))
    n = int(fs.get("rows_per_table", 5))
    seed = int(fs.get("seed", 0))

    parts = ["Real-data reference. Match these distributions, formats and relationships in the "
             "synthetic data. Do NOT copy rows verbatim."]
    for name, df in tables.items():
        parts.append(f"\nTable `{name}` ({len(df):,} real rows; columns: "
                     f"{', '.join(c for c in df.columns if c not in exclude)})")
        parts += summarize_columns(df, exclude)
        if mode == "rows":
            ex = df.drop(columns=list(exclude & set(df.columns))).sample(min(n, len(df)), random_state=seed)
            parts.append("Example rows:\n" + ex.to_csv(index=False).strip())
    text = "\n".join(parts)
    limit = int(fs.get("max_chars", 6000))
    return text if len(text) <= limit else text[:limit] + "\n[reference truncated]"
