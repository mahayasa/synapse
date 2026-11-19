"""Few-shot context for LLM backends: show the model what the REAL data looks like.

Two modes (config `few_shot`):
  stats  (default) per-column summaries: types, ranges, quantiles, top categories.
  rows             a few real example rows per table.

PRIVACY: whatever this builds is sent to the model provider. Aggregates still reveal
something about the real data; raw rows reveal much more. `rows` mode therefore
requires `acknowledge_external_data: true` unless the endpoint is local
(localhost / 127.0.0.1). Use `exclude_columns` to drop sensitive columns.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .base import ConfigError

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


def _summarize(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    lines = []
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c]
        nulls = s.isna().mean()
        tag = f"nulls {nulls:.1%}" if nulls else "no nulls"
        if pd.api.types.is_bool_dtype(s):
            lines.append(f"- {c} (bool): true {s.mean():.1%}, {tag}")
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
    if not real or not Path(real).is_dir():
        raise ConfigError("few_shot.real_data must point to a folder of real CSVs")
    mode = fs.get("mode", "stats")
    if mode not in ("stats", "rows"):
        raise ConfigError("few_shot.mode must be 'stats' or 'rows'")
    if mode == "rows" and not (fs.get("acknowledge_external_data") or _is_local(cfg)):
        raise ConfigError("few_shot.mode 'rows' sends real rows to the model provider. Set "
                          "few_shot.acknowledge_external_data: true to confirm, or use a local base_url.")
    files = sorted(Path(real).glob("*.csv"))
    wanted = fs.get("tables")
    if wanted:
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise ConfigError(f"no CSV tables found in {real}")
    exclude = set(fs.get("exclude_columns", []))
    n = int(fs.get("rows_per_table", 5))
    seed = int(fs.get("seed", 0))

    parts = ["Real-data reference. Match these distributions, formats and relationships in the "
             "synthetic data. Do NOT copy rows verbatim."]
    for f in files:
        df = pd.read_csv(f)
        parts.append(f"\nTable `{f.stem}` ({len(df):,} real rows; columns: {', '.join(c for c in df.columns if c not in exclude)})")
        parts += _summarize(df, exclude)
        if mode == "rows":
            ex = df.drop(columns=list(exclude & set(df.columns))).sample(min(n, len(df)), random_state=seed)
            parts.append("Example rows:\n" + ex.to_csv(index=False).strip())
    text = "\n".join(parts)
    limit = int(fs.get("max_chars", 6000))
    return text if len(text) <= limit else text[:limit] + "\n[reference truncated]"
