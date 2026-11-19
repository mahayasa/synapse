"""Conventional ML backend (SDV): Gaussian copula, CTGAN, TVAE, with a train/sample split.

  train()          fit on REAL data, save models + schema + training log to `model_dir`
  sample_tables()  generate from a saved model (no real data needed at sample time)

Design:
  * Key columns (primary keys, foreign keys) are NOT modelled. They are rebuilt after
    sampling: PKs in the real ID format, FKs drawn from the parent's synthetic keys using
    the real per-parent child-count distribution (keeps skew). So PK uniqueness and FK
    integrity hold by construction.
  * `derived` expressions (e.g. orders.total: "qty * unit_price") are recomputed after
    sampling, since GANs do not preserve arithmetic relationships.
  * `holdout_fraction` of each real table is held out; after training the model is scored
    on it with the evaluator and the result is stored in the model dir (holdout copies are
    deleted afterwards).

SECURITY: saved models are pickles. Only load models you trained yourself.
Not byte-reproducible: GAN training varies run to run even with a seed.
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .base import Backend, ConfigError

ALGOS = {"gaussian_copula": "GaussianCopulaSynthesizer", "ctgan": "CTGANSynthesizer",
         "tvae": "TVAESynthesizer"}


def _synth_class(algo: str):
    if algo not in ALGOS:
        raise ConfigError(f"unknown algorithm {algo!r} ({' | '.join(ALGOS)})")
    try:
        import sdv.single_table as st
    except ImportError as e:
        raise ConfigError("pip install sdv") from e
    return getattr(st, ALGOS[algo])


def _metadata(df: pd.DataFrame, name: str):
    """Detect metadata across SDV API generations (verified against sdv 2.2.0)."""
    from sdv.metadata import Metadata
    if hasattr(Metadata, "detect_from_dataframes"):           # sdv >= 2
        return Metadata.detect_from_dataframes({name: df})
    if hasattr(Metadata, "detect_from_dataframe"):            # sdv 1.x late
        return Metadata.detect_from_dataframe(df, table_name=name)
    from sdv.metadata import SingleTableMetadata              # sdv 1.x early
    m = SingleTableMetadata()
    m.detect_from_dataframe(df)
    return m


def _sdv_major() -> int:
    import sdv
    return int(sdv.__version__.split(".")[0])


def _fit(synth, df: pd.DataFrame, name: str) -> None:
    synth.fit({name: df} if _sdv_major() >= 2 else df)          # sdv 2.x takes {table: df}


def _sample(synth, n: int, name: str) -> pd.DataFrame:
    out = synth.sample(num_rows=n)
    return (out[name] if isinstance(out, dict) else out).reset_index(drop=True)   # 2.x returns a dict


def _load(path: str, algo: str):
    try:
        from sdv.utils import load_synthesizer
        return load_synthesizer(path)
    except ImportError:
        return _synth_class(algo).load(path)


def _loss_frame(synth) -> pd.DataFrame | None:
    lv = synth.get_loss_values()                                   # CTGAN / TVAE only
    if isinstance(lv, dict):
        lv = next(iter(lv.values()))
    return lv


def _seed_everything(seed: int) -> str:
    np.random.seed(seed)
    device = "cpu"
    try:
        import torch
        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass
    return device


def _parse_fks(conv: dict) -> list[tuple[str, str, str, str]]:
    out = []
    for spec in conv.get("foreign_keys", []):
        try:
            child, parent = spec.split("=")
            (ct, cc), (pt, pc) = child.split("."), parent.split(".")
        except ValueError as e:
            raise ConfigError(f"bad foreign_keys entry {spec!r}; use child.col=parent.col") from e
        out.append((ct, cc, pt, pc))
    return out


def _key_format(series: pd.Series) -> dict:
    ms = [re.fullmatch(r"(\D*)(\d+)", str(v)) for v in series.head(50)]
    if ms and all(ms) and len({m.group(1) for m in ms}) == 1:
        return {"kind": "prefixed", "prefix": ms[0].group(1), "width": max(len(m.group(2)) for m in ms)}
    if pd.api.types.is_integer_dtype(series):
        return {"kind": "int"}
    return {"kind": "prefixed", "prefix": "K", "width": 7}


def _make_keys(fmt: dict, n: int) -> list:
    if fmt["kind"] == "int":
        return list(range(1, n + 1))
    return [f"{fmt['prefix']}{i:0{fmt['width']}d}" for i in range(1, n + 1)]


def _order(tables: list[str], fks) -> list[str]:
    done, left = [], list(tables)
    while left:
        ready = [t for t in left if all(pt in done or pt == t or pt not in tables
                                        for ct, _, pt, _ in fks if ct == t)]
        if not ready:
            raise ConfigError("foreign_keys contain a cycle between tables")
        done += ready
        left = [t for t in left if t not in ready]
    return done


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train(cfg: dict, model_dir: Path) -> dict:
    conv = cfg.get("conventional") or {}
    algo = conv.get("algorithm", "gaussian_copula")
    cls = _synth_class(algo)
    real = conv.get("real_data")
    if not real or not Path(real).is_dir():
        raise ConfigError("training needs conventional.real_data (a folder of real CSVs). "
                          "Conventional models cannot learn from a description alone.")
    files = sorted(Path(real).glob("*.csv"))
    if not files:
        raise ConfigError(f"no CSV files in {real}")
    seed = int(cfg.get("seed", 42))
    device = _seed_everything(seed)
    rng = np.random.default_rng(seed)
    data = {f.stem: pd.read_csv(f) for f in files}
    pks, fks = conv.get("primary_keys", {}), _parse_fks(conv)
    for ct, cc, pt, pc in fks:
        for t, c in ((ct, cc), (pt, pc)):
            if t not in data or c not in data[t].columns:
                raise ConfigError(f"foreign key refers to missing column {t}.{c}")
    frac = float(conv.get("holdout_fraction", 0.2))
    params = conv.get("params") or {}
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    schema = {"algorithm": algo, "seed": seed, "derived": conv.get("derived", {}), "tables": {}, "fks": [],
              "scale_tables": conv.get("scale_tables", [])}
    for ct, cc, pt, pc in fks:
        per_parent = data[ct][cc].value_counts().reindex(data[pt][pc].unique(), fill_value=0)
        schema["fks"].append({"child_table": ct, "child_col": cc, "parent_table": pt, "parent_col": pc,
                              "counts": per_parent.astype(int).tolist()})
    t0, loss_log, hold = time.time(), {}, {}
    for name, df in data.items():
        keys = ([pks[name]] if name in pks else []) + [cc for ct, cc, _, _ in fks if ct == name]
        fit_cols = [c for c in df.columns if c not in keys]
        if not fit_cols:
            raise ConfigError(f"table {name!r} has no non-key columns to model")
        perm = rng.permutation(len(df))
        n_hold = int(len(df) * frac) if frac > 0 and len(df) >= 200 else 0
        train_df = df.iloc[perm[n_hold:]][fit_cols].reset_index(drop=True)
        if n_hold:
            hold[name] = df.iloc[perm[:n_hold]].reset_index(drop=True)
        synth = cls(_metadata(train_df, name), **params)
        _fit(synth, train_df, name)
        synth.save(str(model_dir / f"{name}.pkl"))
        try:
            lv = _loss_frame(synth)
            loss_log[name] = {c: [round(float(x), 4) for x in lv[c]] for c in lv.columns}
        except Exception:
            pass
        schema["tables"][name] = {"columns": list(df.columns), "fit_columns": fit_cols,
                                  "pk": pks.get(name), "real_rows": len(df),
                                  "pk_format": _key_format(df[pks[name]]) if name in pks else None}
    (model_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    log = {"algorithm": algo, "params": params, "seed": seed, "device": device,
           "train_seconds": round(time.time() - t0, 1),
           "rows_real": {t: len(d) for t, d in data.items()},
           "rows_held_out": {t: len(d) for t, d in hold.items()}, "loss": loss_log,
           "reproducible": False}
    if hold:
        log["holdout_evaluation"] = _holdout_eval(model_dir, hold, cfg, seed)
    (model_dir / "training_log.json").write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    return log


def _holdout_eval(model_dir: Path, hold: dict, cfg: dict, seed: int) -> dict:
    from evaluation import evaluate_dataset, to_markdown
    ev = dict(cfg.get("evaluation") or {})
    ev.pop("real_data", None)
    with tempfile.TemporaryDirectory() as tmp:
        hd, sd = Path(tmp) / "holdout", Path(tmp) / "sample"
        hd.mkdir(), sd.mkdir()
        for t, df in hold.items():
            df.to_csv(hd / f"{t}.csv", index=False)
        for t, df in sample_tables(model_dir, sizes={t: len(d) for t, d in hold.items()}, seed=seed).items():
            if t in hold:
                df.to_csv(sd / f"{t}.csv", index=False)
        rep = evaluate_dataset(hd, sd, ev, None)   # no integrity checks: holdout may be a partial schema
    (model_dir / "holdout_evaluation.md").write_text(
        to_markdown(rep, "Held-out evaluation (real holdout vs model samples)"), encoding="utf-8")
    return {"composite": rep["composite"], **rep["groups"]}   # holdout copies deleted with tmp


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
def sample_tables(model_dir: Path, rows: int | None = None, sizes: dict | None = None,
                  seed: int = 42) -> dict[str, pd.DataFrame]:
    model_dir = Path(model_dir)
    schema = json.loads((model_dir / "schema.json").read_text(encoding="utf-8"))
    algo = schema["algorithm"]
    _synth_class(algo)                                             # validates the algorithm / sdv install
    _seed_everything(seed)
    rng = np.random.default_rng(seed)
    tables = schema["tables"]
    fks = [(f["child_table"], f["child_col"], f["parent_table"], f["parent_col"]) for f in schema["fks"]]
    parents = {pt for _, _, pt, _ in fks}
    scaled = set(schema.get("scale_tables", []))
    growing = [t["real_rows"] for n, t in tables.items() if n not in parents or n in scaled]
    scale = (rows / max(growing)) if rows else 1.0            # `rows` = size of the largest fact table
    out: dict[str, pd.DataFrame] = {}
    for name in _order(list(tables), fks):
        t = tables[name]
        # Fact tables scale with `rows`; parent/dimension tables keep their real size unless
        # listed in conventional.scale_tables.
        grows = name not in parents or name in scaled
        n = (sizes or {}).get(name) or max(1, round(t["real_rows"] * (scale if grows else 1.0)))
        df = _sample(_load(str(model_dir / f"{name}.pkl"), algo), n, name)
        if t["pk"]:
            df[t["pk"]] = _make_keys(t["pk_format"], n)
        for f in schema["fks"]:
            if f["child_table"] != name:
                continue
            keys = out[f["parent_table"]][f["parent_col"]].to_numpy()
            counts = np.asarray(f["counts"], dtype=float)
            w = rng.choice(counts, size=len(keys)) if len(counts) else np.ones(len(keys))
            w = w if w.sum() > 0 else np.ones(len(keys))
            df[f["child_col"]] = rng.choice(keys, size=n, p=w / w.sum())
        for target, expr in schema["derived"].items():
            tn, col = target.split(".")
            if tn == name:
                v = df.eval(expr)
                df[col] = v.round(2) if pd.api.types.is_float_dtype(v) else v
        out[name] = df[t["columns"]]
    return out


class SDVBackend(Backend):
    name = "sdv"

    def label(self):
        return f"sdv_{self.cfg.get('conventional', {}).get('algorithm', 'gaussian_copula')}"

    def generate(self, out_dir: Path) -> dict:
        conv = self.cfg.get("conventional") or {}
        model_dir = Path(conv.get("model_dir") or (Path(out_dir) / "model")).resolve()
        trained_now = False
        if not (model_dir / "schema.json").exists():
            train(self.cfg, model_dir)
            trained_now = True
        tables = sample_tables(model_dir, rows=self.cfg.get("rows"), seed=int(self.cfg.get("seed", 42)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for t, df in tables.items():
            df.to_csv(Path(out_dir) / f"{t}.csv", index=False)
        log = json.loads((model_dir / "training_log.json").read_text(encoding="utf-8"))
        return {"model": log["algorithm"], "model_dir": str(model_dir), "trained_this_run": trained_now,
                "tables": {t: len(d) for t, d in tables.items()}, "reproducible": False,
                "holdout_evaluation": log.get("holdout_evaluation"), "device": log.get("device")}
