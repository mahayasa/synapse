"""Metric implementations. Every score is in [0, 1]; higher is better.

Methods (all standard in the synthetic-data literature / SDMetrics-style suites):
  column_shapes      per-column marginal fidelity: 1-KS (numeric), 1-TVD (categorical)
  pair_trends        pairwise relationships: Spearman gap (num-num), joint TVD (others)
  c2st               classifier two-sample test: can a model tell real from synthetic?
  utility (TSTR)     train on Synthetic, Test on Real vs train-on-real baseline
  privacy (DCR)      distance to closest real record + exact-copy rate vs a holdout baseline
  coverage_validity  boundary adherence, range/category coverage, novel categories
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

MAX_PAIR_COLS = 25
MAX_LEVELS = 20
_DT_HINT = ("date", "time", "_ts", "timestamp")


# ---------------------------------------------------------------------------
# preparation
# ---------------------------------------------------------------------------
def is_id_col(name, extra=()) -> bool:
    n = str(name).lower()
    return n == "id" or n.endswith("_id") or n in {str(e).lower() for e in extra}


def _to_cat(series: pd.Series) -> pd.Series:
    def f(v):
        if pd.isna(v):
            return "__missing__"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, float, np.number)):
            return f"{v:g}"
        return str(v)
    return series.map(f)


def prepare(real: pd.DataFrame, synth: pd.DataFrame, exclude=()):
    """Align columns, drop identifier columns, coerce types.

    Identifiers are dropped by name (`id`, `*_id`, or `exclude`) and by shape
    (categorical columns that are >90% unique per row).

    Returns (r, s, kinds) with kinds[col] in {"num", "cat"}. Datetimes become
    numeric seconds; low-cardinality integers and booleans become categorical.
    """
    rd, sd, kinds = {}, {}, {}
    for c in real.columns:
        if c not in synth.columns or is_id_col(c, exclude):
            continue
        rc, sc = real[c].reset_index(drop=True), synth[c].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(rc):
            kind = "cat"
        elif pd.api.types.is_numeric_dtype(rc):
            sc = pd.to_numeric(sc, errors="coerce")
            integer_like = bool((rc.dropna() % 1 == 0).all()) if len(rc.dropna()) else False
            kind = "cat" if (rc.nunique() <= 10 and integer_like) else "num"
        else:
            kind = "cat"
            if any(h in c.lower() for h in _DT_HINT):
                a = pd.to_datetime(rc, errors="coerce", utc=True).dt.tz_localize(None)
                b = pd.to_datetime(sc, errors="coerce", utc=True).dt.tz_localize(None)
                if a.notna().mean() > 0.9 and b.notna().mean() > 0.5:
                    epoch = pd.Timestamp("1970-01-01")
                    rc, sc, kind = (a - epoch).dt.total_seconds(), (b - epoch).dt.total_seconds(), "num"
        if kind == "cat":
            rc, sc = _to_cat(rc), _to_cat(sc)
            if len(rc) > 50 and rc.nunique() > 0.9 * len(rc):
                continue  # identifier-like (e.g. email): unique per row, meaningless to compare

        else:
            rc, sc = rc.astype(float), pd.to_numeric(sc, errors="coerce").astype(float)
        rd[c], sd[c], kinds[c] = rc, sc, kind
    return pd.DataFrame(rd), pd.DataFrame(sd), kinds


def _tvd(p: pd.Series, q: pd.Series) -> float:
    idx = p.index.union(q.index)
    return 0.5 * float(np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())


# ---------------------------------------------------------------------------
# fidelity
# ---------------------------------------------------------------------------
def column_shapes(r, s, kinds) -> dict:
    cols = {}
    for c, k in kinds.items():
        if k == "num":
            a, b = r[c].dropna(), s[c].dropna()
            cols[c] = 0.0 if len(a) == 0 or len(b) == 0 else float(1 - stats.ks_2samp(a, b).statistic)
        else:
            cols[c] = 1.0 - _tvd(r[c].value_counts(normalize=True), s[c].value_counts(normalize=True))
    return {"score": float(np.mean(list(cols.values()))) if cols else None, "columns": cols}


def _discretize(r, s, kinds, c):
    if kinds[c] == "num":
        a = r[c].dropna()
        edges = np.unique(np.quantile(a, [0.2, 0.4, 0.6, 0.8])) if len(a) else np.array([])
        f = lambda x: pd.Series(np.where(x.isna(), -1, np.digitize(x.fillna(0).to_numpy(), edges)))  # noqa: E731
        return f(r[c]), f(s[c])
    top = r[c].value_counts().index[:MAX_LEVELS]
    f = lambda x: x.where(x.isin(top), "__other__")  # noqa: E731
    return f(r[c]), f(s[c])


def pair_trends(r, s, kinds) -> dict:
    cols = list(kinds)[:MAX_PAIR_COLS]
    cache = {}
    pairs = {}
    for a, b in itertools.combinations(cols, 2):
        if kinds[a] == "num" and kinds[b] == "num":
            def rho(df):
                x = df[[a, b]].dropna()
                if len(x) < 3 or x[a].nunique() < 2 or x[b].nunique() < 2:
                    return np.nan
                return stats.spearmanr(x[a], x[b])[0]
            ra, sa = rho(r), rho(s)
            if np.isnan(ra) and np.isnan(sa):
                continue
            pairs[f"{a}~{b}"] = float(1 - abs(np.nan_to_num(ra) - np.nan_to_num(sa)) / 2)
        else:
            for c in (a, b):
                if c not in cache:
                    cache[c] = _discretize(r, s, kinds, c)
            pr = pd.crosstab(cache[a][0], cache[b][0], normalize=True).stack()
            ps = pd.crosstab(cache[a][1], cache[b][1], normalize=True).stack()
            pairs[f"{a}~{b}"] = float(1 - _tvd(pr, ps))
    return {"score": float(np.mean(list(pairs.values()))) if pairs else None, "pairs": pairs}


def c2st(r, s, kinds, seed=0, max_rows=4000) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    n = min(len(r), len(s), max_rows)
    if n < 30 or not kinds:
        return None
    X = pd.concat([r.sample(n, random_state=seed), s.sample(n, random_state=seed)], ignore_index=True)
    for c, k in kinds.items():
        X[c] = pd.Categorical(X[c]).codes if k == "cat" else X[c].astype(float)
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed)
    p = cross_val_predict(clf, X, y, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    auc = float(roc_auc_score(y, p))
    return {"auc": auc, "score": float(1 - max(0.0, 2 * (auc - 0.5)))}


# ---------------------------------------------------------------------------
# coverage / validity
# ---------------------------------------------------------------------------
def coverage_validity(r, s, kinds) -> dict:
    det, vals = {}, []
    for c, k in kinds.items():
        if k == "num":
            a, b = r[c].dropna(), s[c].dropna()
            if len(a) == 0 or len(b) == 0:
                continue
            lo, hi = a.min(), a.max()
            adh = float(((b >= lo) & (b <= hi)).mean())
            cov = 1.0 if hi == lo else float(np.clip((min(b.max(), hi) - max(b.min(), lo)) / (hi - lo), 0, 1))
            det[c] = {"boundary_adherence": adh, "range_coverage": cov}
            vals += [adh, cov]
        else:
            rc = set(r[c].unique())
            cov = len(rc & set(s[c].unique())) / len(rc) if rc else 1.0
            novel = float((~s[c].isin(rc)).mean())
            det[c] = {"category_coverage": cov, "novel_category_rate": novel}
            vals += [cov, 1 - novel]
    return {"score": float(np.mean(vals)) if vals else None, "columns": det}


# ---------------------------------------------------------------------------
# privacy (DCR)
# ---------------------------------------------------------------------------
def _encode(frames, kinds, ref, max_levels=15):
    blocks = [[] for _ in frames]
    for c, k in kinds.items():
        if k == "num":
            mu, sd = ref[c].mean(), ref[c].std()
            mu = mu if np.isfinite(mu) else 0.0
            sd = sd if np.isfinite(sd) and sd > 0 else 1.0
            for i, f in enumerate(frames):
                blocks[i].append(((f[c] - mu) / sd).fillna(0).clip(-5, 5).to_numpy()[:, None])
        elif ref[c].nunique() <= max_levels:
            levels = np.asarray(ref[c].value_counts().index)
            for i, f in enumerate(frames):
                blocks[i].append((f[c].to_numpy()[:, None] == levels[None, :]).astype(float))
    return [np.hstack(b) if b else np.zeros((len(f), 0)) for b, f in zip(blocks, frames)]


def _exact_rate(a: pd.DataFrame, b: pd.DataFrame) -> float:
    m = a.merge(b.drop_duplicates(), how="left", indicator=True)
    return float((m["_merge"] == "both").mean())


def privacy(r, s, kinds, seed=0, max_rows=3000) -> dict | None:
    """Synthetic rows should be no closer to real training rows than unseen real rows are."""
    from sklearn.neighbors import NearestNeighbors
    if len(r) < 100 or not kinds:
        return None
    idx = np.random.default_rng(seed).permutation(len(r))
    k = int(len(r) * 0.8)
    tr = r.iloc[idx[:k]].reset_index(drop=True)
    ho = r.iloc[idx[k:]].reset_index(drop=True).head(max_rows)
    ss = s.sample(min(len(s), max_rows), random_state=seed).reset_index(drop=True)
    Xtr, Xs, Xh = _encode([tr, ss, ho], kinds, tr)
    if Xtr.shape[1] == 0:
        return None
    nn = NearestNeighbors(n_neighbors=1).fit(Xtr)
    ds, dh = nn.kneighbors(Xs)[0][:, 0], nn.kneighbors(Xh)[0][:, 0]
    ms, mh = float(np.median(ds)), float(np.median(dh))
    ratio = 1.0 if mh == 0 else min(1.0, ms / mh)
    ex_s, ex_h = _exact_rate(ss, tr), _exact_rate(ho, tr)
    excess = max(0.0, ex_s - ex_h)
    return {"score": float(ratio * (1 - min(1.0, excess))), "median_dcr_synthetic": ms,
            "median_dcr_holdout": mh, "exact_match_rate_synthetic": ex_s,
            "exact_match_rate_holdout_baseline": ex_h}


# ---------------------------------------------------------------------------
# utility (TSTR)
# ---------------------------------------------------------------------------
def utility(r, s, kinds, cfg, seed=0) -> dict | None:
    """Train on Synthetic / Test on Real, relative to a train-on-real baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import f1_score, r2_score, roc_auc_score
    target = cfg.get("target")
    if target not in kinds or len(r) < 100:
        return None
    task = cfg.get("task") or ("regression" if kinds[target] == "num" else "classification")
    feats = [c for c in kinds if c != target and (not cfg.get("features") or c in cfg["features"])]
    if not feats:
        return None
    cats = {c: sorted(set(r[c]) | set(s[c])) for c in feats if kinds[c] == "cat"}

    def X_of(df):
        return pd.DataFrame({c: (pd.Categorical(df[c], categories=cats[c]).codes if c in cats
                                 else df[c].astype(float)) for c in feats})

    idx = np.random.default_rng(seed).permutation(len(r))
    cut = int(len(r) * 0.7)
    tr, te = r.iloc[idx[:cut]], r.iloc[idx[cut:]]
    mk = (lambda: HistGradientBoostingRegressor(max_iter=100, random_state=seed)) if task == "regression" \
        else (lambda: HistGradientBoostingClassifier(max_iter=100, random_state=seed))
    y = (lambda d: d[target].astype(float)) if task == "regression" else (lambda d: d[target].astype(str))
    if task != "regression" and s[target].nunique() < 2:
        return {"score": 0.0, "note": "synthetic target has a single class"}
    m_real, m_syn = mk().fit(X_of(tr), y(tr)), mk().fit(X_of(s), y(s))
    Xte, yte = X_of(te), y(te)

    if task == "regression":
        metric = "r2"
        a, b = r2_score(yte, m_real.predict(Xte)), r2_score(yte, m_syn.predict(Xte))
    else:
        same = set(m_real.classes_) == set(m_syn.classes_) == set(yte) and len(set(yte)) == 2
        if same:
            metric = "gini"
            pos = m_real.classes_[1]
            a = 2 * roc_auc_score(yte == pos, m_real.predict_proba(Xte)[:, 1]) - 1
            b = 2 * roc_auc_score(yte == pos, m_syn.predict_proba(Xte)[:, list(m_syn.classes_).index(pos)]) - 1
        else:
            metric = "macro_f1"
            a, b = (f1_score(yte, m.predict(Xte), average="macro") for m in (m_real, m_syn))
    score = float(np.clip(b / a, 0, 1)) if a > 0.05 else None
    return {"score": score, "metric": metric, "train_real_test_real": float(a),
            "train_synth_test_real": float(b), "task": task, "target": target}
