#!/usr/bin/env python3
"""Seeded synthetic data generator skeleton.

Copy this file, rename entities, and edit the DOMAIN section. Everything else
(seeding, calendar, seasonality, skewed assignment, anomaly injection, output)
is reusable.

    python generator_template.py --seed 42 --orders 10000 --out out
    python validate.py --dir out --pk customers=customer_id products=product_id \
        orders=order_id --fk orders.customer_id=customers.customer_id \
        orders.product_id=products.product_id --range orders.total=0:1000000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG: every realism knob is a named parameter
# ----------------------------------------------------------------------------
CONFIG = dict(
    n_customers=1_000,
    n_products=200,
    start_date="2025-01-01",
    end_date="2025-12-31",          # fixed, never datetime.now()
    weekday_factors=[1.0, 1.0, 1.0, 1.05, 1.2, 1.4, 0.8],  # Mon..Sun
    yearly_amplitude=0.25,          # seasonal swing (fraction of base)
    peak_month=12,                  # month of seasonal peak
    annual_growth=0.15,             # linear trend over the range
    customer_skew=1.2,              # Zipf exponent: higher = heavier top customers
    price_sigma=0.8,                # lognormal sigma for product prices
    anomaly_rate=0.005,             # share of fact rows with injected anomalies
    null_rate_email=0.03,           # intentional missingness
)


# ----------------------------------------------------------------------------
# REUSABLE HELPERS
# ----------------------------------------------------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def daily_rates(dates: pd.DatetimeIndex, base: float, cfg: dict) -> np.ndarray:
    """Expected events per day = base * weekday * yearly season * growth."""
    weekday = np.array(cfg["weekday_factors"])[np.asarray(dates.dayofweek)]
    phase = 2 * np.pi * (np.asarray(dates.dayofyear) - _peak_doy(cfg["peak_month"])) / 365.25
    season = 1 + cfg["yearly_amplitude"] * np.cos(phase)
    t = np.linspace(0, 1, len(dates))
    trend = 1 + cfg["annual_growth"] * t
    return base * weekday * season * trend


def _peak_doy(month: int) -> int:
    return int((month - 1) * 30.4 + 15)


def zipf_weights(n: int, s: float) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** s
    return w / w.sum()


def inject_anomalies(df: pd.DataFrame, rng, rate: float, col: str, mult=(5, 20)) -> pd.DataFrame:
    """Multiply `col` by a large factor on a random subset and flag the rows."""
    df = df.copy()
    df["is_anomaly"] = False
    k = int(len(df) * rate)
    if k:
        idx = rng.choice(df.index.to_numpy(), size=k, replace=False)
        df.loc[idx, col] = (df.loc[idx, col] * rng.uniform(*mult, size=k)).round(2)
        df.loc[idx, "is_anomaly"] = True
    return df


def mask_nulls(series: pd.Series, rng, rate: float) -> pd.Series:
    s = series.copy().astype(object)
    s[rng.random(len(s)) < rate] = None
    return s


# ----------------------------------------------------------------------------
# DOMAIN: edit below. Dimensions first, then facts derived from them.
# ----------------------------------------------------------------------------
def gen_customers(rng, cfg) -> pd.DataFrame:
    n = cfg["n_customers"]
    ids = [f"C{i:06d}" for i in range(1, n + 1)]
    df = pd.DataFrame({
        "customer_id": ids,
        "email": [f"user{i}@example.com" for i in range(1, n + 1)],
        "signup_date": pd.to_datetime(cfg["start_date"])
        - pd.to_timedelta(rng.integers(0, 730, n), unit="D"),
        "segment": rng.choice(["retail", "member", "wholesale"], n, p=[0.7, 0.25, 0.05]),
    })
    df["email"] = mask_nulls(df["email"], rng, cfg["null_rate_email"])
    return df


def gen_products(rng, cfg) -> pd.DataFrame:
    n = cfg["n_products"]
    cats = rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1])
    price = np.exp(rng.normal(3.0, cfg["price_sigma"], n)).round(2).clip(0.5, 5000)
    return pd.DataFrame({
        "product_id": [f"P{i:05d}" for i in range(1, n + 1)],
        "category": cats,
        "unit_price": price,
    })


def gen_orders(rng, cfg, customers, products, n_orders: int) -> pd.DataFrame:
    dates = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D")
    rates = daily_rates(dates, 1.0, cfg)
    p_day = rates / rates.sum()
    order_dates = rng.choice(dates, size=n_orders, p=p_day)

    # Skewed customer activity (few heavy buyers, long tail)
    cust_idx = rng.choice(len(customers), size=n_orders,
                          p=zipf_weights(len(customers), cfg["customer_skew"]))
    # Product popularity also skewed, milder
    prod_idx = rng.choice(len(products), size=n_orders,
                          p=zipf_weights(len(products), 0.8))

    qty = rng.geometric(0.6, n_orders).clip(1, 20)
    unit_price = products["unit_price"].to_numpy()[prod_idx]
    df = pd.DataFrame({
        "order_id": [f"O{i:08d}" for i in range(1, n_orders + 1)],
        "order_date": order_dates,
        "customer_id": customers["customer_id"].to_numpy()[cust_idx],
        "product_id": products["product_id"].to_numpy()[prod_idx],
        "qty": qty,
        "unit_price": unit_price,
    })
    df["total"] = (df["qty"] * df["unit_price"]).round(2)   # derived, not sampled
    df = df.sort_values("order_date").reset_index(drop=True)
    df["order_id"] = [f"O{i:08d}" for i in range(1, len(df) + 1)]
    return inject_anomalies(df, rng, cfg["anomaly_rate"], "total")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--orders", type=int, default=10_000)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    rng = make_rng(args.seed)
    cfg = CONFIG
    customers = gen_customers(rng, cfg)
    products = gen_products(rng, cfg)
    orders = gen_orders(rng, cfg, customers, products, args.orders)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in {"customers": customers, "products": products, "orders": orders}.items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(df):,} rows")
    print(f"seed={args.seed} -> {out.resolve()}")


if __name__ == "__main__":
    main()
