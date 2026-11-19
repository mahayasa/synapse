#!/usr/bin/env python3
"""Small synthetic diabetes classification table (Pima-style), following references/health_ehr.md.

    python generate_diabetes.py --seed 42 --rows 500 --out data

Design: a latent risk (age, BMI, family history, smoking) sets P(diabetes); the label is drawn from it,
then glucose / HbA1c / blood pressure / triglycerides are generated conditional on the label (with
overlap), so label and features are related but not deterministic. All data is synthetic.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CFG = dict(
    prevalence=0.15,          # target share of outcome=1
    frac_female=0.52,
    age_mean=47, age_sd=15,
    smoker_rate=0.20,
    anomaly_rate=0.02,        # entry errors, flagged in is_anomaly / anomaly_type
    prediabetes_share=0.25,   # share of non-diabetics with raised glucose (feature overlap)
    controlled_share=0.30,    # share of diabetics on treatment with near-normal glucose
    # P(HbA1c missing) by fasting glucose band: tested less when low-risk (informative missingness)
    hba1c_missing=dict(low=0.50, mid=0.20, high=0.05),
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calibrate_intercept(score, target, iters=50):
    """Bisection so that mean(sigmoid(c + score)) == target prevalence."""
    lo, hi = -15.0, 15.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if sigmoid(mid + score).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def generate(n: int, seed: int, cfg: dict = CFG) -> pd.DataFrame:
    rng = np.random.default_rng(seed)             # single seeded generator, no global state
    age = np.clip(rng.normal(cfg["age_mean"], cfg["age_sd"], n), 18, 90).round().astype(int)
    sex = np.where(rng.random(n) < cfg["frac_female"], "F", "M")
    smoker = (rng.random(n) < cfg["smoker_rate"]).astype(int)
    family = rng.beta(1.2, 3.0, n).round(2)                       # family-history score, 0-1
    bmi = np.exp(rng.normal(np.log(27), 0.2, n)) * (1 + 0.003 * (age - 47))
    bmi = np.clip(bmi, 15, 60).round(1)

    # latent risk -> disease label
    score = 0.04 * (age - 47) + 0.12 * (bmi - 27) + 1.5 * (family - 0.29) + 0.3 * smoker + rng.normal(0, 0.8, n)
    p = sigmoid(calibrate_intercept(score, cfg["prevalence"]) + score)
    outcome = (rng.random(n) < p).astype(int)

    # features conditional on the label (overlapping distributions)
    base = 92 + 0.6 * (bmi - 27) + 0.15 * (age - 47)
    healthy = rng.normal(base, 11)
    prediab = rng.random(n) < cfg["prediabetes_share"]            # undiagnosed / borderline: label 0, glucose raised
    healthy = healthy + np.where(prediab, rng.uniform(10, 40, n), 0)
    diabetic = 100 + rng.gamma(2.5, 16, n) + 0.4 * (bmi - 27)
    controlled = (outcome == 1) & (rng.random(n) < cfg["controlled_share"])   # treated: label 1, glucose near-normal
    diabetic = np.where(controlled, healthy + rng.uniform(10, 30, n), diabetic)
    glucose = np.clip(np.where(outcome == 1, diabetic, healthy), 60, 400).round().astype(int)

    avg_glucose = glucose * 1.15 + rng.normal(0, 12, n)           # HbA1c ~ (eAG + 46.7) / 28.7
    hba1c = np.clip((avg_glucose + 46.7) / 28.7, 4.0, 14.0).round(1)
    miss_p = np.where(glucose < 100, cfg["hba1c_missing"]["low"],
                      np.where(glucose < 126, cfg["hba1c_missing"]["mid"], cfg["hba1c_missing"]["high"]))
    hba1c = np.where(rng.random(n) < miss_p, np.nan, hba1c)

    sbp = 118 + 0.35 * (age - 47) + 0.45 * (bmi - 27) + 4 * outcome + 3 * smoker + rng.normal(0, 10, n)
    sbp = np.clip(sbp, 85, 220).round().astype(int)
    dbp = 0.45 * sbp + 24 + rng.normal(0, 6, n)
    dbp = np.clip(np.minimum(dbp, sbp - 20), 50, 120).round().astype(int)
    tg = np.exp(rng.normal(np.log(120) + 0.012 * (bmi - 27) + 0.15 * outcome, 0.4, n))
    tg = np.clip(tg, 40, 800).round().astype(int)
    preg = np.where(sex == "F", np.clip(rng.poisson(np.maximum(0, (age - 20) / 12)), 0, 12), 0)

    df = pd.DataFrame({
        "patient_id": [f"P{i:05d}" for i in range(1, n + 1)], "age": age, "sex": sex,
        "pregnancies": preg, "bmi": bmi, "family_history": family, "smoker": smoker,
        "sbp": sbp, "dbp": dbp, "fasting_glucose": glucose, "hba1c": hba1c,
        "triglycerides": tg, "outcome": outcome,
    })

    # flagged entry errors that stay inside plausible ranges
    df["is_anomaly"], df["anomaly_type"] = False, ""
    k = int(n * cfg["anomaly_rate"])
    idx = rng.choice(n, size=k, replace=False)
    for j, i in enumerate(idx):
        if j % 2 == 0:                                   # weight entered in lb instead of kg
            df.loc[i, "bmi"] = min(round(df.loc[i, "bmi"] * 2.2, 1), 69.0)
            df.loc[i, "anomaly_type"] = "bmi_unit_error"
        else:                                            # systolic and diastolic swapped
            df.loc[i, ["sbp", "dbp"]] = df.loc[i, ["dbp", "sbp"]].to_numpy()
            df.loc[i, "anomaly_type"] = "bp_swapped"
        df.loc[i, "is_anomaly"] = True
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(a.rows, a.seed)
    df.to_csv(out / "diabetes.csv", index=False)
    print(f"diabetes: {len(df):,} rows, prevalence {df.outcome.mean():.1%}, seed={a.seed} -> {out.resolve()}")


if __name__ == "__main__":
    main()
