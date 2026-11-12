"""Generate a SYNTHETIC stand-in for Kaggle's "Give Me Some Credit" training file.

Why this exists: it lets anyone run and test the whole pipeline without a Kaggle
login. It copies the *schema* and the known *data quirks* of the real file
(missing income, 96/98 delinquency codes, utilisation above 1, an age of 0,
debt ratios that explode when income is missing). It is NOT real data, so
metrics produced from it must never be quoted as real results.

    python -m src.make_synthetic --out data/synthetic_training.csv
"""
import argparse

import numpy as np
import pandas as pd


def make(n: int = 150_000, seed: int = 42, target_rate: float = 0.067) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(52, 14.5, n), 21, 100).round()
    z = rng.normal(0, 1, n)  # latent "financial stress" that drives several features

    util = rng.beta(0.6, 1.5, n) + 0.12 * np.maximum(z, 0)
    outlier = rng.random(n) < 0.003  # real file has utilisation values in the thousands
    util[outlier] = rng.uniform(2, 5000, outlier.sum())

    late30 = np.minimum(rng.poisson(0.08 * np.exp(0.9 * z)), 20)
    late60 = np.minimum(rng.poisson(0.03 * np.exp(1.0 * z)), 20)
    late90 = np.minimum(rng.poisson(0.03 * np.exp(1.1 * z)), 20)

    income = np.exp(rng.normal(8.6, 0.65, n)).round()
    income_missing = rng.random(n) < 0.198
    debt_ratio = np.exp(rng.normal(-1.0, 0.9, n))
    # In the real file, when income is missing DebtRatio looks like an absolute amount.
    debt_ratio[income_missing] = np.exp(rng.normal(6.5, 1.0, income_missing.sum())).round()

    open_lines = np.maximum(rng.poisson(8, n) - (z > 1.5).astype(int), 0)
    real_estate = rng.poisson(1.0, n)
    dependents = rng.poisson(0.75, n).astype(float)
    dependents[rng.random(n) < 0.026] = np.nan

    # risk used only to draw the target (keeps the signal realistic but not perfect)
    risk = (
        -0.035 * (age - 52)
        + 1.6 * np.clip(util, 0, 1.5)
        + 0.45 * np.minimum(late30, 6)
        + 0.55 * np.minimum(late60, 6)
        + 0.70 * np.minimum(late90, 6)
        + 0.5 * ((debt_ratio > 1) & ~income_missing)
        - 0.25 * np.where(income_missing, 0, np.log(income / 5400))
        + 0.03 * (open_lines - 8)
        + 0.5 * rng.normal(0, 1, n)  # unobserved heterogeneity
    )

    # 96 / 98 are the "error codes" seen in the real delinquency columns
    special = rng.random(n) < 0.0018
    code = rng.choice([96, 98], special.sum())
    for col in (late30, late60, late90):
        col[special] = code
    risk = risk + 2.0 * special

    lo, hi = -15.0, 5.0  # bisection for the intercept that hits the target default rate
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(mid + risk)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(mid + risk)))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "SeriousDlqin2yrs": y,
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age.astype(int),
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": np.where(income_missing, np.nan, income),
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late90,
        "NumberRealEstateLoansOrLines": real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": dependents,
    })
    df.iloc[0, df.columns.get_loc("age")] = 0  # the real file has one borrower aged 0
    df.index = np.arange(1, n + 1)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_training.csv")
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    d = make(a.n, a.seed)
    d.to_csv(a.out, index_label="")
    print(f"wrote {a.out}  shape={d.shape}  default rate={d['SeriousDlqin2yrs'].mean():.4f}")
