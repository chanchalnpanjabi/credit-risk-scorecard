"""Run with:  python -m pytest -q     (or: python tests/test_core.py)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D
from src import metrics as M
from src import scorecard as S
from src.binning import WoEBinner
from src.make_synthetic import make


def _toy(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(-2.5 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    x_missing = x.copy()
    x_missing[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({"x": x_missing, "noise": rng.normal(0, 1, n)}), y


def test_woe_sign_and_iv_formula():
    X, y = _toy()
    b = WoEBinner().fit(X, y)
    t = b.tables_["x"]
    # higher x -> more risk -> lower WoE in higher bins
    nonmiss = t[t["bin"] != "Missing"]
    assert nonmiss["woe"].is_monotonic_decreasing
    # IV recomputed by hand matches
    assert abs(t["iv"].sum() - b.iv_["x"]) < 1e-12
    assert b.iv_["x"] > 0.3 and b.iv_["noise"] < 0.02  # signal vs noise


def test_bins_are_monotonic_and_missing_has_own_bin():
    X, y = _toy()
    t = WoEBinner().fit(X, y).tables_["x"]
    assert "Missing" in set(t["bin"])
    rates = t[t["bin"] != "Missing"]["bad_rate"].to_numpy()
    d = np.diff(rates)
    assert np.all(d >= 0) or np.all(d <= 0)


def test_transform_unseen_missing_is_neutral():
    X, y = _toy()
    X.loc[:, "x"] = X["x"].fillna(0.0)  # no missing in train
    b = WoEBinner().fit(X, y)
    X2 = X.copy()
    X2.loc[0, "x"] = np.nan
    assert b.transform(X2)["x"].iloc[0] == 0.0


def test_scaling_doubles_odds_per_pdo():
    f, o = S.scaling(600, 50, 20)
    score = lambda odds: o + f * np.log(odds)
    assert abs(score(50) - 600) < 1e-9
    assert abs(score(100) - 620) < 1e-9
    assert abs(score(25) - 580) < 1e-9
    assert abs(S.score_to_pd(600) - 1 / 51) < 1e-9  # 50:1 odds -> PD 1/51


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50000)
    assert M.psi_continuous(a, a) < 1e-9
    assert M.psi_continuous(a, rng.normal(0.6, 1, 50000)) > 0.25


def test_ks_auc_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([.1, .2, .3, .4, .6, .7, .8, .9])
    assert M.auc(y, perfect) == 1.0 and M.ks_stat(y, perfect) == 1.0 and M.gini(y, perfect) == 1.0
    assert abs(M.auc(y, np.zeros(8)) - 0.5) < 1e-12


def test_cleaning_rules():
    df = make(5000, seed=3)
    clean, log = D.clean(df)
    assert (clean["age"] >= 18).all()
    for c in D.DELINQ_COLS:
        assert not clean[c].isin(D.SPECIAL_CODES).any()
    assert log["delinquency_special_codes_set_to_nan"] > 0


def test_end_to_end_points_scorecard_matches_model():
    df, _ = D.clean(make(40000, seed=5))
    tr, te = D.split(df)
    feats_all = [c for c in df.columns if c != D.TARGET]
    b = WoEBinner().fit(tr[feats_all], tr[D.TARGET])
    w = b.transform(tr[feats_all])
    cand, _ = S.select_features(b.iv_table(), w)
    m, feats, _ = S.fit_logit(w, tr[D.TARGET], cand, b.iv_)
    assert (m.coef_[0] < 0).all()  # sign convention holds
    sc = S.build_scorecard(b, feats, m.coef_[0], float(m.intercept_[0]))
    s = S.apply_scorecard(sc, b, te)
    p = m.predict_proba(b.transform(te[feats_all], feats))[:, 1]
    assert abs(M.auc(te[D.TARGET], -s) - M.auc(te[D.TARGET], p)) < 0.005  # rounding is harmless
    assert np.corrcoef(S.score_to_pd(s), p)[0, 1] > 0.99


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("PASS", f.__name__)
