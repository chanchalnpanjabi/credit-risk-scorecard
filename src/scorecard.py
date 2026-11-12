"""Feature selection, logistic regression on WoE, and scaling probabilities to points.

Sign convention: we model P(bad). WoE is ln(good/bad), so a safer bin has HIGHER WoE
and the fitted coefficient should be NEGATIVE. A positive coefficient means
multicollinearity is distorting the model, so we drop that feature and refit.

Scaling (industry standard):
    Score = Offset + Factor * ln(odds_good)
    Factor = PDO / ln 2            Offset = BaseScore - Factor * ln(BaseOdds)
"Base score 600 at 50:1 good:bad odds, every +20 points doubles the odds."
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20


def scaling(base_score=BASE_SCORE, base_odds=BASE_ODDS, pdo=PDO):
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def select_features(iv: pd.DataFrame, woe: pd.DataFrame, iv_min=0.02, iv_max=0.5, corr_max=0.75):
    """Keep features with useful IV, then drop the weaker one of any highly correlated pair."""
    log, kept = [], []
    cand = iv[(iv["iv"] >= iv_min)]
    for f in iv.loc[iv["iv"] < iv_min, "feature"]:
        log.append(f"drop {f}: IV < {iv_min}")
    for _, r in cand.sort_values("iv", ascending=False).iterrows():
        f = r["feature"]
        if r["iv"] > iv_max:
            log.append(f"WARNING {f}: IV {r['iv']:.2f} > {iv_max} (check for leakage)")
        clash = [k for k in kept if abs(woe[f].corr(woe[k])) > corr_max]
        if clash:
            log.append(f"drop {f}: |corr| > {corr_max} with {clash[0]} (higher IV kept)")
        else:
            kept.append(f)
    return kept, log


def fit_logit(woe: pd.DataFrame, y, features, iv: dict, C=1e6):
    """Fit; while any coefficient is positive, drop the lowest-IV offender and refit."""
    log, feats = [], list(features)
    while True:
        m = LogisticRegression(C=C, max_iter=2000).fit(woe[feats], y)
        pos = [f for f, c in zip(feats, m.coef_[0]) if c > 0]
        if not pos:
            return m, feats, log
        worst = min(pos, key=lambda f: iv[f])
        log.append(f"drop {worst}: positive coefficient (sign flip from collinearity)")
        feats.remove(worst)


def build_scorecard(binner, features, coefs, intercept, **kw) -> pd.DataFrame:
    """One row per (feature, bin) with integer points. Higher score = safer."""
    factor, offset = scaling(**kw)
    n = len(features)
    base_pts = (offset - factor * intercept) / n
    rows = []
    for f, b in zip(features, coefs):
        for r in binner.tables_[f].itertuples():
            raw = -b * r.woe * factor + base_pts
            rows.append({"feature": f, "bin_id": r.bin_id, "bin": r.bin, "count": r.count,
                         "bad_rate": r.bad_rate, "woe": r.woe, "coef": b,
                         "points_raw": raw, "points": int(round(raw))})
    return pd.DataFrame(rows)


def apply_scorecard(sc: pd.DataFrame, binner, X: pd.DataFrame, factor_kw=None) -> np.ndarray:
    """Score borrowers purely from the points table (what a loan officer would do by hand)."""
    factor_kw = factor_kw or {}
    factor, offset = scaling(**factor_kw)
    total = np.zeros(len(X))
    feats = sc["feature"].unique()
    for f in feats:
        sub = sc[sc["feature"] == f]
        lookup = dict(zip(sub["bin_id"], sub["points"]))
        # unseen bin -> neutral (WoE = 0) points = this feature's share of the intercept
        neutral = int(round((sub["points_raw"] + sub["coef"] * sub["woe"] * factor).mean()))
        ids = binner.assign(f, X[f])
        total += np.array([lookup.get(i, neutral) for i in ids])
    return total


def score_to_pd(score, **kw):
    factor, offset = scaling(**kw)
    return 1.0 / (1.0 + np.exp((np.asarray(score, float) - offset) / factor))
