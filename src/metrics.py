"""Model-quality and stability metrics used in credit risk."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks_stat(y, p) -> float:
    """Max gap between cumulative bad and cumulative good distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def bootstrap_auc_ci(y, p, n_boot=200, seed=0, alpha=0.05):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[i], p[i]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def psi_from_dist(expected, actual, eps=1e-6) -> float:
    """PSI = sum (actual% - expected%) * ln(actual% / expected%).  <0.1 stable, 0.1-0.25 watch, >0.25 shift."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_continuous(expected, actual, bins=10) -> float:
    expected, actual = np.asarray(expected), np.asarray(actual)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    return psi_from_dist(e, a)


def psi_binned(ids_expected, ids_actual) -> float:
    levels = np.union1d(np.unique(ids_expected), np.unique(ids_actual))
    e = np.array([(ids_expected == l).mean() for l in levels])
    a = np.array([(ids_actual == l).mean() for l in levels])
    return psi_from_dist(e, a)


def score_band_table(y, score, n=10) -> pd.DataFrame:
    """Deciles by score, riskiest (lowest score) first. Bad rate must fall as score rises."""
    d = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(score)})
    d["band"] = pd.qcut(d["score"].rank(method="first"), n, labels=False) + 1
    g = d.groupby("band").agg(n=("y", "size"), min_score=("score", "min"),
                              max_score=("score", "max"), bads=("y", "sum"),
                              bad_rate=("y", "mean")).reset_index()
    g["share_of_all_bads"] = g["bads"] / g["bads"].sum()
    g["cum_share_of_bads"] = g["share_of_all_bads"].cumsum()
    return g


def calibration_table(y, pd_hat, n=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    d["bucket"] = pd.qcut(d["p"].rank(method="first"), n, labels=False) + 1
    return d.groupby("bucket").agg(n=("y", "size"), mean_predicted_pd=("p", "mean"),
                                   observed_bad_rate=("y", "mean")).reset_index()


def cutoff_table(y, score, cutoffs) -> pd.DataFrame:
    """Business view: approve everyone with score >= cutoff."""
    y, score = np.asarray(y), np.asarray(score)
    overall = y.mean()
    rows = []
    for c in cutoffs:
        m = score >= c
        rows.append({"cutoff": c, "approval_rate": m.mean(),
                     "bad_rate_among_approved": y[m].mean() if m.any() else np.nan,
                     "bads_caught_by_rejecting": 1 - (y[m].sum() / y.sum()),
                     "overall_bad_rate": overall})
    return pd.DataFrame(rows)
