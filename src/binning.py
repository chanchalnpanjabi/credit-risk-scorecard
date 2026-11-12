"""Supervised binning + Weight of Evidence (WoE) + Information Value (IV).

WoE of a bin  = ln( %goods in bin / %bads in bin )     (higher = safer)
IV of a var   = sum over bins of (%goods - %bads) * WoE (predictive power)

Binning method: a shallow decision tree finds the cut-points that best separate
goods from bads; we then require the bad rate to move in ONE direction across bins
(monotonic), because a credit officer must be able to explain "more utilisation =
more risk". If a split is not monotonic we try fewer bins. Missing values get their
own bin so "missing" is treated as information, not silently imputed.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

MISSING_ID = -1


def _monotonic(rates) -> bool:
    d = np.diff(rates)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def _labels(edges):
    if not edges:
        return ["all values"]
    f = lambda v: f"{v:.4g}"
    out = [f"(-inf, {f(edges[0])}]"]
    out += [f"({f(a)}, {f(b)}]" for a, b in zip(edges[:-1], edges[1:])]
    out.append(f"({f(edges[-1])}, inf)")
    return out


class WoEBinner:
    def __init__(self, max_bins=6, min_bin_frac=0.02, monotonic=True, smoothing=0.5):
        self.max_bins = max_bins
        self.min_bin_frac = min_bin_frac
        self.monotonic = monotonic
        self.smoothing = smoothing

    # ---------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        self.edges_, self.tables_, self.woe_maps_, self.iv_ = {}, {}, {}, {}
        for col in X.columns:
            self._fit_col(col, X[col].to_numpy(dtype=float), y)
        return self

    def _fit_col(self, col, x, y):
        ok = ~np.isnan(x)
        xv, yv = x[ok], y[ok]
        min_leaf = max(int(self.min_bin_frac * len(x)), 1)
        edges = []
        for k in range(self.max_bins, 1, -1):
            tree = DecisionTreeClassifier(max_leaf_nodes=k, min_samples_leaf=min_leaf,
                                          random_state=0).fit(xv.reshape(-1, 1), yv)
            thr = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
            if len(thr) == 0:
                edges = []
                break
            idx = np.searchsorted(thr, xv, side="left")
            rates = [yv[idx == i].mean() for i in range(len(thr) + 1)]
            edges = thr.tolist()
            if not self.monotonic or _monotonic(rates):
                break
        self.edges_[col] = edges

        ids = self._assign(col, x)
        labels = _labels(edges)
        goods_total, bads_total = (y == 0).sum(), (y == 1).sum()
        uniq = sorted(np.unique(ids))
        s, nb = self.smoothing, len(uniq)
        rows = []
        for b in uniq:
            m = ids == b
            cnt, bad = int(m.sum()), int(y[m].sum())
            good = cnt - bad
            dg = (good + s) / (goods_total + s * nb)
            db = (bad + s) / (bads_total + s * nb)
            woe = float(np.log(dg / db))
            rows.append({"bin_id": int(b), "bin": "Missing" if b == MISSING_ID else labels[b],
                         "count": cnt, "pct": cnt / len(x), "goods": good, "bads": bad,
                         "bad_rate": bad / cnt, "woe": woe, "iv": float((dg - db) * woe)})
        tab = pd.DataFrame(rows)
        self.tables_[col] = tab
        self.woe_maps_[col] = dict(zip(tab["bin_id"], tab["woe"]))
        self.iv_[col] = float(tab["iv"].sum())

    # ---------------------------------------------------------- transform
    def _assign(self, col, x):
        ids = np.full(len(x), MISSING_ID, dtype=int)
        ok = ~np.isnan(x)
        ids[ok] = np.searchsorted(self.edges_[col], x[ok], side="left")
        return ids

    def assign(self, col, s: pd.Series):
        return self._assign(col, s.to_numpy(dtype=float))

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.edges_)
        out = {}
        for c in cols:
            ids = self.assign(c, X[c])
            m = self.woe_maps_[c]
            out[c] = np.array([m.get(i, 0.0) for i in ids])  # unseen bin -> neutral WoE
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        t = pd.DataFrame({"feature": list(self.iv_), "iv": list(self.iv_.values())})
        t["strength"] = pd.cut(t["iv"], [-1, 0.02, 0.1, 0.3, 0.5, 99],
                               labels=["useless", "weak", "medium", "strong", "suspiciously strong"])
        return t.sort_values("iv", ascending=False).reset_index(drop=True)
