"""End-to-end pipeline:  python -m src.train --data data/cs-training.csv --out reports

load -> clean -> split -> bin/WoE/IV (train only) -> select -> logistic -> evaluate ->
scale to points -> stability (PSI) -> benchmark -> save figures, tables, metrics.json, results.md
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from . import data as D
from . import metrics as M
from . import plots as P
from . import scorecard as S
from .binning import WoEBinner


def get_benchmark_model():
    try:
        from xgboost import XGBClassifier
        return "XGBoost", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
                                        random_state=42, n_jobs=-1)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "HistGradientBoosting (sklearn)", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)


def main(path, out, seed=42, label=None):
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    label = label or ("SYNTHETIC DATA" if "synthetic" in os.path.basename(path).lower()
                      else "Give Me Some Credit (Kaggle)")

    # 1. data ---------------------------------------------------------------
    raw = D.load_raw(path)
    df, clean_log = D.clean(raw)
    train, test = D.split(df, seed=seed)
    features_all = [c for c in df.columns if c != D.TARGET]
    ytr, yte = train[D.TARGET].to_numpy(), test[D.TARGET].to_numpy()
    print(f"[{label}] clean_log={clean_log}  train={len(train)} test={len(test)}")

    # 2. binning / WoE / IV (fit on TRAIN only) -------------------------------
    binner = WoEBinner(max_bins=6, min_bin_frac=0.02).fit(train[features_all], ytr)
    iv = binner.iv_table()
    woe_tr = binner.transform(train[features_all])
    woe_te = binner.transform(test[features_all])

    # 3. selection + model ------------------------------------------------------
    cand, sel_log = S.select_features(iv, woe_tr)
    model, feats, sign_log = S.fit_logit(woe_tr, ytr, cand, binner.iv_)
    sel_log += sign_log
    coefs, b0 = model.coef_[0], float(model.intercept_[0])
    p_tr, p_te = model.predict_proba(woe_tr[feats])[:, 1], model.predict_proba(woe_te[feats])[:, 1]

    # 4. scorecard ----------------------------------------------------------------
    sc = S.build_scorecard(binner, feats, coefs, b0)
    s_tr = S.apply_scorecard(sc, binner, train)
    s_te = S.apply_scorecard(sc, binner, test)
    pd_te_from_score = S.score_to_pd(s_te)

    # 5. metrics ---------------------------------------------------------------------
    lo, hi = M.bootstrap_auc_ci(yte, -s_te)
    res = {
        "data": label, "n_train": len(train), "n_test": len(test),
        "default_rate": clean_log["default_rate"], "cleaning": clean_log,
        "features_used": feats,
        "train": {"auc": M.auc(ytr, p_tr), "ks": M.ks_stat(ytr, p_tr), "gini": M.gini(ytr, p_tr)},
        "test": {"auc": M.auc(yte, p_te), "ks": M.ks_stat(yte, p_te), "gini": M.gini(yte, p_te),
                 "auc_ci95": [lo, hi], "brier": M.brier(yte, p_te)},
        "points_scorecard_test": {"auc": M.auc(yte, -s_te), "ks": M.ks_stat(yte, -s_te)},
        "score_range_test": [float(s_te.min()), float(s_te.max())],
        "calibration_mean_pd_vs_actual": [float(pd_te_from_score.mean()), float(yte.mean())],
        "scaling": {"base_score": S.BASE_SCORE, "base_odds": S.BASE_ODDS, "pdo": S.PDO},
    }

    # 6. stability ------------------------------------------------------------------------
    res["psi_score_train_vs_test"] = M.psi_continuous(s_tr, s_te)
    res["psi_features_train_vs_test"] = {
        f: M.psi_binned(binner.assign(f, train[f]), binner.assign(f, test[f])) for f in feats}

    # 7. benchmark (raw features, same split) -----------------------------------------------
    name, bm = get_benchmark_model()
    bm.fit(train[features_all], ytr)
    bm_auc = M.auc(yte, bm.predict_proba(test[features_all])[:, 1])
    res["benchmark"] = {"model": name, "test_auc": bm_auc}

    # 8. tables -------------------------------------------------------------------------------
    bands = M.score_band_table(yte, s_te)
    cal = M.calibration_table(yte, pd_te_from_score)
    cuts = np.quantile(s_te, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cut_tab = M.cutoff_table(yte, s_te, np.round(cuts).astype(int))
    iv.to_csv(f"{out}/iv_table.csv", index=False)
    sc.to_csv(f"{out}/scorecard_points.csv", index=False)
    bands.to_csv(f"{out}/score_bands_test.csv", index=False)
    cal.to_csv(f"{out}/calibration_test.csv", index=False)
    cut_tab.to_csv(f"{out}/cutoff_strategy_test.csv", index=False)
    pd.concat([t.assign(feature=f) for f, t in binner.tables_.items()]).to_csv(
        f"{out}/woe_bins_all.csv", index=False)
    with open(f"{out}/selection_log.txt", "w") as fh:
        fh.write("\n".join(sel_log) or "no features dropped")
    with open(f"{out}/metrics.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    joblib.dump({"binner": binner, "features": feats, "scorecard": sc, "model": model},
                f"{out}/scorecard_model.joblib")

    # 9. figures -----------------------------------------------------------------------------------
    P.iv_bar(iv, f"{fig_dir}/01_information_value.png")
    top = [f for f in iv["feature"] if f in feats][:3]
    for i, f in enumerate(top, 1):
        P.woe_bins(binner, f, f"{fig_dir}/02_bins_{i}_{f[:25]}.png")
    P.roc_ks(None, (ytr, p_tr), (yte, p_te), f"{fig_dir}/03_roc_ks.png")
    P.score_dist(s_te, yte, f"{fig_dir}/04_score_distribution.png")
    P.band_bar(bands, f"{fig_dir}/05_bad_rate_by_score_band.png")
    P.calibration(cal, f"{fig_dir}/06_calibration.png")
    P.benchmark(["Scorecard (logistic + WoE)", name], [res["test"]["auc"], bm_auc],
                f"{fig_dir}/07_benchmark.png")

    write_results_md(res, iv, sc, bands, cut_tab, sel_log, out)
    print(json.dumps({k: res[k] for k in ("train", "test", "benchmark", "psi_score_train_vs_test")},
                     indent=2, default=float))
    return res


def write_results_md(res, iv, sc, bands, cut_tab, sel_log, out):
    t, tr = res["test"], res["train"]
    L = [f"# Results  \n*Data source: **{res['data']}***\n"]
    if "SYNTHETIC" in res["data"]:
        L.append("> These numbers come from synthetic data and are for pipeline testing only. "
                 "Do not quote them as real results.\n")
    L += [
        f"- Borrowers: {res['n_train']:,} train / {res['n_test']:,} test; default rate {res['default_rate']:.2%}",
        f"- **Test AUC {t['auc']:.3f}** (95% CI {t['auc_ci95'][0]:.3f} to {t['auc_ci95'][1]:.3f}), "
        f"**KS {t['ks']:.3f}**, **Gini {t['gini']:.3f}**",
        f"- Train AUC {tr['auc']:.3f} vs test {t['auc']:.3f} (gap {tr['auc']-t['auc']:.3f}; small gap = no overfitting)",
        f"- Integer-points scorecard test AUC {res['points_scorecard_test']['auc']:.3f} "
        f"(rounding to whole points costs essentially nothing)",
        f"- {res['benchmark']['model']} benchmark test AUC {res['benchmark']['test_auc']:.3f}",
        f"- Score PSI train vs test {res['psi_score_train_vs_test']:.4f} (<0.10 = stable)",
        f"- Scale: {res['scaling']['base_score']} points at {res['scaling']['base_odds']}:1 odds, "
        f"+{res['scaling']['pdo']} points doubles the odds; test score range "
        f"{res['score_range_test'][0]:.0f} to {res['score_range_test'][1]:.0f}",
        f"- Variables in final model: {', '.join(res['features_used'])}",
        "\n## Information value\n", iv.to_markdown(index=False, floatfmt='.3f') if _has_tab() else iv.to_string(index=False),
        "\n## Score bands (test)\n", bands.round(4).to_markdown(index=False) if _has_tab() else bands.round(4).to_string(index=False),
        "\n## Cut-off strategy (test)\n", cut_tab.round(4).to_markdown(index=False) if _has_tab() else cut_tab.round(4).to_string(index=False),
        "\n## Selection log\n", "\n".join(f"- {s}" for s in sel_log) or "- nothing dropped",
    ]
    with open(f"{out}/results.md", "w") as fh:
        fh.write("\n".join(L))


def _has_tab():
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cs-training.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    main(a.data, a.out, a.seed, a.label)
