"""Fill README results, LinkedIn post and CV bullets from reports/metrics.json.

    python -m src.render_docs

Every number you publish therefore comes straight from your own run, never from memory.
"""
import argparse
import glob
import json
import os
import re

import pandas as pd


def tokens(rep):
    m = json.load(open(f"{rep}/metrics.json"))
    iv = pd.read_csv(f"{rep}/iv_table.csv")
    t, tr, b = m["test"], m["train"], m["benchmark"]
    gap = b["test_auc"] - t["auc"]
    if gap > 0.0005:
        lead = "gives up only" if gap <= 0.01 else "gives up"
        phrase = f"{lead} {gap:.3f} AUC while staying fully explainable"
    elif gap < -0.0005:
        phrase = f"edges it out by {-gap:.3f} AUC while staying fully explainable"
    else:
        phrase = "matches it on AUC while staying fully explainable"
    return {
        "DATASET": m["data"], "N_TRAIN": f"{m['n_train']:,}", "N_TEST": f"{m['n_test']:,}",
        "N_TOTAL": f"{m['n_train'] + m['n_test']:,}", "DEFAULT_RATE": f"{m['default_rate']:.1%}",
        "AUC": f"{t['auc']:.3f}", "KS": f"{t['ks']:.2f}", "GINI": f"{t['gini']:.2f}",
        "AUC_LO": f"{t['auc_ci95'][0]:.3f}", "AUC_HI": f"{t['auc_ci95'][1]:.3f}",
        "TRAIN_AUC": f"{tr['auc']:.3f}", "OVERFIT_GAP": f"{abs(tr['auc'] - t['auc']):.3f}",
        "BENCH_NAME": b["model"], "BENCH_AUC": f"{b['test_auc']:.3f}", "TRADEOFF": phrase,
        "PSI": f"{m['psi_score_train_vs_test']:.4f}", "N_FEATURES": str(len(m["features_used"])),
        "TOP_FEATURE": iv.loc[0, "feature"], "TOP_IV": f"{iv.loc[0, 'iv']:.2f}",
        "SCORE_MIN": f"{m['score_range_test'][0]:.0f}", "SCORE_MAX": f"{m['score_range_test'][1]:.0f}",
        "FEATURE_LIST": ", ".join(m["features_used"]),
    }


def fill(text, tok):
    return re.sub(r"\{\{(\w+)\}\}", lambda g: tok.get(g.group(1), g.group(0)), text)


def results_block(rep, tok):
    figs = sorted(glob.glob(f"{rep}/figures/*.png"))
    rel = lambda p: p.replace("\\", "/")
    lines = [
        f"**Data:** {tok['DATASET']} | **{tok['N_TOTAL']}** borrowers | default rate **{tok['DEFAULT_RATE']}** "
        f"| 70/30 stratified split",
        "",
        "| Metric (test set) | Value |", "|---|---|",
        f"| AUC | **{tok['AUC']}** (95% CI {tok['AUC_LO']} to {tok['AUC_HI']}) |",
        f"| KS | **{tok['KS']}** |", f"| Gini | **{tok['GINI']}** |",
        f"| Train AUC (overfitting gap) | {tok['TRAIN_AUC']} ({tok['OVERFIT_GAP']}) |",
        f"| {tok['BENCH_NAME']} benchmark AUC | {tok['BENCH_AUC']} |",
        f"| Score PSI, train vs test | {tok['PSI']} (< 0.10 = stable) |",
        f"| Score range | {tok['SCORE_MIN']} to {tok['SCORE_MAX']} |",
        "",
        f"The interpretable scorecard {tok['TRADEOFF']}.",
        f"Variables in the final model ({tok['N_FEATURES']}): {tok['FEATURE_LIST']}.",
        "",
    ]
    for f in figs:
        lines.append(f"![{os.path.basename(f)}]({rel(os.path.relpath(f, '.'))})")
    lines += ["", "Full tables: [`reports/results.md`](reports/results.md), "
              "[`reports/scorecard_points.csv`](reports/scorecard_points.csv)."]
    return "\n".join(lines)


def main(rep="reports", docs="docs", readme="README.md"):
    tok = tokens(rep)
    if "SYNTHETIC" in tok["DATASET"]:
        print("\n*** WARNING: these reports come from SYNTHETIC data. Do not publish. "
              "Re-run src.train on data/cs-training.csv first. ***\n")
    for tpl in glob.glob(f"{docs}/*.template.md"):
        out = tpl.replace(".template.md", ".md")
        open(out, "w").write(fill(open(tpl).read(), tok))
        print("wrote", out)
    txt = open(readme).read()
    block = "<!-- RESULTS:START -->\n" + results_block(rep, tok) + "\n<!-- RESULTS:END -->"
    new = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block, txt, flags=re.S)
    open(readme, "w").write(new)
    print("updated", readme)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    main(a.reports, a.docs, a.readme)
