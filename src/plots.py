"""Static figures for the README. Two-colour story: blue = good (repaid), orange = bad (default)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

GOOD, BAD = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "legend.frameon": False, "figure.dpi": 100,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def iv_bar(iv_table, path, threshold=0.02):
    t = iv_table.sort_values("iv")
    short = [s if len(s) < 34 else s[:31] + "..." for s in t["feature"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(short, t["iv"], color=GOOD, height=0.6)
    ax.axvline(threshold, color=MUTED, lw=1, ls="--")
    ax.text(threshold, len(t) - 0.45, "  min IV 0.02", color=MUTED, fontsize=8, va="bottom")
    ax.set_xlabel("Information Value")
    ax.set_title("Which variables separate good from bad borrowers?")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def woe_bins(binner, feature, path, title=None):
    t = binner.tables_[feature]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(t)), t["bad_rate"] * 100, color=BAD, width=0.6)
    for i, (r, c) in enumerate(zip(t["bad_rate"], t["pct"])):
        ax.text(i, r * 100, f"{r*100:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels([f"{b}\n({p*100:.0f}% of loans)" for b, p in zip(t["bin"], t["pct"])],
                       fontsize=7)
    ax.set_ylabel("Default rate (%)")
    ax.set_title(title or f"Default rate by bin: {feature}")
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def roc_ks(y, p_train_pair, p_test_pair, path):
    """p_*_pair = (y, p). Left: ROC train vs test. Right: KS on test."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (yy, pp), lab, col in ((p_train_pair, "Train", MUTED), (p_test_pair, "Test", GOOD)):
        f, t, _ = roc_curve(yy, pp)
        a.plot(f, t, color=col, lw=2, label=lab)
    a.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_title("ROC curve: train vs test"); a.legend(loc="lower right")
    yt, pt = p_test_pair
    f, t, _ = roc_curve(yt, pt)
    k = np.argmax(t - f)
    b.plot(f, t, color=BAD, lw=2, label="Cumulative bads")
    b.plot(f, f, color=GOOD, lw=2, label="Cumulative goods")
    b.vlines(f[k], f[k], t[k], color=INK, lw=1.5)
    b.text(f[k] - 0.02, (f[k] + t[k]) / 2, f"KS = {t[k]-f[k]:.2f}", fontsize=10, ha="right", fontweight="bold")
    b.set_xlabel("Share of borrowers ranked riskiest-first"); b.set_ylabel("Cumulative share")
    b.set_title("KS: how far apart bads and goods separate"); b.legend(loc="lower right")
    _save(fig, path)


def score_dist(score, y, path):
    score, y = np.asarray(score), np.asarray(y)
    bins = np.linspace(np.quantile(score, 0.002), np.quantile(score, 0.998), 45)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(score[y == 0], bins=bins, color=GOOD, alpha=0.75, density=True, label="Repaid (good)")
    ax.hist(score[y == 1], bins=bins, color=BAD, alpha=0.75, density=True, label="Defaulted (bad)")
    ax.set_xlabel("Credit score (higher = safer)"); ax.set_ylabel("Density")
    ax.set_title("Score distribution on the test set"); ax.legend()
    _save(fig, path)


def calibration(cal, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    m = max(cal["mean_predicted_pd"].max(), cal["observed_bad_rate"].max()) * 100 * 1.05
    ax.plot([0, m], [0, m], color=GRID, lw=1.5, ls="--", label="Perfect")
    ax.plot(cal["mean_predicted_pd"] * 100, cal["observed_bad_rate"] * 100, "o-",
            color=GOOD, lw=2, ms=6, label="Model (deciles)")
    ax.set_xlabel("Predicted PD (%)"); ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Calibration on the test set"); ax.legend(loc="upper left")
    _save(fig, path)


def band_bar(bands, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bands["band"], bands["bad_rate"] * 100, color=BAD, width=0.65)
    for x, r in zip(bands["band"], bands["bad_rate"]):
        ax.text(x, r * 100, f"{r*100:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(bands["band"])
    ax.set_xlabel("Score decile (1 = lowest scores / riskiest)"); ax.set_ylabel("Default rate (%)")
    ax.set_title("Default rate falls as the score rises"); ax.grid(axis="x", visible=False)
    _save(fig, path)


def benchmark(names, aucs, path):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    cols = [GOOD] + [MUTED] * (len(names) - 1)
    ax.barh(names[::-1], aucs[::-1], color=cols[::-1], height=0.5)
    lo = min(aucs) - 0.03
    ax.set_xlim(lo, max(aucs) + 0.015)
    for i, v in enumerate(aucs[::-1]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Test AUC (axis starts at %.2f)" % lo)
    ax.set_title("Interpretable scorecard vs black-box benchmark")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
