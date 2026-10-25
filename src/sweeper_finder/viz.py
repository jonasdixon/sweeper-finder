"""Figures. No radar charts; every plot shows the data or an interval, not a composite."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402


def movement_map(pool: pd.DataFrame, mixture, path: str, n: int = 40_000, seed: int = 0) -> None:
    """Break against induced vertical break, coloured by sweeper probability, with 2-sigma component ellipses."""
    s = pool.sample(min(n, len(pool)), random_state=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sc = axes[0].scatter(s["hb_in"], s["ivb_in"], c=s["sweeper_prob"], s=2, alpha=0.35, cmap="coolwarm", vmin=0, vmax=1,
                         rasterized=True)
    for mean, cov, sw in zip(mixture.gmm.means_, mixture.gmm.covariances_, mixture.sweeper_like, strict=True):
        vals, vecs = np.linalg.eigh(cov[:2, :2])
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        axes[0].add_patch(Ellipse(mean[:2], 4 * np.sqrt(vals[1]), 4 * np.sqrt(vals[0]), angle=angle, fill=False,
                                  ec="black" if sw else "grey", lw=1.6 if sw else 1, ls="-" if sw else "--"))
    axes[0].set_xlabel("Glove-side break (in)")
    axes[0].set_ylabel("Induced vertical break (in)")
    fig.colorbar(sc, ax=axes[0], label="P(sweeper-like)")
    for flag, color, label in ((False, "#4c78a8", "other sliders"), (True, "#c2410c", "sweeper-like")):
        d = s.loc[s["sweeper_like"] == flag]
        axes[1].scatter(d["hb_in"], d["speed_mph"], s=2, alpha=0.3, c=color, label=label, rasterized=True)
    axes[1].set_xlabel("Glove-side break (in)")
    axes[1].set_ylabel("Velocity (mph)")
    axes[1].legend(markerscale=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contrast_forest(table: pd.DataFrame, path: str) -> None:
    """Within-pitcher sweeper-like minus other-pitch differences with 95% intervals, one panel per metric."""
    metrics = list(dict.fromkeys(table["metric"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 2.8 + 0.3 * table["split"].nunique()),
                             squeeze=False)
    for ax, m in zip(axes[0], metrics, strict=True):
        t = table.loc[table["metric"] == m].reset_index(drop=True)
        y = np.arange(len(t))[::-1]
        ax.errorbar(t["difference"], y, xerr=[t["difference"] - t["ci_lo"], t["ci_hi"] - t["difference"]], fmt="o",
                    color="#1d3f63", capsize=2)
        ax.set_yticks(y, t["split"] + " / " + t["versus"] if m == metrics[0] else [""] * len(t), fontsize=7)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(m, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def quality_vs_usage(table: pd.DataFrame, path: str) -> None:
    t = table.copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    other = t.loc[~t["flag_underused"]]
    ax.errorbar(100 * other["usage"], other["rv100_post"], yerr=1.96 * other["rv100_post_sd"], fmt="o", ms=3,
                color="#9aa5b1", alpha=0.6, lw=0.6)
    flagged = t.loc[t["flag_underused"]]
    ax.errorbar(100 * flagged["usage"], flagged["rv100_post"], yerr=1.96 * flagged["rv100_post_sd"], fmt="o", ms=5,
                color="#c2410c", lw=1)
    ax.axhline(t["league_rv100"].iloc[0], color="black", lw=0.8, ls="--")
    ax.set_xlabel("Sweeper-like usage (% of all pitches)")
    ax.set_ylabel("Runs saved per 100 sweepers (shrunk, 95% interval)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def usage_change(table: pd.DataFrame, slope: dict, path: str, min_n: int = 75) -> None:
    a = table.loc[(table["game_year"] == table["game_year"].min()) & (table["n_sweeper"] >= min_n)]
    b = table.loc[(table["game_year"] == table["game_year"].max()) & (table["n_sweeper"] >= min_n)]
    m = a.merge(b, on="pitcher", suffixes=("_a", "_b"))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(100 * (m["usage_b"] - m["usage_a"]), m["rv100_b"] - m["rv100_a"], s=10, alpha=0.5, color="#4c78a8")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Change in sweeper-like usage, 2023 to 2024 (points)")
    ax.set_ylabel("Change in runs saved per 100 sweepers")
    ax.set_title(f"slope {slope['slope_per_10pp']:.2f} per +10 pts "
                 f"({slope['ci_lo_per_10pp']:.2f}, {slope['ci_hi_per_10pp']:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
