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


def feature_distributions(pool: pd.DataFrame, path: str, mixture_cut_in: float | None = None) -> None:
    """Pre-mixture look at the shape features: does the mirroring work, and is there a gap to split?

    The lower-left panel carries the argument. The sweeper category is only a boundary between two
    groups if the break distribution dips between two humps; otherwise it is a line drawn through
    one group.
    """
    from sweeper_finder import eda

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    hands = (("R", "#1d3f63", "right-handed"), ("L", "#c2410c", "left-handed"))
    # a whole number of lattice steps: pfx_x is rounded, so other widths add a comb of spikes
    bins = np.arange(-24.0, 26.0, 4 * eda.BREAK_QUANTUM_IN)

    ax = axes[0, 0]
    raw = 12.0 * pd.to_numeric(pool["pfx_x"], errors="coerce")
    for hand, color, label in hands:
        ax.hist(raw[pool["p_throws"] == hand], bins=bins, color=color, alpha=0.6, density=True, label=label)
    ax.set_xlabel("Horizontal break as exported, 12 * pfx_x (in)")
    ax.set_ylabel("Density")
    ax.set_title("Before mirroring: the hands are opposites", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for hand, color, label in hands:
        ax.hist(pool.loc[pool["p_throws"] == hand, "hb_in"], bins=bins, color=color, alpha=0.6,
                density=True, label=label)
    gap = eda.max_mirror_gap(pool)
    ax.set_xlabel("Glove-side break after mirroring (in)")
    ax.set_ylabel("Density")
    ax.set_title(f"After mirroring: largest decile gap {gap['after_mirroring']:.1f} in\n"
                 f"(was {gap['before_mirroring']:.1f} in)", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    g = eda.break_density(pool)
    v = eda.valley(pool)
    ax.plot(g["break_in"], g["density"], color="#1d3f63", lw=1.4)
    if not np.isnan(v["valley_in"]):
        ax.axvline(v["valley_in"], color="#c2410c", lw=1.4, label=f"density dip ({v['valley_in']:.1f} in)")
        for m in (v["left_mode_in"], v["right_mode_in"]):
            ax.axvline(m, color="grey", lw=0.8, ls=":")
    if mixture_cut_in is not None:
        ax.axvline(mixture_cut_in, color="#2f7d32", lw=1.4, ls="--",
                   label=f"mixture boundary ({mixture_cut_in:.1f} in)")
    ax.set_xlabel("Glove-side break (in)")
    ax.set_ylabel("Density")
    ax.set_title(f"Two humps, but a shallow dip: ratio {v['dip_ratio']:.2f}\n"
                 "(0 = separate groups, 1 = one group)", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    curve = eda.outcome_curve(pool)
    if len(curve):
        ax.plot(curve["hb_in"], curve["whiff_rate"], "o-", ms=3, color="#1d3f63")
        ax.set_ylabel("Whiff rate per swing (raw)")
    ax.set_xlabel("Glove-side break (in)")
    ax.set_title("Does the feature track the outcome?", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
