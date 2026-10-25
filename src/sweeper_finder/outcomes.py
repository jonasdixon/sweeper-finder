"""Outcome flags, rates with explicit denominators, and shrinkage for small samples.

  csw      (called strikes + whiffs) / pitches
  whiff    whiffs / swings
  chase    swings at out-of-zone pitches / out-of-zone pitches
  xwOBAcon mean expected wOBA on balls in play (Statcast's estimate from exit velocity and angle;
           the actual wOBA value is used where no estimate exists, e.g. bunts)
  rv100    runs saved per 100 pitches, from the change in run expectancy on each pitch
           (`delta_run_exp` is from the batting side's view, so the sign is flipped)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SWING = frozenset({
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "foul_bunt",
    "missed_bunt", "bunt_foul_tip", "hit_into_play",
})
WHIFF = frozenset({"swinging_strike", "swinging_strike_blocked", "missed_bunt"})
NON_COMPETITIVE = frozenset({"automatic_ball", "automatic_strike", "pitchout", "foul_pitchout", "swinging_pitchout"})
HALF_WIDTH_FT = 0.83

# numerator column, denominator column for each reported rate
METRICS = {
    "csw": ("csw_n", "n"),
    "whiff": ("whiffs", "swings"),
    "chase": ("chases", "out_of_zone"),
    "xwobacon": ("xwobacon_sum", "contacts"),
    "rv100": ("rv_sum", "n"),
}


def in_zone(df: pd.DataFrame) -> pd.Series:
    geometric = (
        (df["plate_x"].abs() <= HALF_WIDTH_FT) & (df["plate_z"] >= df["sz_bot"]) & (df["plate_z"] <= df["sz_top"])
    )
    zone = df["zone"]
    return pd.Series(np.where(zone.notna(), zone.between(1, 9), geometric), index=df.index)


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pitch flags. Pitchouts and automatic balls/strikes are dropped first."""
    out = df.loc[~df["description"].isin(NON_COMPETITIVE)].copy()
    d = out["description"]
    out["swing"] = d.isin(SWING)
    out["whiff"] = d.isin(WHIFF)
    out["csw"] = d.eq("called_strike") | out["whiff"]
    out["in_zone"] = in_zone(out).astype(bool)
    out["chase"] = out["swing"] & ~out["in_zone"]
    out["contact"] = d.eq("hit_into_play")
    xw = out["estimated_woba_using_speedangle"].fillna(out["woba_value"])
    out["xwobacon_val"] = xw.where(out["contact"])
    out["rv"] = -out["delta_run_exp"]
    return out


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sums per group, from which every metric in METRICS is a ratio of two columns."""
    g = df.groupby(by, observed=True)
    return g.agg(
        n=("swing", "size"), swings=("swing", "sum"), whiffs=("whiff", "sum"), csw_n=("csw", "sum"),
        out_of_zone=("in_zone", lambda s: int((~s).sum())), chases=("chase", "sum"),
        contacts=("contact", "sum"), xwobacon_sum=("xwobacon_val", "sum"), rv_sum=("rv", "sum"),
    ).reset_index()


def rates(agg: pd.DataFrame) -> pd.DataFrame:
    out = agg.copy()
    for name, (num, den) in METRICS.items():
        out[name] = out[num] / out[den].where(out[den] > 0)
    out["rv100"] = out["rv100"] * 100
    return out


def beta_binomial_prior(successes, trials) -> tuple[float, float]:
    """Method-of-moments Beta(alpha, beta) prior from per-pitcher counts."""
    x, n = np.asarray(successes, float), np.asarray(trials, float)
    keep = n > 0
    x, n = x[keep], n[keep]
    p = x.sum() / n.sum()
    obs_var = np.average((x / n - p) ** 2, weights=n)
    noise = p * (1 - p) * np.average(1.0 / n, weights=n)
    tau2 = max(obs_var - noise, 1e-6)
    strength = max(p * (1 - p) / tau2 - 1.0, 1.0)
    return p * strength, (1 - p) * strength


def shrink_rate(successes, trials, alpha: float, beta: float) -> np.ndarray:
    """Posterior mean of a rate under Beta(alpha, beta)."""
    return (np.asarray(successes, float) + alpha) / (np.asarray(trials, float) + alpha + beta)


def normal_prior(means, ses) -> tuple[float, float]:
    """Precision-weighted league mean and between-pitcher variance for continuous metrics."""
    m, s2 = np.asarray(means, float), np.asarray(ses, float) ** 2
    w = 1.0 / (s2 + 1e-12)
    mu = float(np.sum(w * m) / np.sum(w))
    tau2 = max(float(np.average((m - mu) ** 2) - np.mean(s2)), 1e-9)
    return mu, tau2


def shrink_normal(means, ses, mu: float, tau2: float) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation under a Normal(mu, tau2) prior."""
    m, s2 = np.asarray(means, float), np.asarray(ses, float) ** 2
    k = tau2 / (tau2 + s2)
    return mu + k * (m - mu), np.sqrt(k * s2)


def pitcher_flag_table(pitches: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Per-pitcher sums for pitches with `flag` True and False, side by side (pitchers with both)."""
    a = aggregate(pitches.assign(_f=pitches[flag]), ["pitcher", "_f"])
    cols = sorted({c for pair in METRICS.values() for c in pair} | {"n"})
    return a.pivot(index="pitcher", columns="_f", values=cols).dropna()


def within_pitcher_contrast(
    wide: pd.DataFrame, metric: str, n_boot: int = 300, min_n: int = 30, seed: int = 0
) -> dict:
    """Mean within-pitcher difference in `metric` between flagged and unflagged pitches.

    `wide` comes from `pitcher_flag_table`. Each pitcher contributes flagged-minus-other for
    himself, so differences in pitcher quality cancel. Pitchers are weighted by
    n_f*n_o/(n_f+n_o); the interval resamples pitchers.
    """
    num, den = METRICS[metric]
    wide = wide[(wide[("n", True)] >= min_n) & (wide[("n", False)] >= min_n)]
    scale = 100.0 if metric == "rv100" else 1.0
    ns, no = wide[("n", True)].to_numpy(float), wide[("n", False)].to_numpy(float)
    cols = {k: wide[k].to_numpy(float) for k in [(num, True), (den, True), (num, False), (den, False)]}

    def estimate(idx):
        with np.errstate(invalid="ignore", divide="ignore"):
            d = scale * (cols[(num, True)][idx] / cols[(den, True)][idx]
                         - cols[(num, False)][idx] / cols[(den, False)][idx])
        w = (ns[idx] * no[idx]) / (ns[idx] + no[idx])
        ok = np.isfinite(d)
        return float(np.sum(w[ok] * d[ok]) / np.sum(w[ok]))

    n = len(wide)
    point = estimate(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [estimate(rng.integers(0, n, n)) for _ in range(n_boot)]
    return {"metric": metric, "difference": point, "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "pitchers": n}
