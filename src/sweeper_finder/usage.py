"""Sweeper usage, an expected-usage model, a usage-versus-quality curve, and under-use flags.

Under-use is a hypothesis for a human to check, not a recommendation. A flag requires all of:
  1. quality: the pitcher's sweeper-like pitch is better than the league's sweeper-like pitch
     (shrunk run value per 100, probability above league at least `min_prob_better`);
  2. usage: observed usage is below what pitchers with the same hand, arm angle, platoon exposure
     and repertoire breadth use, by more than binomial noise and at least `min_gap` in absolute
     terms;
  3. no steep diminishing returns: even if raising usage to the expected level costs run value at
     the pessimistic end of the estimated within-pitcher usage-quality slope, quality stays above
     league.
Condition 3 exists because pitchers who use a pitch less often usually use it in its best spots;
extra usage is not free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from sweeper_finder import outcomes

FASTBALLS = {"FF", "SI", "FC"}


def pitcher_season_table(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher-season: usage of the sweeper-like pitch and context for expectations.

    `pitches` needs: pitcher, player_name, game_year, p_throws, stand, pitch_type, arm_angle,
    sweeper_like (bool, False outside the slider pool), rv (runs saved, pitcher's view).
    """
    key = ["pitcher", "player_name", "game_year"]
    d = pitches.copy()
    base = d.groupby(key).agg(
        n_total=("rv", "size"), n_sweeper=("sweeper_like", "sum"), arm_angle=("arm_angle", "mean"),
        p_throws=("p_throws", "first"),
    ).reset_index()
    same = d.assign(_same=(d["stand"] == d["p_throws"])).groupby(key)["_same"].mean().rename("same_hand_share")
    base = base.merge(same.reset_index(), on=key)
    type_share = d.groupby(key + ["pitch_type"]).size().rename("c").reset_index()
    type_share = type_share.merge(base[key + ["n_total"]], on=key)
    n_types = (type_share.assign(s=type_share["c"] / type_share["n_total"]).query("s >= 0.05")
               .groupby(key).size().rename("n_pitch_types").reset_index())
    base = base.merge(n_types, on=key, how="left")
    sw = d.loc[d["sweeper_like"]].groupby(key)["rv"].agg(rv_mean="mean", rv_sd="std").reset_index()
    base = base.merge(sw, on=key, how="left")
    base["usage"] = base["n_sweeper"] / base["n_total"]
    base["rv100"] = 100 * base["rv_mean"]
    base["rv100_se"] = 100 * base["rv_sd"] / np.sqrt(base["n_sweeper"].clip(lower=1))
    return base.drop(columns=["rv_mean", "rv_sd"])


def _design(t: pd.DataFrame, arm_fill: float) -> np.ndarray:
    return np.column_stack([
        (t["p_throws"] == "L").astype(float), t["arm_angle"].fillna(arm_fill),
        t["same_hand_share"], t["n_pitch_types"].fillna(2),
    ])


def fit_expected_usage(table: pd.DataFrame, min_sweepers: int = 50):
    """Fractional logistic regression of usage on hand, arm angle, platoon exposure and repertoire
    breadth, over pitcher-seasons that throw a sweeper-like pitch at all. Returns a predictor."""
    fit = table.loc[table["n_sweeper"] >= min_sweepers]
    arm_fill = float(fit["arm_angle"].mean())
    X = _design(fit, arm_fill)
    mean, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mean) / sd
    n = fit["n_total"].to_numpy(float)
    u = fit["usage"].to_numpy(float)
    clf = LogisticRegression(C=1e3, max_iter=2000).fit(
        np.vstack([Xs, Xs]), np.r_[np.ones(len(fit)), np.zeros(len(fit))],
        sample_weight=np.r_[u * n, (1 - u) * n])

    def predict(t: pd.DataFrame) -> np.ndarray:
        return clf.predict_proba((_design(t, arm_fill) - mean) / sd)[:, 1]

    return predict


def usage_quality_slope(table: pd.DataFrame, min_n: int = 75, n_boot: int = 500, seed: int = 0) -> dict:
    """Within-pitcher change in sweeper run value per +10 points of usage, between the two seasons.

    Differencing a pitcher against himself removes stable pitcher quality. Weighted least squares
    with an intercept (the intercept absorbs a league-wide season shift); pitchers are weighted by
    the harmonic mean of their sweeper counts and resampled for the interval.
    """
    a = table.loc[(table["game_year"] == table["game_year"].min()) & (table["n_sweeper"] >= min_n)]
    b = table.loc[(table["game_year"] == table["game_year"].max()) & (table["n_sweeper"] >= min_n)]
    m = a.merge(b, on="pitcher", suffixes=("_a", "_b"))
    du = (m["usage_b"] - m["usage_a"]).to_numpy()
    dq = (m["rv100_b"] - m["rv100_a"]).to_numpy()
    w = 1.0 / (1.0 / m["n_sweeper_a"].to_numpy() + 1.0 / m["n_sweeper_b"].to_numpy())

    def slope(idx):
        x, y, ww = du[idx], dq[idx], w[idx]
        xm, ym = np.average(x, weights=ww), np.average(y, weights=ww)
        return np.sum(ww * (x - xm) * (y - ym)) / np.sum(ww * (x - xm) ** 2)

    n = len(m)
    rng = np.random.default_rng(seed)
    point = slope(np.arange(n))
    boots = np.array([slope(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"pitchers": n, "slope_per_10pp": float(0.10 * point),
            "ci_lo_per_10pp": float(0.10 * np.percentile(boots, 2.5)),
            "ci_hi_per_10pp": float(0.10 * np.percentile(boots, 97.5))}


def flag_underused(
    table: pd.DataFrame, expected, slope: dict, min_sweepers: int = 100, min_prob_better: float = 0.8,
    min_gap: float = 0.03,
) -> pd.DataFrame:
    """Add quality, expected usage and the flag to pitcher-seasons with enough sweeper-like pitches."""
    t = table.loc[table["n_sweeper"] >= min_sweepers].copy()
    mu, tau2 = outcomes.normal_prior(t["rv100"], t["rv100_se"])
    t["rv100_post"], t["rv100_post_sd"] = outcomes.shrink_normal(t["rv100"], t["rv100_se"], mu, tau2)
    t["league_rv100"] = mu
    t["p_better_than_league"] = 1 - norm.cdf((mu - t["rv100_post"]) / t["rv100_post_sd"])
    t["expected_usage"] = expected(t)
    t["usage_gap"] = t["usage"] - t["expected_usage"]
    t["usage_se"] = np.sqrt(t["usage"] * (1 - t["usage"]) / t["n_total"])
    raise_by = (-t["usage_gap"]).clip(lower=0)
    pessimistic = t["rv100_post"] + slope["ci_lo_per_10pp"] * (raise_by / 0.10)
    t["rv100_at_expected_usage_pessimistic"] = pessimistic
    t["flag_underused"] = (
        (t["p_better_than_league"] >= min_prob_better)
        & (t["usage_gap"] <= -np.maximum(min_gap, 1.5 * t["usage_se"]))
        & (pessimistic > mu)
    )
    return t.sort_values(["flag_underused", "p_better_than_league"], ascending=False).reset_index(drop=True)
