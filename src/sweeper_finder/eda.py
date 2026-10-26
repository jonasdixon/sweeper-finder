"""Checks on the shape features, run before the mixture is fitted.

Two things have to be true before any sweeper number means anything.

First, the mirroring has to work. `shapes.glove_side_break` flips left-handers so both hands can be
pooled. `integrity.mirror_check` compares means, which would pass even if the two distributions had
different shapes. `mirror_gap` compares them decile by decile instead, before and after the flip.

Second, there has to be something to separate. The mixture splits its components into sweeper-like
and ordinary by a 2-means cut on glove-side break, and that is only meaningful if the break
distribution really has two humps with a dip between them. `valley` finds the dip in a smoothed
histogram, with no mixture involved, and reports how deep it is. A flat or unimodal distribution
gives a dip ratio near 1, which would mean the sweeper category is a line drawn through a single
population rather than a boundary between two.

`outcome_curve` is the third question, the one any new feature has to answer: does it track the
outcome? It gives whiff rate against break in fine steps. The steps are for plotting and nothing is
compared across buckets, so no continuous variable is dichotomised.

One thing this found that was not expected: `pfx_x` is exported rounded to 0.01 ft, so break is not
continuous but sits on a lattice 0.12 in apart. Bin widths that are not a whole multiple of that
step produce a comb of spikes in the density and made the dip look about 0.25 deeper than it is.
`break_quantisation` reports the lattice and every bin width here is a multiple of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sweeper_finder.shapes import FEATURES, glove_side_break

# Savant exports `pfx_x` rounded to 0.01 ft, so break lands on a lattice 0.12 in apart. A histogram
# bin that is not a whole number of lattice steps holds two steps in some bins and three in others,
# which puts a regular comb of spikes in the density and makes the dip between the humps look deeper
# than it is. Every bin width here is a multiple of this.
BREAK_QUANTUM_IN = 0.12


def feature_table(pool: pd.DataFrame) -> pd.DataFrame:
    """Distribution of each shape feature, with the share of rows missing it."""
    rows = []
    for col in FEATURES:
        s = pd.to_numeric(pool[col], errors="coerce")
        rows.append({
            "feature": col,
            "n": int(s.notna().sum()),
            "missing_pct": round(100.0 * s.isna().mean(), 4),
            "mean": s.mean(), "sd": s.std(),
            "p01": s.quantile(0.01), "p25": s.quantile(0.25), "median": s.median(),
            "p75": s.quantile(0.75), "p99": s.quantile(0.99),
        })
    return pd.DataFrame(rows).round(4)


def mirror_gap(pool: pd.DataFrame, n_quantiles: int = 9) -> pd.DataFrame:
    """Left- minus right-handed glove-side break at each decile, before and after mirroring.

    Before the flip the two hands are near mirror images, so the gap is large and negative. After it
    they should describe the same pitch, so every decile gap should be small.
    """
    q = np.linspace(0.1, 0.9, n_quantiles)
    raw = 12.0 * pd.to_numeric(pool["pfx_x"], errors="coerce")
    mirrored = pd.Series(glove_side_break(pool["pfx_x"], pool["p_throws"]), index=pool.index)
    left = pool["p_throws"].eq("L").to_numpy()
    rows = []
    for name, values in (("before_mirroring", raw), ("after_mirroring", mirrored)):
        lq = values[left].quantile(q).to_numpy()
        rq = values[~left].quantile(q).to_numpy()
        for p, lv, rv in zip(q, lq, rq, strict=True):
            rows.append({"stage": name, "quantile": round(float(p), 2),
                         "left_in": round(float(lv), 3), "right_in": round(float(rv), 3),
                         "gap_in": round(float(lv - rv), 3)})
    return pd.DataFrame(rows)


def max_mirror_gap(pool: pd.DataFrame, n_quantiles: int = 9) -> dict:
    """The largest decile gap between the hands, before and after mirroring, in inches."""
    t = mirror_gap(pool, n_quantiles)
    return {
        stage: round(float(part["gap_in"].abs().max()), 3)
        for stage, part in t.groupby("stage", observed=True)
    }


def break_quantisation(pool: pd.DataFrame, source: str = "pfx_x") -> dict:
    """How finely the export records horizontal break.

    `pfx_x` arrives rounded, so break is not continuous: it sits on a lattice. The bin widths used
    below have to be whole multiples of that lattice step or the histogram grows a comb of spikes
    that is an artefact of binning, not a property of sliders.
    """
    x = pd.to_numeric(pool[source], errors="coerce").dropna()
    if x.empty:
        return {"distinct_values": 0, "step_ft": float("nan"), "step_in": float("nan")}
    gaps = np.diff(np.unique(x.to_numpy()))
    gaps = gaps[gaps > 1e-9]
    step_ft = float(np.min(gaps)) if len(gaps) else float("nan")
    return {
        "distinct_values": int(x.nunique()),
        "step_ft": round(step_ft, 5),
        "step_in": round(12.0 * step_ft, 4),
        "distinct_break_values": int(pd.to_numeric(pool["hb_in"], errors="coerce").nunique()),
    }


def break_density(pool: pd.DataFrame, lo: float = -8.04, hi: float = 26.04,
                  step: float = 4 * BREAK_QUANTUM_IN, smooth: int = 1) -> pd.DataFrame:
    """Histogram of mirrored glove-side break.

    The default is the raw histogram at four lattice steps per bin. With 300,000 pitches that is
    already well populated, and smoothing is what turns a shallow dip into no dip at all, or invents
    a hump that is not there; `valley_sensitivity` shows both. `smooth` applies a moving average of
    that many bins when set above 1.
    """
    hb = pd.to_numeric(pool["hb_in"], errors="coerce").dropna().to_numpy()
    edges = np.arange(lo, hi + step, step)
    counts, _ = np.histogram(hb, bins=edges)
    density = counts / max(counts.sum(), 1) / step
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        density = np.convolve(density, kernel, mode="same")
    return pd.DataFrame({"break_in": edges[:-1] + step / 2, "density": density})


def valley(pool: pd.DataFrame, min_separation_in: float = 4.0, min_prominence: float = 0.05,
           **kwargs) -> dict:
    """The dip between the two largest humps in the break distribution.

    `dip_ratio` is the density at the dip divided by the density at the lower of the two humps.
    Near 0 means two clearly separate populations; near 1 means one population with no boundary in
    it, and the sweeper category would then be a cut through the middle of a single group.

    A second hump only counts if its prominence -- its height above the dip leading back to the main
    peak -- is at least `min_prominence` of the main peak. Without that, any bump in the tail of a
    single smooth distribution is picked up as a second mode and a real-looking dip is reported for
    data that has none; the tests check exactly that case.

    Returns nan values when no second hump qualifies.
    """
    g = break_density(pool, **kwargs)
    x, y = g["break_in"].to_numpy(), g["density"].to_numpy()
    nan = {"left_mode_in": float("nan"), "right_mode_in": float("nan"), "valley_in": float("nan"),
           "dip_ratio": float("nan"), "prominence": float("nan")}
    if len(x) < 3 or y.max() <= 0:
        return nan
    first = int(np.argmax(y))
    interior = np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:])) + 1
    best = None
    for i in interior:
        if abs(x[i] - x[first]) < min_separation_in:
            continue
        lo_i, hi_i = sorted((first, int(i)))
        trough = float(np.min(y[lo_i:hi_i + 1]))
        prominence = (min(y[first], y[i]) - trough) / y[first]
        if prominence >= min_prominence and (best is None or prominence > best[1]):
            best = (int(i), prominence)
    if best is None:
        return nan
    second, prominence = best
    lo_i, hi_i = sorted((first, second))
    dip = int(np.argmin(y[lo_i:hi_i + 1])) + lo_i
    lower_mode = min(y[first], y[second])
    return {
        "left_mode_in": round(float(x[lo_i]), 2),
        "right_mode_in": round(float(x[hi_i]), 2),
        "valley_in": round(float(x[dip]), 2),
        "dip_ratio": round(float(y[dip] / lower_mode), 4) if lower_mode > 0 else float("nan"),
        "prominence": round(float(prominence), 4),
    }


def outcome_curve(pool: pd.DataFrame, feature: str = "hb_in", step: float = 8 * BREAK_QUANTUM_IN,
                  min_n: int = 200) -> pd.DataFrame:
    """Whiff rate per swing against a shape feature, raw. No model, no smoothing."""
    d = pool.loc[pool["swing"].astype(bool)] if "swing" in pool.columns else pool
    if "whiff" not in d.columns or not len(d):
        return pd.DataFrame(columns=[feature, "whiff_rate", "n"])
    bin_ = (np.round(pd.to_numeric(d[feature], errors="coerce") / step) * step).rename(feature)
    g = d.groupby(bin_, observed=True)["whiff"].agg(["mean", "size"]).reset_index()
    g = g.loc[g["size"] >= min_n].rename(columns={"mean": "whiff_rate", "size": "n"})
    return g.sort_values(feature).reset_index(drop=True)


def valley_sensitivity(pool: pd.DataFrame, quanta=(2, 3, 4, 5, 8), smooths=(1, 3, 5)) -> pd.DataFrame:
    """The dip under every bin width and smoothing window, because the answer depends on them.

    Bin widths are given in lattice steps (`BREAK_QUANTUM_IN`). Smoothing a shallow dip removes it,
    and a wide enough window invents modes that are not there. Reporting one dip ratio without this
    table would make the number look firmer than it is.
    """
    rows = []
    for q in quanta:
        for smooth in smooths:
            step = q * BREAK_QUANTUM_IN
            v = valley(pool, step=step, smooth=smooth)
            rows.append({"bin_quanta": q, "step_in": round(step, 4), "smooth_bins": smooth, **v})
    return pd.DataFrame(rows)


def probability_crossing(pool: pd.DataFrame, prob_col: str = "sweeper_prob", feature: str = "hb_in",
                         step: float = 4 * BREAK_QUANTUM_IN, min_n: int = 200) -> float:
    """Where a fitted sweeper probability crosses 0.5, read in units of the feature.

    This is how the mixture's boundary, which lives in three dimensions, can be compared with the
    dip in the one-dimensional break distribution. The two are found by unrelated means, so landing
    in the same place is evidence that the boundary is a property of the data.
    """
    if prob_col not in pool.columns:
        return float("nan")
    d = pool.dropna(subset=[prob_col, feature])
    bin_ = (np.round(pd.to_numeric(d[feature], errors="coerce") / step) * step).rename(feature)
    g = d.groupby(bin_, observed=True)[prob_col].agg(["mean", "size"]).reset_index()
    g = g.loc[g["size"] >= min_n].sort_values(feature)
    if len(g) < 2:
        return float("nan")
    x, y = g[feature].to_numpy(), g["mean"].to_numpy()
    below = y < 0.5
    cross = np.flatnonzero(below[:-1] & ~below[1:])
    if len(cross) == 0:
        return float("nan")
    i = int(cross[0])
    if y[i + 1] == y[i]:
        return float(x[i])
    return float(x[i] + (0.5 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def report(pool: pd.DataFrame) -> dict:
    """Everything above as one dictionary, for `summary.json`."""
    return {
        "max_mirror_gap_in": max_mirror_gap(pool),
        "break_quantisation": break_quantisation(pool),
        "break_valley": valley(pool),
    }
