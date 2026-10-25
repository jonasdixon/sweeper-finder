"""Pitch shape features and the slider candidate pool."""

from __future__ import annotations

import numpy as np
import pandas as pd

POOL_TYPES = ("SL", "ST", "SV")  # Statcast slider, sweeper, slurve
FEATURES = ["hb_in", "ivb_in", "speed_mph"]
SPEED_RANGE_MPH = (65.0, 100.0)  # for a slider; slower is an eephus or a position player, faster a mislabel
MAX_BREAK_IN = 36.0  # three feet of movement is a tracking error


def glove_side_break(pfx_x_ft, p_throws) -> np.ndarray:
    """Horizontal movement in inches, positive toward the pitcher's glove side for both hands.

    In this Statcast export `pfx_x` is positive for a right-hander's glove-side break and negative
    for a left-hander's, so right-handers keep the sign and left-handers are flipped. This is
    checked against the data in `integrity.mirror_check`, not assumed.
    """
    sign = np.where(np.asarray(p_throws) == "R", 1.0, -1.0)
    return 12.0 * np.asarray(pfx_x_ft, float) * sign


def add_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hb_in"] = glove_side_break(out["pfx_x"], out["p_throws"])
    out["ivb_in"] = 12.0 * out["pfx_z"].astype(float)  # induced: gravity already removed by Statcast
    out["speed_mph"] = out["release_speed"].astype(float)
    return out


def slider_pool(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Slider-class pitches with usable shape data, and how many rows each filter removed."""
    counts = {"all_pitches": len(df)}
    pool = df.loc[df["pitch_type"].isin(POOL_TYPES)]
    counts["slider_class"] = len(pool)
    pool = add_shape_features(pool)
    missing = pool[FEATURES].isna().any(axis=1)
    counts["missing_shape_data"] = int(missing.sum())
    pool = pool.loc[~missing]
    lo, hi = SPEED_RANGE_MPH
    bad_speed = ~pool["speed_mph"].between(lo, hi)
    counts["implausible_speed"] = int(bad_speed.sum())
    pool = pool.loc[~bad_speed]
    bad_move = (pool["hb_in"].abs() > MAX_BREAK_IN) | (pool["ivb_in"].abs() > MAX_BREAK_IN)
    counts["implausible_movement"] = int(bad_move.sum())
    pool = pool.loc[~bad_move]
    counts["modelled"] = len(pool)
    return pool, counts


def shape_summary(pool: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return pool.groupby(by, observed=True)[FEATURES].agg(["mean", "std", "count"]).round(2)
