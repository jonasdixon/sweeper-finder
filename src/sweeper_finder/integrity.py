"""Integrity checks for the slider pool, run before any model is fit."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["game_pk", "at_bat_number", "pitch_number"]


def duplicate_keys(df: pd.DataFrame) -> int:
    return int(df.duplicated(KEY).sum())


def mirror_check(pool: pd.DataFrame) -> pd.DataFrame:
    """Mean mirrored glove-side break by pitch type and hand. After mirroring, every slider-class
    pitch type should be clearly positive for both hands; a negative cell means a flipped sign."""
    return pool.groupby(["pitch_type", "p_throws"])["hb_in"].mean().unstack().round(2)


def mixed_handedness_pitchers(df: pd.DataFrame) -> int:
    """Pitchers recorded with both L and R throwing hands (data errors; switch-pitchers are
    essentially nonexistent in this window)."""
    return int((df.groupby("pitcher")["p_throws"].nunique() > 1).sum())


def coverage(pool: pd.DataFrame) -> dict:
    return {
        "spin_axis_present": float(pool["spin_axis"].notna().mean()),
        "arm_angle_present": float(pool["arm_angle"].notna().mean()),
        "spin_rate_present": float(pool["release_spin_rate"].notna().mean()),
    }


def label_switches(pool: pd.DataFrame, min_pitches: int = 150, min_shift: float = 0.5) -> pd.DataFrame:
    """Pitchers whose share of ST-labelled pitches (among slider-class pitches) jumps between the
    first and last third of their pitches. That is either a new pitch or a relabel, and a person
    should look at the video before it is treated as either."""
    rows = []
    ordered = pool.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    for pitcher, g in ordered.groupby("pitcher"):
        if len(g) < min_pitches:
            continue
        st = (g["pitch_type"] == "ST").to_numpy(float)
        third = len(st) // 3
        first, last = st[:third].mean(), st[-third:].mean()
        if abs(last - first) >= min_shift:
            rows.append({
                "pitcher": pitcher, "name": g["player_name"].iloc[0], "pitches": len(g),
                "st_share_first_third": round(first, 2), "st_share_last_third": round(last, 2),
            })
    return pd.DataFrame(rows)


def report(raw: pd.DataFrame, pool: pd.DataFrame, filter_counts: dict) -> dict:
    return {
        "filter_counts": filter_counts,
        "duplicate_keys": duplicate_keys(raw),
        "mixed_handedness_pitchers": mixed_handedness_pitchers(raw),
        "coverage": coverage(pool),
        "mirror_check_hb_in": mirror_check(pool).to_dict(),
        "mirror_ok": bool(np.all(mirror_check(pool).to_numpy() > 0)),
    }
