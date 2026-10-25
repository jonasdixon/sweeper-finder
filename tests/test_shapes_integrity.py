import numpy as np
import pandas as pd

from sweeper_finder import integrity, shapes


def _pitches(n=6):
    return pd.DataFrame({
        "pitch_type": ["SL"] * n, "p_throws": ["R", "L"] * (n // 2),
        "pfx_x": [0.5, -0.5] * (n // 2), "pfx_z": [0.1] * n, "release_speed": [85.0] * n,
        "game_pk": range(n), "at_bat_number": 1, "pitch_number": 1,
        "spin_axis": 90.0, "arm_angle": 40.0, "release_spin_rate": 2500.0,
    })


def test_glove_side_break_is_positive_for_both_hands_after_mirroring():
    hb = shapes.glove_side_break([0.5, -0.5], ["R", "L"])
    assert np.allclose(hb, [6.0, 6.0])


def test_slider_pool_drops_implausible_rows_and_counts_them():
    df = _pitches(8)
    df.loc[0, "release_speed"] = 125.0  # tracking error
    df.loc[1, "pfx_x"] = 4.0  # four feet of break
    df.loc[2, "pfx_z"] = np.nan
    df.loc[3, "pitch_type"] = "FF"  # not slider class
    pool, counts = shapes.slider_pool(df)
    assert counts["slider_class"] == 7 and counts["missing_shape_data"] == 1
    assert counts["implausible_speed"] == 1 and counts["implausible_movement"] == 1
    assert len(pool) == counts["modelled"] == 4


def test_mirror_check_flags_a_flipped_sign_for_one_hand():
    df = shapes.add_shape_features(_pitches(8))
    assert integrity.mirror_check(df).to_numpy().min() > 0
    flipped = df.assign(hb_in=np.where(df["p_throws"] == "L", -df["hb_in"], df["hb_in"]))
    assert integrity.mirror_check(flipped).to_numpy().min() < 0


def test_duplicate_keys_and_mixed_handedness_are_counted():
    df = _pitches(4)
    key = ["game_pk", "at_bat_number", "pitch_number"]
    df.loc[1, key] = df.loc[0, key].to_numpy()
    assert integrity.duplicate_keys(df) == 1
    df["pitcher"] = [1, 1, 2, 2]
    df["p_throws"] = ["R", "L", "R", "R"]
    assert integrity.mixed_handedness_pitchers(df) == 1


def test_label_switch_detection_finds_a_midseason_relabel():
    n = 300
    df = pd.DataFrame({
        "pitcher": 1, "player_name": "A", "game_date": pd.date_range("2024-04-01", periods=n),
        "game_pk": range(n), "at_bat_number": 1, "pitch_number": 1,
        "pitch_type": ["SL"] * 150 + ["ST"] * 150,
    })
    out = integrity.label_switches(df)
    assert len(out) == 1 and out.iloc[0]["st_share_first_third"] == 0 and out.iloc[0]["st_share_last_third"] == 1
