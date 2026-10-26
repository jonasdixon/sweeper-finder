"""Shape-feature checks: the mirroring test must notice a hand that was not flipped, and the dip
test must tell two populations from one."""

import numpy as np
import pandas as pd
import pytest

from sweeper_finder import eda
from sweeper_finder.shapes import add_shape_features


def pitches(pfx_x, p_throws, seed: int = 0) -> pd.DataFrame:
    df = pd.DataFrame({"pfx_x": np.asarray(pfx_x, float), "p_throws": np.asarray(p_throws),
                       "pfx_z": 0.0, "release_speed": 83.0})
    return add_shape_features(df)


def two_populations(gap_in: float, n: int = 60_000, sd_in: float = 2.0, seed: int = 1) -> pd.DataFrame:
    """Half the pitches near 3 in of break, half near `3 + gap_in`, rounded onto the export lattice."""
    rng = np.random.default_rng(seed)
    centres = np.where(rng.random(n) < 0.5, 3.0, 3.0 + gap_in)
    hb = rng.normal(centres, sd_in)
    pfx_x = np.round(hb / 12.0, 2)  # Savant rounds to 0.01 ft
    return pitches(pfx_x, np.full(n, "R"))


def one_population(n: int = 60_000, sd_in: float = 5.0, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pfx_x = np.round(rng.normal(8.0, sd_in, n) / 12.0, 2)
    return pitches(pfx_x, np.full(n, "R"))


# ---------- the dip ----------


def test_two_separated_populations_leave_a_deep_dip():
    v = eda.valley(two_populations(gap_in=16.0))
    assert v["dip_ratio"] < 0.25
    assert v["left_mode_in"] < v["valley_in"] < v["right_mode_in"]
    assert v["valley_in"] == pytest.approx(11.0, abs=2.5)


def test_one_population_leaves_no_usable_dip():
    """A single hump must not be reported as two groups with a boundary between them."""
    v = eda.valley(one_population())
    assert np.isnan(v["dip_ratio"]) or v["dip_ratio"] > 0.9


def test_dip_gets_shallower_as_the_populations_move_together():
    ratios = [eda.valley(two_populations(gap_in=g))["dip_ratio"] for g in (16.0, 11.0, 7.0)]
    assert ratios[0] < ratios[1] < ratios[2]


def test_valley_sensitivity_covers_every_combination_and_uses_lattice_bins():
    t = eda.valley_sensitivity(two_populations(gap_in=16.0), quanta=(2, 4), smooths=(1, 3))
    assert len(t) == 4
    assert np.allclose(t["step_in"], t["bin_quanta"] * eda.BREAK_QUANTUM_IN)
    assert (t["dip_ratio"] < 0.4).all()


def test_a_weak_second_hump_is_reported_and_not_invented():
    """The prominence rule has to keep a shallow but real hump while rejecting a noise bump."""
    weak = eda.valley(two_populations(gap_in=7.0))
    assert not np.isnan(weak["dip_ratio"])
    assert 0 < weak["prominence"] < 0.9
    assert np.isnan(eda.valley(one_population())["prominence"])


# ---------- quantisation ----------


def test_break_quantisation_finds_the_export_lattice():
    q = eda.break_quantisation(two_populations(gap_in=16.0))
    assert q["step_ft"] == pytest.approx(0.01, abs=1e-6)
    assert q["step_in"] == pytest.approx(0.12, abs=1e-4)
    assert q["distinct_values"] > 50


def test_default_bin_width_is_a_whole_number_of_lattice_steps():
    g = eda.break_density(two_populations(gap_in=16.0))
    width = float(np.diff(g["break_in"].to_numpy())[0])
    assert width / eda.BREAK_QUANTUM_IN == pytest.approx(round(width / eda.BREAK_QUANTUM_IN), abs=1e-6)


# ---------- mirroring ----------


def test_mirror_gap_is_large_before_the_flip_and_small_after():
    rng = np.random.default_rng(3)
    n = 40_000
    hand = np.where(rng.random(n) < 0.5, "R", "L")
    true_break = rng.normal(10.0, 3.0, n)  # glove side, same pitch for both hands
    pfx_x = np.round(np.where(hand == "R", true_break, -true_break) / 12.0, 2)
    gaps = eda.max_mirror_gap(pitches(pfx_x, hand))
    assert gaps["before_mirroring"] > 10.0
    assert gaps["after_mirroring"] < 0.5


def test_mirror_gap_stays_large_when_one_hand_is_not_flipped():
    """Planting the bug the mirroring is there to remove: left-handers left unflipped."""
    rng = np.random.default_rng(4)
    n = 40_000
    hand = np.where(rng.random(n) < 0.5, "R", "L")
    pfx_x = np.round(rng.normal(10.0, 3.0, n) / 12.0, 2)  # both hands stored with the same sign
    assert eda.max_mirror_gap(pitches(pfx_x, hand))["after_mirroring"] > 10.0


def test_mirror_gap_reports_every_quantile_for_both_stages():
    t = eda.mirror_gap(two_populations(gap_in=16.0).assign(p_throws="R"))
    assert set(t["stage"]) == {"before_mirroring", "after_mirroring"}
    assert len(t) == 18


# ---------- descriptive summaries ----------


def test_feature_table_reports_missingness():
    d = two_populations(gap_in=16.0, n=5_000)
    d.loc[d.index[:250], "speed_mph"] = np.nan
    t = eda.feature_table(d).set_index("feature")
    assert t.loc["speed_mph", "missing_pct"] == pytest.approx(5.0, abs=0.01)
    assert t.loc["hb_in", "missing_pct"] == 0.0


def test_outcome_curve_needs_outcome_columns():
    assert eda.outcome_curve(two_populations(gap_in=16.0, n=2_000)).empty


def test_outcome_curve_tracks_a_planted_relationship():
    rng = np.random.default_rng(5)
    d = two_populations(gap_in=16.0, n=80_000)
    p_whiff = 0.45 - 0.01 * d["hb_in"]  # more break, fewer whiffs per swing
    d = d.assign(swing=True, whiff=rng.random(len(d)) < p_whiff)
    curve = eda.outcome_curve(d, min_n=300)
    assert len(curve) > 5
    assert curve["whiff_rate"].iloc[0] > curve["whiff_rate"].iloc[-1]
    assert np.corrcoef(curve["hb_in"], curve["whiff_rate"])[0, 1] < -0.9


def test_report_bundles_the_three_checks():
    r = eda.report(two_populations(gap_in=16.0))
    assert set(r) == {"max_mirror_gap_in", "break_quantisation", "break_valley"}
