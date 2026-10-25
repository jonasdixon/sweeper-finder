import numpy as np
import pandas as pd

from sweeper_finder import outcomes, usage


def _frame(descs, in_zone):
    n = len(descs)
    return pd.DataFrame({
        "description": descs, "plate_x": 0.0, "plate_z": 2.5, "sz_top": 3.4, "sz_bot": 1.6,
        "zone": [5 if z else 12 for z in in_zone], "delta_run_exp": 0.0,
        "estimated_woba_using_speedangle": np.nan, "woba_value": np.nan, "pitcher": 1,
    }).iloc[:n]


def test_rates_use_the_right_denominators():
    df = _frame(["swinging_strike", "foul", "ball", "ball", "called_strike", "hit_into_play"],
                [True, False, False, False, True, False])
    flagged = outcomes.add_outcomes(df)
    r = outcomes.rates(outcomes.aggregate(flagged, ["pitcher"])).iloc[0]
    assert r["n"] == 6 and r["swings"] == 3 and r["whiffs"] == 1
    assert np.isclose(r["whiff"], 1 / 3)
    assert r["out_of_zone"] == 4 and r["chases"] == 2 and np.isclose(r["chase"], 2 / 4)  # not 2 / swings
    assert r["csw_n"] == 2 and np.isclose(r["csw"], 2 / 6)


def test_pitchouts_and_automatic_calls_are_dropped():
    df = _frame(["ball", "pitchout", "automatic_ball"], [False] * 3)
    assert len(outcomes.add_outcomes(df)) == 1


def test_xwobacon_falls_back_to_actual_woba_when_no_estimate():
    df = _frame(["hit_into_play", "hit_into_play"], [True, True])
    df["estimated_woba_using_speedangle"] = [0.5, np.nan]
    df["woba_value"] = [0.9, 0.7]
    r = outcomes.rates(outcomes.aggregate(outcomes.add_outcomes(df), ["pitcher"])).iloc[0]
    assert np.isclose(r["xwobacon"], 0.6)


def test_run_value_sign_is_the_pitchers_view():
    df = _frame(["called_strike", "ball"], [True, False])
    df["delta_run_exp"] = [-0.05, 0.03]
    f = outcomes.add_outcomes(df)
    assert list(np.round(f["rv"], 2)) == [0.05, -0.03]


def test_beta_binomial_shrinkage_pulls_small_samples_toward_the_mean():
    rng = np.random.default_rng(0)
    n = rng.integers(20, 400, 500)
    true = rng.beta(30, 70, 500)
    x = rng.binomial(n, true)
    a, b = outcomes.beta_binomial_prior(x, n)
    assert abs(a / (a + b) - 0.3) < 0.02
    shrunk = outcomes.shrink_rate(x, n, a, b)
    small = n < 40
    raw_err = np.abs(x / n - true)[small].mean()
    assert np.abs(shrunk - true)[small].mean() < raw_err
    assert np.isclose(outcomes.shrink_rate(3, 10, 2, 8), 5 / 20)


def test_normal_shrinkage_limits():
    mu, tau2 = 1.0, 0.25
    post, sd = outcomes.shrink_normal([3.0, 3.0], [100.0, 1e-4], mu, tau2)
    assert np.isclose(post[0], mu, atol=1e-3) and np.isclose(post[1], 3.0, atol=1e-3)
    assert sd[0] < 0.5 + 1e-6  # never wider than the prior


def test_within_pitcher_contrast_removes_pitcher_quality_confounding():
    rng = np.random.default_rng(3)
    rows = []
    for p in range(150):
        quality = rng.normal(0, 0.05)  # good pitchers get more csw on everything
        has_sweeper = p % 2 == 0
        for flag in (True, False):
            n = 300 if (flag and has_sweeper) or not flag else 0
            if n == 0:
                continue
            base = 0.28 + quality + (0.04 if flag else 0.0)  # planted sweeper effect +4 points
            csw = rng.random(n) < base
            rows.append(pd.DataFrame({"pitcher": p, "sweeper_like": flag, "csw": csw, "swing": csw, "whiff": csw,
                                      "in_zone": True, "chase": False, "contact": False, "xwobacon_val": np.nan,
                                      "rv": 0.0}))
    d = pd.concat(rows, ignore_index=True)
    wide = outcomes.pitcher_flag_table(d, "sweeper_like")
    res = outcomes.within_pitcher_contrast(wide, "csw", n_boot=100)
    assert abs(res["difference"] - 0.04) < 0.015 and res["ci_lo"] < 0.04 < res["ci_hi"]


def _season_table(**over):
    base = {
        "pitcher": [1, 2, 3, 4], "player_name": list("ABCD"), "game_year": 2024,
        "n_total": 1500, "n_sweeper": 300, "arm_angle": 40.0, "p_throws": "R", "same_hand_share": 0.4,
        "n_pitch_types": 3, "usage": [0.10, 0.10, 0.25, 0.25], "rv100": [3.0, 0.2, 0.2, 3.0],
        "rv100_se": 0.4,
    }
    base.update(over)
    return pd.DataFrame(base)


def test_flag_requires_high_quality_low_usage_and_no_steep_diminishing_returns():
    t = _season_table()
    expected = lambda x: np.full(len(x), 0.20)  # noqa: E731
    mild = {"slope_per_10pp": -0.1, "ci_lo_per_10pp": -0.3, "ci_hi_per_10pp": 0.1}
    out = usage.flag_underused(t, expected, mild).set_index("pitcher")
    assert out.loc[1, "flag_underused"]  # good pitch, thrown less than expected
    assert not out.loc[2, "flag_underused"]  # low usage but not good
    assert not out.loc[3, "flag_underused"] and not out.loc[4, "flag_underused"]  # at or above expected usage
    steep = {"slope_per_10pp": -3.0, "ci_lo_per_10pp": -6.0, "ci_hi_per_10pp": -1.0}
    assert not usage.flag_underused(t, expected, steep).set_index("pitcher").loc[1, "flag_underused"]


def test_usage_quality_slope_recovers_a_planted_relationship():
    rng = np.random.default_rng(4)
    n = 300
    u23 = rng.uniform(0.05, 0.3, n)
    du = rng.normal(0, 0.06, n)
    q23 = rng.normal(1.0, 0.5, n)
    q24 = q23 - 2.0 * du + rng.normal(0, 0.15, n)  # -0.2 runs/100 per +10 points
    rows = []
    for yr, u, q in ((2023, u23, q23), (2024, u23 + du, q24)):
        rows.append(pd.DataFrame({"pitcher": range(n), "game_year": yr, "usage": u, "rv100": q, "n_sweeper": 400}))
    res = usage.usage_quality_slope(pd.concat(rows), n_boot=200)
    assert abs(res["slope_per_10pp"] - (-0.2)) < 0.05
    assert res["ci_lo_per_10pp"] < -0.2 < res["ci_hi_per_10pp"]


def test_expected_usage_model_learns_that_lefties_throw_more():
    rng = np.random.default_rng(5)
    n = 400
    t = pd.DataFrame({
        "p_throws": rng.choice(["L", "R"], n), "arm_angle": rng.normal(40, 8, n),
        "same_hand_share": rng.uniform(0.2, 0.6, n), "n_pitch_types": 3, "n_total": 1500, "n_sweeper": 300,
    })
    t["usage"] = np.clip(0.15 + 0.08 * (t["p_throws"] == "L") + rng.normal(0, 0.02, n), 0.02, 0.6)
    pred = usage.fit_expected_usage(t)
    p = pred(t)
    assert p[t["p_throws"] == "L"].mean() - p[t["p_throws"] == "R"].mean() > 0.05
