import numpy as np
import pandas as pd

from sweeper_finder import mixture, shapes
from sweeper_finder.mixture import SweeperMixture


def simulate_shapes(n_per=4000, seed=0):
    """Gyro sliders, standard sliders and sweepers, thrown by both hands, in raw pfx units."""
    rng = np.random.default_rng(seed)
    classes = {  # glove break in, induced vertical in, mph
        "gyro": ((3.0, -1.0, 88.0), (1.2, 1.5, 1.5)),
        "slider": ((6.0, 1.0, 85.0), (1.5, 1.5, 1.5)),
        "sweeper": ((15.0, 0.0, 81.0), (1.8, 1.8, 1.5)),
    }
    rows = []
    for name, (mu, sd) in classes.items():
        for hand in ("R", "L"):
            x = rng.normal(mu, sd, (n_per, 3))
            sign = 1 if hand == "R" else -1
            rows.append(pd.DataFrame({
                "pfx_x": sign * x[:, 0] / 12, "pfx_z": x[:, 1] / 12, "release_speed": x[:, 2],
                "p_throws": hand, "truth": name,
            }))
    df = pd.concat(rows, ignore_index=True)
    return shapes.add_shape_features(df)


def test_mixture_recovers_planted_sweepers_for_both_hands():
    df = simulate_shapes()
    X = df[shapes.FEATURES].to_numpy()
    m = SweeperMixture.fit(X, 3)
    like = m.sweeper_prob(X) >= 0.5
    truth = (df["truth"] == "sweeper").to_numpy()
    assert (like == truth).mean() > 0.98
    for hand in ("R", "L"):
        sel = (df["p_throws"] == hand).to_numpy()
        assert (like[sel] == truth[sel]).mean() > 0.97


def test_high_break_split_is_a_property_of_the_data_not_a_typed_threshold():
    means = np.array([3.0, 6.0, 15.0])
    w = np.array([0.4, 0.4, 0.2])
    assert list(mixture.high_break_components(means, w)) == [False, False, True]
    # shift everything up: the same structure is found, no absolute cut-off is involved
    assert list(mixture.high_break_components(means + 10, w)) == [False, False, True]


def test_choose_k_stops_when_bic_stops_improving():
    df = simulate_shapes(n_per=3000)
    X = df[shapes.FEATURES].to_numpy()
    k, table = mixture.choose_k(X[::2], X[1::2], ks=range(2, 7))
    assert k in (3, 4, 5, 6) and len(table) == 5
    assert table.loc[table["k"] == 3, "bic"].iloc[0] < table.loc[table["k"] == 2, "bic"].iloc[0]


def test_bootstrap_stability_is_high_on_separated_classes():
    df = simulate_shapes(n_per=2000)
    X = df[shapes.FEATURES].to_numpy()
    assert mixture.stability(X, 3, n_boot=3, sample=8000)["ari_min"] > 0.9


def test_label_agreement_arithmetic():
    types = pd.Series(["ST", "ST", "SL", "SL"])
    out = mixture.label_agreement(types, np.array([True, False, True, False]))
    assert out["st_share"] == 0.5 and out["sweeper_like_share"] == 0.5
    assert out["st_that_are_sweeper_like"] == 0.5 and out["sweeper_like_that_are_st"] == 0.5


def test_changepoint_detected_when_a_sweeper_is_added():
    rng = np.random.default_rng(1)
    x = np.r_[rng.random(200) < 0.02, rng.random(200) < 0.9].astype(float)
    r = mixture.detect_changepoint(x)
    assert abs(r["index"] - 200) <= 10 and r["lr"] > 100
    assert r["before"] < 0.1 and r["after"] > 0.8


def test_no_changepoint_in_a_stationary_series():
    rng = np.random.default_rng(2)
    r = mixture.detect_changepoint((rng.random(400) < 0.4).astype(float))
    assert r["lr"] < 15


def test_review_queue_lists_ambiguous_and_disagreeing_pitchers():
    n = 200
    pool = pd.DataFrame({
        "pitcher": [1] * n + [2] * n + [3] * n, "player_name": ["A"] * n + ["B"] * n + ["C"] * n,
        "game_year": 2024, "pitch_type": ["SL"] * n + ["ST"] * n + ["SL"] * n,
        "hb_in": 5.0, "ivb_in": 0.0, "speed_mph": 85.0,
    })
    prob = np.r_[np.full(n, 0.02), np.full(n, 0.97), np.full(n, 0.5)]
    q = mixture.review_queue(pool, prob)
    assert set(q["pitcher"]) == {3}
