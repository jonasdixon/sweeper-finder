import numpy as np

from sweeper_finder import integrity, shapes
from sweeper_finder.data.sample import load_sample
from sweeper_finder.mixture import SweeperMixture


def test_packaged_sample_loads_and_passes_the_mirror_check():
    pool, counts = shapes.slider_pool(load_sample())
    assert counts["modelled"] > 2000
    assert (integrity.mirror_check(pool).to_numpy() > 0).all()


def test_mixture_on_the_sample_finds_a_high_break_group_that_mostly_matches_st():
    pool, _ = shapes.slider_pool(load_sample())
    X = pool[shapes.FEATURES].to_numpy()
    m = SweeperMixture.fit(X, 4)
    like = m.sweeper_prob(X) >= 0.5
    assert 0.25 < like.mean() < 0.5
    st = (pool["pitch_type"] == "ST").to_numpy()
    assert (st & like).sum() / st.sum() > 0.8
    comps = m.components()
    assert comps.loc[comps["sweeper_like"], "hb_in"].min() > comps.loc[~comps["sweeper_like"], "hb_in"].max()
    assert np.isclose(comps["weight"].sum(), 1.0, atol=0.01)
