"""Gaussian mixture over (glove-side break, induced vertical break, velocity) of slider-class pitches.

The mixture is unsupervised. Statcast's own pitch-type label is used afterwards as an external
check (`label_agreement`), never as the target.

Which components are "sweeper-like" is decided from the mixture's own structure: component means
are split into two groups by weighted 2-means on glove-side break, and the high-break group is
the sweeper-like set. That makes the cut a property of the data (the valley in the break
distribution) and not a number typed in. Components inside the high-break group can differ in
vertical break (riding versus dropping) and are reported separately in the component table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from sweeper_finder.shapes import FEATURES


def choose_k(X_train, X_test, ks=range(2, 9), min_gain: float = 0.001, seed: int = 0) -> tuple[int, pd.DataFrame]:
    """Number of components: the largest k reached before BIC improves by less than `min_gain`
    (as a fraction of BIC) when one more component is added. Held-out log likelihood on `X_test`
    is reported alongside as a second opinion."""
    rows = []
    for k in ks:
        g = GaussianMixture(k, n_init=3, random_state=seed).fit(X_train)
        rows.append({"k": k, "bic": g.bic(X_train), "heldout_loglik": g.score(X_test)})
    table = pd.DataFrame(rows)
    chosen = int(table["k"].iloc[-1])
    for i in range(1, len(table)):
        gain = (table["bic"].iloc[i - 1] - table["bic"].iloc[i]) / abs(table["bic"].iloc[i - 1])
        if gain < min_gain:
            chosen = int(table["k"].iloc[i - 1])
            break
    return chosen, table


def high_break_components(means_hb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Boolean mask of the high-break group from a weighted 2-means split of component means."""
    order = np.argsort(means_hb)
    m, w = means_hb[order], weights[order]
    best, split = np.inf, 1
    for s in range(1, len(m)):
        ss = 0.0
        for grp, gw in ((m[:s], w[:s]), (m[s:], w[s:])):
            c = np.average(grp, weights=gw)
            ss += float(np.sum(gw * (grp - c) ** 2))
        if ss < best:
            best, split = ss, s
    mask = np.zeros(len(m), bool)
    mask[order[split:]] = True
    return mask


@dataclass
class SweeperMixture:
    gmm: GaussianMixture
    sweeper_like: np.ndarray

    @classmethod
    def fit(cls, X, k: int, seed: int = 0) -> SweeperMixture:
        g = GaussianMixture(k, n_init=3, random_state=seed).fit(X)
        return cls(g, high_break_components(g.means_[:, 0], g.weights_))

    def sweeper_prob(self, X) -> np.ndarray:
        return self.gmm.predict_proba(X)[:, self.sweeper_like].sum(axis=1)

    def components(self) -> pd.DataFrame:
        t = pd.DataFrame(self.gmm.means_, columns=FEATURES).round(2)
        t["weight"] = self.gmm.weights_.round(3)
        t["sweeper_like"] = self.sweeper_like
        return t.sort_values("hb_in", ascending=False).reset_index(names="component")


def stability(X, k: int, n_boot: int = 10, sample: int = 60_000, seed: int = 0) -> dict:
    """Refit on bootstrap samples and compare sweeper-like labels on all of X with the full fit
    (adjusted Rand index; 1 is identical)."""
    rng = np.random.default_rng(seed)
    ref = SweeperMixture.fit(X, k, seed).sweeper_prob(X) >= 0.5
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=min(sample, len(X)), replace=True)
        m = SweeperMixture.fit(X[idx], k, seed + b + 1)
        aris.append(adjusted_rand_score(ref, m.sweeper_prob(X) >= 0.5))
    return {"ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)), "n_boot": n_boot}


def k_sensitivity(X, ks=(3, 4, 5, 6, 7, 8), seed: int = 0) -> pd.DataFrame:
    """Share of pitches labelled sweeper-like and agreement with the k = 4 labelling, by k."""
    ref = SweeperMixture.fit(X, 4, seed).sweeper_prob(X) >= 0.5
    rows = []
    for k in ks:
        lab = SweeperMixture.fit(X, k, seed).sweeper_prob(X) >= 0.5
        rows.append({"k": k, "share_sweeper_like": lab.mean(), "ari_vs_k4": adjusted_rand_score(ref, lab)})
    return pd.DataFrame(rows)


def label_agreement(pitch_type: pd.Series, sweeper_like: np.ndarray) -> dict:
    """Agreement between the mixture and Statcast's ST label (an external check)."""
    st = (pitch_type == "ST").to_numpy()
    sl = np.asarray(sweeper_like, bool)
    return {
        "ari": float(adjusted_rand_score(st, sl)),
        "st_share": float(st.mean()),
        "sweeper_like_share": float(sl.mean()),
        "st_that_are_sweeper_like": float((st & sl).sum() / max(st.sum(), 1)),
        "sweeper_like_that_are_st": float((st & sl).sum() / max(sl.sum(), 1)),
    }


def pitcher_disagreements(pool: pd.DataFrame, sweeper_like: np.ndarray, min_pitches: int = 100) -> pd.DataFrame:
    """Pitchers where the share labelled ST and the share labelled sweeper-like differ most."""
    d = pool.assign(st=(pool["pitch_type"] == "ST").to_numpy(), sw=np.asarray(sweeper_like))
    g = d.groupby(["pitcher", "player_name"]).agg(pitches=("st", "size"), st_share=("st", "mean"),
                                                   sweeper_like_share=("sw", "mean"),
                                                   hb_in=("hb_in", "mean"), ivb_in=("ivb_in", "mean"),
                                                   speed_mph=("speed_mph", "mean")).reset_index()
    g = g[g["pitches"] >= min_pitches].copy()
    g["gap"] = g["sweeper_like_share"] - g["st_share"]
    return g.reindex(g["gap"].abs().sort_values(ascending=False).index).round(3)


def detect_changepoint(x: np.ndarray, min_segment: int = 30) -> dict:
    """Single change in a 0/1 series by Bernoulli likelihood ratio. Returns the split index, the
    likelihood-ratio statistic 2*(LL_split - LL_constant) and the rate before and after."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2 * min_segment:
        return {"index": None, "lr": 0.0, "before": np.nan, "after": np.nan}
    cs = np.cumsum(x)
    total = cs[-1]

    def ll(s, m):
        p = np.clip(s / m, 1e-9, 1 - 1e-9)
        return s * np.log(p) + (m - s) * np.log(1 - p)

    idx = np.arange(min_segment, n - min_segment + 1)
    s1, m1 = cs[idx - 1], idx.astype(float)
    s2, m2 = total - s1, n - m1
    stat = 2 * (ll(s1, m1) + ll(s2, m2) - ll(total, n))
    j = int(np.argmax(stat))
    return {"index": int(idx[j]), "lr": float(stat[j]), "before": float(s1[j] / m1[j]), "after": float(s2[j] / m2[j])}


def changepoint_flags(pool: pd.DataFrame, sweeper_like: np.ndarray, min_pitches: int = 120,
                      lr_threshold: float = 25.0, min_shift: float = 0.5) -> pd.DataFrame:
    """Pitchers whose sweeper-like rate shifts abruptly (a sweeper added, dropped or relabelled)."""
    d = pool.assign(sw=np.asarray(sweeper_like)).sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"])
    rows = []
    for (pitcher, name), g in d.groupby(["pitcher", "player_name"]):
        if len(g) < min_pitches:
            continue
        r = detect_changepoint(g["sw"].to_numpy())
        if r["lr"] >= lr_threshold and abs(r["after"] - r["before"]) >= min_shift:
            rows.append({"pitcher": pitcher, "name": name, "pitches": len(g),
                         "change_date": g["game_date"].iloc[r["index"]], "lr": round(r["lr"], 1),
                         "rate_before": round(r["before"], 2), "rate_after": round(r["after"], 2)})
    return pd.DataFrame(rows)


def review_queue(pool: pd.DataFrame, prob: np.ndarray, min_pitches: int = 75) -> pd.DataFrame:
    """Pitcher-seasons a person should look at: many pitches near the 50% boundary, a mix of
    sweeper-like and not, or a Statcast label that disagrees with the mixture."""
    d = pool.assign(p=prob, ambiguous=(prob > 0.2) & (prob < 0.8), st=(pool["pitch_type"] == "ST"))
    g = d.groupby(["pitcher", "player_name", "game_year"]).agg(
        pitches=("p", "size"), mean_prob=("p", "mean"), ambiguous_share=("ambiguous", "mean"),
        st_share=("st", "mean"), hb_in=("hb_in", "mean"), ivb_in=("ivb_in", "mean"),
        speed_mph=("speed_mph", "mean"),
    ).reset_index()
    g = g[g["pitches"] >= min_pitches].copy()
    g["label_gap"] = (g["mean_prob"] - g["st_share"]).abs()
    reasons = np.select(
        [g["ambiguous_share"] >= 0.35, g["label_gap"] >= 0.4],
        ["many pitches near the boundary", "Statcast label disagrees with the mixture"], default="")
    g["reason"] = reasons
    q = g[g["reason"] != ""].sort_values(["ambiguous_share", "label_gap"], ascending=False)
    return q.round(3).reset_index(drop=True)
