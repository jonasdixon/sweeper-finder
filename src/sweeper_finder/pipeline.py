"""End-to-end run: load -> integrity -> mixture -> label checks -> outcomes -> usage -> flags."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from sweeper_finder import integrity, mixture, outcomes, shapes, usage, viz
from sweeper_finder.data.statcast import load_seasons
from sweeper_finder.mixture import SweeperMixture
from sweeper_finder.shapes import FEATURES

SPLITS = {
    "all batters": lambda d: d.index == d.index,
    "same-handed batters": lambda d: (d["stand"] == d["p_throws"]).to_numpy(),
    "opposite-handed batters": lambda d: (d["stand"] != d["p_throws"]).to_numpy(),
    "two strikes": lambda d: (d["strikes"] == 2).to_numpy(),
}


def label_pitches(
    pitches: pd.DataFrame, years: tuple[int, ...], out_tables: Path
) -> tuple[pd.DataFrame, pd.DataFrame, SweeperMixture, dict]:
    pool, counts = shapes.slider_pool(pitches)
    X = pool[FEATURES].to_numpy()
    first = (pool["game_year"] == years[0]).to_numpy()
    k, k_table = mixture.choose_k(X[first], X[~first])
    k_table.to_csv(out_tables / "component_count.csv", index=False)
    mix = SweeperMixture.fit(X, k)
    prob = mix.sweeper_prob(X)
    like = prob >= 0.5
    pool = pool.assign(sweeper_prob=prob, sweeper_like=like)

    m_first = SweeperMixture.fit(X[first], k)
    transfer = adjusted_rand_score(like[~first], m_first.sweeper_prob(X[~first]) >= 0.5)
    per_hand = np.zeros(len(pool), bool)
    for hand in ("R", "L"):
        sel = (pool["p_throws"] == hand).to_numpy()
        per_hand[sel] = SweeperMixture.fit(X[sel], k).sweeper_prob(X[sel]) >= 0.5
    ksens = mixture.k_sensitivity(X)
    ksens.to_csv(out_tables / "component_count_sensitivity.csv", index=False)
    diag = {
        "k": k, "pool_filter_counts": counts, "sweeper_like_share": float(like.mean()),
        "stability_bootstrap": mixture.stability(X, k),
        "transfer_ari_fit_on_first_season_applied_to_second": float(transfer),
        "pooled_vs_per_hand_fit_ari": float(adjusted_rand_score(like, per_hand)),
        "agreement_with_statcast_ST": mixture.label_agreement(pool["pitch_type"], like),
    }
    mix.components().to_csv(out_tables / "components.csv", index=False)
    mixture.pitcher_disagreements(pool, like).head(60).to_csv(out_tables / "st_label_disagreements.csv", index=False)
    mixture.changepoint_flags(pool, like).to_csv(out_tables / "changepoints.csv", index=False)
    mixture.review_queue(pool, prob).head(200).to_csv(out_tables / "review_queue.csv", index=False)
    integrity.label_switches(pool).to_csv(out_tables / "st_label_switches.csv", index=False)
    return pool, k_table, mix, diag


def outcome_tables(pitches: pd.DataFrame, out_tables: Path, figs: Path) -> pd.DataFrame:
    groups = {
        "sweeper-like": pitches["sweeper_like"],
        "other sliders": pitches["in_pool"] & ~pitches["sweeper_like"],
        "all other pitches": ~pitches["in_pool"],
        "Statcast ST label": pitches["pitch_type"] == "ST",
    }
    rows = []
    for name, mask in groups.items():
        r = outcomes.rates(outcomes.aggregate(pitches.loc[mask].assign(_g=name), ["_g"]))
        rows.append(r)
    pd.concat(rows).rename(columns={"_g": "group"}).round(4).to_csv(out_tables / "outcomes_by_group.csv", index=False)

    contrasts = []
    for split, pick in SPLITS.items():
        sub = pitches.loc[pick(pitches)]
        for versus, frame in (("other sliders", sub.loc[sub["in_pool"]]), ("all other pitches", sub)):
            wide = outcomes.pitcher_flag_table(frame, "sweeper_like")
            for metric in outcomes.METRICS:
                contrasts.append({"split": split, "versus": versus,
                                  **outcomes.within_pitcher_contrast(wide, metric)})
    table = pd.DataFrame(contrasts)
    table.round(4).to_csv(out_tables / "within_pitcher_contrasts.csv", index=False)
    viz.contrast_forest(table.loc[table["split"].isin(["all batters", "same-handed batters", "opposite-handed batters"])],
                        str(figs / "within_pitcher_contrasts.png"))
    return table


def leaders(pitches: pd.DataFrame, flagged: pd.DataFrame, out_tables: Path) -> pd.DataFrame:
    key = ["pitcher", "player_name", "game_year"]
    agg = outcomes.rates(outcomes.aggregate(pitches.loc[pitches["sweeper_like"]], key))
    a_csw, b_csw = outcomes.beta_binomial_prior(agg["csw_n"], agg["n"])
    a_wh, b_wh = outcomes.beta_binomial_prior(agg["whiffs"], agg["swings"])
    agg["csw_shrunk"] = outcomes.shrink_rate(agg["csw_n"], agg["n"], a_csw, b_csw)
    agg["whiff_shrunk"] = outcomes.shrink_rate(agg["whiffs"], agg["swings"], a_wh, b_wh)
    cols = key + ["n_total", "n_sweeper", "usage", "expected_usage", "usage_gap", "rv100", "rv100_post", "rv100_post_sd",
                  "p_better_than_league", "flag_underused"]
    out = flagged[cols].merge(agg[key + ["csw", "csw_shrunk", "whiff", "whiff_shrunk", "chase", "xwobacon"]],
                              on=key, how="left", suffixes=("", "_raw"))
    out = out.sort_values("rv100_post", ascending=False)
    out.round(4).to_csv(out_tables / "sweeper_pitcher_seasons.csv", index=False)
    return out


def run(cache: str, out_dir: str, years: tuple[int, ...] = (2023, 2024)) -> dict:
    out = Path(out_dir)
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    raw = load_seasons(cache, years)
    pitches = outcomes.add_outcomes(raw)
    pool, _, mix, diag = label_pitches(pitches, years, tables)
    diag["integrity"] = integrity.report(raw, pool, diag["pool_filter_counts"])
    pitches = pitches.assign(in_pool=False, sweeper_like=False, sweeper_prob=np.nan)
    pitches.loc[pool.index, "in_pool"] = True
    pitches.loc[pool.index, "sweeper_like"] = pool["sweeper_like"]
    pitches.loc[pool.index, "sweeper_prob"] = pool["sweeper_prob"]
    viz.movement_map(pool, mix, str(figs / "movement_map.png"))

    contrasts = outcome_tables(pitches, tables, figs)
    table = usage.pitcher_season_table(pitches)
    expected = usage.fit_expected_usage(table)
    slope = usage.usage_quality_slope(table)
    flagged = usage.flag_underused(table, expected, slope)
    leaders(pitches, flagged, tables)
    viz.quality_vs_usage(flagged, str(figs / "quality_vs_usage.png"))
    viz.usage_change(table, slope, str(figs / "usage_change.png"))

    diag["usage_quality_slope"] = slope
    diag["pitcher_seasons_assessed"] = len(flagged)
    diag["pitcher_seasons_flagged"] = int(flagged["flag_underused"].sum())
    diag["within_pitcher_headline"] = contrasts.loc[
        (contrasts["split"] == "all batters") & (contrasts["versus"] == "all other pitches")
    ].round(4).to_dict(orient="records")
    (tables / "summary.json").write_text(json.dumps(diag, indent=1, default=float))
    return diag
