"""A few thousand real slider-class pitches shipped for offline tests and the quickstart."""

from __future__ import annotations

from importlib import resources

import pandas as pd

SAMPLE_COLUMNS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number", "pitcher", "player_name", "p_throws",
    "stand", "balls", "strikes", "pitch_type", "pfx_x", "pfx_z", "release_speed", "spin_axis", "arm_angle",
    "release_spin_rate", "description", "zone", "plate_x", "plate_z", "sz_top", "sz_bot", "delta_run_exp",
    "estimated_woba_using_speedangle", "woba_value",
]


def load_sample() -> pd.DataFrame:
    with resources.files("sweeper_finder").joinpath("data/sample_sliders.csv").open("rb") as fh:
        return pd.read_csv(fh)
