"""Write the offline sample (random slider-class pitches from 2024) shipped inside the package.

Usage: python scripts/make_sample.py --cache /path/to/cache
"""

import argparse

from sweeper_finder.data.sample import SAMPLE_COLUMNS
from sweeper_finder.data.statcast import SEASONS, load_cached
from sweeper_finder.shapes import POOL_TYPES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rows", type=int, default=2500)
    a = ap.parse_args()
    df = load_cached(a.cache, *SEASONS[2024])
    df = df.loc[df["pitch_type"].isin(POOL_TYPES)].dropna(subset=["pfx_x", "pfx_z", "release_speed"])
    sample = df.sample(a.rows, random_state=0).sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    sample[SAMPLE_COLUMNS].round(4).to_csv("src/sweeper_finder/data/sample_sliders.csv", index=False)
    print(len(sample), "pitches,", sample["pitcher"].nunique(), "pitchers")


if __name__ == "__main__":
    main()
