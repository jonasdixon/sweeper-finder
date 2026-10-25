"""Resumable Statcast download: one parquet file per game day (Savant caps exports at 25,000 rows)."""

from __future__ import annotations

import datetime as dt
import io
import time
from pathlib import Path

import pandas as pd
import requests

try:  # use the OS trust store; some networks re-sign TLS with a private root CA
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

KEEP_COLUMNS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "pitcher", "player_name", "batter", "stand", "p_throws", "balls", "strikes",
    "pitch_type", "pitch_name", "release_speed", "description", "type", "zone",
    "plate_x", "plate_z", "sz_top", "sz_bot", "pfx_x", "pfx_z",
    "release_spin_rate", "spin_axis", "arm_angle",
    "delta_run_exp", "events", "bb_type", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]
SEASONS = {2023: ("2023-03-20", "2023-10-02"), 2024: ("2024-03-20", "2024-10-02")}


def _get(url: str, params: dict, retries: int = 5, pause: float = 2.0) -> requests.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=90)
            if resp.status_code == 200:
                return resp
            last = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_day(day: dt.date) -> pd.DataFrame:
    params = {
        "all": "true", "type": "details", "hfGT": "R|", "player_type": "pitcher",
        "game_date_gt": day.isoformat(), "game_date_lt": day.isoformat(),
        "min_pitches": "0", "min_results": "0",
    }
    text = _get(SAVANT_URL, params).text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= 25_000:
        raise RuntimeError(f"{day}: hit the 25,000-row export cap")
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def fetch_range(start: str, end: str, cache_dir: str | Path) -> None:
    """Download each missing day in [start, end]; off days are stored as empty files."""
    daily = Path(cache_dir) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    day, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while day <= last:
        path = daily / f"{day.isoformat()}.parquet"
        if not path.exists():
            df = fetch_day(day)
            df.to_parquet(path, index=False)
            print(f"{day} {len(df):>6} pitches", flush=True)
            time.sleep(0.5)
        day += dt.timedelta(days=1)


def load_cached(cache_dir: str | Path, start: str, end: str) -> pd.DataFrame:
    daily = Path(cache_dir) / "daily"
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    files = sorted(p for p in daily.glob("*.parquet") if lo <= dt.date.fromisoformat(p.stem) <= hi)
    frames = [f for f in (pd.read_parquet(p) for p in files) if len(f)]
    if not frames:
        raise FileNotFoundError(f"no cached days in {daily} for {start}..{end}")
    df = pd.concat(frames, ignore_index=True)
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def load_seasons(cache_dir: str | Path, years=(2023, 2024)) -> pd.DataFrame:
    return pd.concat([load_cached(cache_dir, *SEASONS[y]) for y in years], ignore_index=True)
