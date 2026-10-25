"""Command line: sweeper-finder fetch | run | demo."""

from __future__ import annotations

import argparse

from sweeper_finder import pipeline
from sweeper_finder.data import statcast


def _demo() -> None:
    from sweeper_finder import shapes
    from sweeper_finder.data.sample import load_sample
    from sweeper_finder.mixture import SweeperMixture

    pool, counts = shapes.slider_pool(load_sample())
    X = pool[shapes.FEATURES].to_numpy()
    mix = SweeperMixture.fit(X, k=4)
    print(counts)
    print(mix.components().to_string(index=False))
    print(f"sweeper-like share: {(mix.sweeper_prob(X) >= 0.5).mean():.1%}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sweeper-finder", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download Statcast days into a cache directory")
    f.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    f.add_argument("--cache", default="data/cache")
    r = sub.add_parser("run", help="label pitches, summarize outcomes, flag under-use")
    r.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    r.add_argument("--cache", default="data/cache")
    r.add_argument("--out", default="reports")
    sub.add_parser("demo", help="fit the mixture on the packaged sample (offline)")
    a = p.parse_args(argv)
    if a.cmd == "fetch":
        for y in a.years:
            statcast.fetch_range(*statcast.SEASONS[y], a.cache)
    elif a.cmd == "demo":
        _demo()
    else:
        print(pipeline.run(a.cache, a.out, tuple(a.years)))
    return 0
