# Changelog

## [Unreleased]

- `eda.py`: exploratory checks that run before the mixture is fit. Distribution of each shape
  feature, a decile-by-decile test that handedness mirroring worked, recovery of the export's
  rounding step from the data, the dip between the two break modes with a sensitivity table over
  bin width and smoothing, and the relationship between break and whiffs per swing.
- A second mode now has to clear a prominence threshold before a dip is reported, so a single
  smooth distribution no longer yields a spurious boundary. A test plants exactly that case.
- Histogram bin widths are multiples of the 0.12-inch export lattice. Unaligned bins produced a
  comb of spikes and overstated the depth of the dip between the break modes.
- `reports/figures/feature_distributions.png` and `reports/tables/eda_*.csv`; `summary.json` gains
  an `eda` block. No previously reported result changed.
- README reports the dip as real but shallow, and compares the mixture's boundary (8.9 inches of
  break) with the independent density dip (10.0 inches).
- Added `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- Fixed: `python -m sweeper_finder.cli` exited silently without running anything, because the
  module had no `__main__` guard. Only the installed `sweeper-finder` script worked.

## [0.1.0]

First release.

- Resumable Statcast download; slider-class pool with counted filters and handedness mirroring.
- Gaussian mixture on glove-side break, induced vertical break and velocity, with a data-derived
  sweeper-like cut; stability, season-transfer and per-hand checks; agreement table against Statcast's
  `ST` label; review queue, label-switch and changepoint lists.
- Outcomes with explicit denominators, shrinkage and within-pitcher comparisons.
- Usage, an expected-usage model, a within-pitcher usage-quality slope and under-use flags with intervals.
- Tests include recovery of planted shape classes for both hands and planted data bugs.
