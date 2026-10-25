# Changelog

## [0.1.0]

First release.

- Resumable Statcast download; slider-class pool with counted filters and handedness mirroring.
- Gaussian mixture on glove-side break, induced vertical break and velocity, with a data-derived
  sweeper-like cut; stability, season-transfer and per-hand checks; agreement table against Statcast's
  `ST` label; review queue, label-switch and changepoint lists.
- Outcomes with explicit denominators, shrinkage and within-pitcher comparisons.
- Usage, an expected-usage model, a within-pitcher usage-quality slope and under-use flags with intervals.
- Tests include recovery of planted shape classes for both hands and planted data bugs.
