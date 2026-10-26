# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
pytest
```

`pytest` should be green before you change anything. If it is not, say so in the issue rather than
working around it.

## What this repository is careful about

The project assigns a label — "sweeper-like" — that does not exist in the data. Everything here is
built so that the label can be argued with. Three habits follow from that, and changes are reviewed
against them.

**A grouping method always returns groups.** A mixture model will split one population as readily
as two. So the claim that sweepers are a distinct group has to be supported outside the model:
`eda.valley` looks for a dip in the one-dimensional break distribution, and the pipeline compares
where that dip falls with where the mixture's own probability crosses 0.5. They currently agree to
about an inch. If you change the features, the pool filters or the component count, re-run and
check that they still agree. If they stop agreeing, that is a result worth reporting, not a bug to
suppress.

**Report the strength of a finding, not just its direction.** The dip is real and shallow: the
density at the low point is 85% of the smaller mode. An earlier version of the histogram used bin
widths that were not multiples of the export's rounding step, which made the dip look about a fifth
deeper. The corrected, less impressive number is the one in the README. Please keep that direction
of error — when a check is sensitive to an arbitrary choice, show the sensitivity table rather than
picking the best-looking setting.

**Break lives on a 0.12-inch lattice.** `pfx_x` is exported rounded to 0.01 ft. Any new histogram,
binned rate or density must use a bin width that is a whole multiple of `eda.BREAK_QUANTUM_IN`, or
it will show a comb of spikes that is an artifact of the export. `eda.break_quantisation`
recomputes the step from the data; if a future export changes it, that function will say so.

## Things not to do

- Do not dichotomize a continuous variable to make a comparison easier. Break, velocity and usage
  are continuous and are modelled that way. The one place a cut is made — the sweeper-like label —
  carries a probability alongside it and a review queue for the pitches near the line.
- Do not compare a pitcher's sweeper with the league. Compare it with his own other pitches. The
  headline result exists because those two comparisons disagree.
- Do not report a rate without its denominator. `outcomes.py` keeps swings, pitches and
  out-of-zone pitches separate on purpose; whiffs per swing and whiffs per pitch are different
  quantities and have been confused before.
- Do not add a dependency for something small.

## Tests

New analysis code needs a test that would fail if the code were wrong, not one that restates it.
The existing pattern is to plant a known answer and recover it: `test_eda.py` plants two
populations a known distance apart and checks the dip is found between them, and plants a single
population and checks that no boundary is reported. `test_integrity.py` plants data bugs — a
flipped handedness sign, duplicated rows — and checks the run stops.

Tests must not reach the network or read the Statcast cache. The packaged 2,500-pitch sample and
generated data cover everything.

## Before opening a pull request

```bash
ruff check src tests scripts
pytest
```

If you changed anything the pipeline writes, re-run it against a cache and commit the regenerated
tables and figures in the same commit, so the numbers in the README always match the files:

```bash
sweeper-finder run --cache data/cache --out reports
```

Then say in the description which numbers in the README changed and why. If none changed, say that
too — it is useful information.
