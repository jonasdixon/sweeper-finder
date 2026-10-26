# Data quality

Numbers are from `reports/tables/summary.json`, written by `sweeper-finder run` over the 2023 and
2024 regular seasons.

## Where the pitches went

| Step | Pitches |
|---|---|
| Competitive pitches (pitchouts and automatic balls/strikes removed) | 1,427,357 |
| Slider class (`SL`, `ST`, `SV`) | 322,410 |
| Missing shape data | 0 |
| Speed outside 65-100 mph | 19 |
| Movement beyond 36 inches in either direction | 0 |
| **Pitches in the mixture** | **322,391** |

## Findings

**The sign of `pfx_x` depends on handedness.** Right-handers' glove-side break is positive and
left-handers' is negative in this export. Unmirrored, a right-hander's sweeper sits near +14
inches and a left-hander's near -14, so a pooled mixture would need a separate set of components
for each hand. After mirroring, the mean glove-side break of every
slider-class type is positive for both hands:

| | SL | ST | SV |
|---|---|---|---|
| Right-handed | 4.65 in | 13.91 in | 14.28 in |
| Left-handed | 4.69 in | 13.90 in | 12.86 in |

The tests plant a flipped sign for one hand and confirm that the check catches it.

Equal means would not be enough on their own, since two differently shaped distributions can share
a mean. `eda.mirror_gap` compares the two hands decile by decile: the largest gap is 17.4 inches
before mirroring and 0.6 inches after (`eda_mirror_gap.csv`). The remaining 0.6 inches is five
lattice steps, at the deciles where the distribution is flattest. Pooling the hands is justified.

**Pitch-clock and pitchouts.** 5,226 pitches are automatic balls/strikes or pitchouts and are removed before
any rate is computed (they are not pitch-quality outcomes).

**Duplicates and handedness.** No duplicate (game, at-bat, pitch) keys and no pitcher recorded with
both hands.

**Spin and arm-angle coverage.** Spin rate 99.4%, spin axis 99.4%, arm angle 99.3%. Arm angle is
used only for the expected-usage model; spin is not used. The mixture does not need them.

**Pitchers whose label jumps.** 33 pitchers' share of `ST`-labelled slider-class pitches changes
by 0.5 or more between the first and last third of their pitches (`st_label_switches.csv`). Each is
either a new pitch or a relabel. The two cannot be told apart from the tracking data; a person
should check video before treating either as real. 50 pitchers have an abrupt change in
the mixture's own sweeper-like rate (`changepoints.csv`), for example a pitcher whose rate goes from 0.01 to
0.82 within one season. Some of those are real new pitches.

**Break is recorded on a 0.12-inch lattice.** `pfx_x` is exported rounded to 0.01 ft, so glove-side
break takes only 382 distinct values across 322,391 pitches, spaced 0.12 inches apart. This is
harmless for the mixture, which never bins, but it breaks histograms: a bin width that is not a
whole multiple of 0.12 inches alternately covers two or three lattice points, and the density comes
out as a regular comb of spikes with no physical meaning. It also distorts anything read off a
binned density. With an unaligned 0.25-inch bin the dip between the two break modes measured 0.59
of the smaller mode; with lattice-aligned bins it is 0.83 to 0.86. Every bin width in `eda.py` and
`viz.py` is a multiple of `BREAK_QUANTUM_IN`, and `eda.break_quantisation` recomputes the step from
the data rather than trusting the constant.

**The dip between the two break modes is real but shallow.** Pooled break has modes at 3.7 and 13.3
inches with a low point at 10.0 inches, where the density is about 85% of the smaller mode. Varying
the bin width and the amount of smoothing moves the dip ratio between 0.83 and 0.89, and under the
heaviest smoothing the second mode fails a prominence test and no dip is reported at all
(`eda_break_valley_sensitivity.csv`). So the sweeper-like cut separates two genuine concentrations,
but they overlap a great deal; the cut is not a gap. This is the quantitative reason the review
queue exists.

**Plausibility ranges are a judgement.** 65 to 100 mph for a slider-class pitch is wide. It removed
19 pitches. Movement beyond three feet removed none.

## What is not checked

- Spin and tracking error on individual pitches. A pitch measured 2 inches off could cross the
  mixture's boundary; the labels near the boundary are the review queue. The 0.12-inch export
  rounding is a floor on precision, not an estimate of tracking error, which is larger.
- Whether `pitch_type` `SL` vs `ST` changes over time because Statcast's classifier changed. Pitchers
  with a label jump are listed, but the classifier itself is not audited.
- Arm angle is a tracking estimate, missing for under 1% of pitches.
