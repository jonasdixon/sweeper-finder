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

**Plausibility ranges are a judgement.** 65 to 100 mph for a slider-class pitch is wide. It removed
19 pitches. Movement beyond three feet removed none.

## What is not checked

- Spin and tracking error on individual pitches. A pitch measured 2 inches off could cross the
  mixture's boundary; the labels near the boundary are the review queue.
- Whether `pitch_type` `SL` vs `ST` changes over time because Statcast's classifier changed. Pitchers
  with a label jump are listed, but the classifier itself is not audited.
- Arm angle is a tracking estimate, missing for under 1% of pitches.
