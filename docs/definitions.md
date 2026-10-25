# Definitions

Fixed before fitting. Where a definition is a choice, `design_decisions.md` says what else was
possible.

## Slider-class pitch

A pitch Statcast labels `SL` (slider), `ST` (sweeper) or `SV` (slurve). Only these are candidates
for the mixture. Curveballs and cutters are excluded: a sweeper thrown by a pitcher whose
tracking label is `CU` would be missed, and the review queue is where that would show up.

## Shape

Three numbers per pitch, all from the tracking system:

- **Glove-side break** (inches): horizontal movement, positive toward the pitcher's glove side for both
  hands. In this Statcast export `pfx_x` is positive for a right-hander's glove-side break and
  negative for a left-hander's, so left-handers' values are flipped. The check that the flip is
  right is in `integrity.mirror_check`.
- **Induced vertical break** (inches): vertical movement with gravity removed (`pfx_z`). Positive
  means the pitch falls less than a spinless pitch would.
- **Velocity** (mph): `release_speed`.

## Sweeper-like

A pitch is sweeper-like if its posterior probability of belonging to a high-break mixture component
is at least 0.5. The mixture is fit to the shapes of all slider-class pitches. The components are split
into a high-break and a low-break group by a weighted 2-means on the components' mean glove-side
break; the high-break group is the sweeper-like set. This makes the cut a property of the
data, not a number typed in.

"Sweeper" in baseball usage means a slider with a lot of horizontal sweep. The mixture's
sweeper-like set is broader than that in one respect: it also contains riding high-break sliders
(vertical break near +4 inches) and slurve-shaped pitches (vertical break near -1), because they
share large glove-side break. They are separate components in `reports/tables/components.csv`.

Statcast's `ST` label is **not** the definition and is not used in fitting. It is compared against
afterwards (`mixture.label_agreement`).

## Outcomes

- **CSW:** called strikes plus whiffs, per pitch.
- **Whiff:** swings and misses per swing. `foul_tip` is a swing, not a whiff.
- **Chase:** swings at pitches outside the zone, per pitch outside the zone.
- **xwOBAcon:** mean expected wOBA on balls in play (Statcast's estimate from exit velocity and launch angle;
  the actual wOBA value is used when no estimate exists).
- **Runs saved per 100 pitches (rv100):** from the change in run expectancy on each pitch, signed so that
  positive is good for the pitcher.

## Usage

Sweeper-like pitches as a share of all of that pitcher's pitches in the season.

## Under-used

A pitcher-season is flagged only if all three hold:

1. His sweeper-like pitch is better than the league's, with at least an 80% posterior probability
   (shrunken runs saved per 100).
2. His usage is below what pitchers with the same hand, arm angle, platoon exposure and
   repertoire breadth use, by more than noise and by at least 3 points.
3. Raising usage to that expected level would still leave him above league even if each +10
   points of usage costs the pessimistic end of the estimated within-pitcher usage-quality slope.

A flag is a hypothesis for a person to check against video and the pitcher's own usage
reasons. It is not advice to throw the pitch more.
