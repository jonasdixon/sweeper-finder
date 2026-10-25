# Design decisions

## Unsupervised mixture, with Statcast's label as a check

**Chosen.** A Gaussian mixture on three tracking fields, fit without labels.

**Why.** The project question is which pitches *behave* like sweepers. Statcast's `ST` label
is itself an output of a classifier on similar tracking data; fitting to it would reproduce it and
learn nothing independent. Comparing against it afterwards shows where this approach differs.

**Rejected.**
- Supervised classification of the `ST` label. Circular, as above.
- A fixed rule (for example glove break above 13 inches). It hides the choice of threshold.
- Clustering on pitcher averages. Throws away within-pitcher variation, which is where
  mid-season changes show up.

**Weakness.** A mixture does not know what a sweeper is. It finds shape groups, and
the high-break group contains riding and dropping shapes. The sweeper-like set is therefore a
broader family than the pitch most people mean by the word, and the Statcast label overlaps it only
partly (see Agreement below). Spin axis and arm angle were left out of the fit on purpose: they
carry real information about how the pitch is thrown, but they make the clusters depend on arm slot
and would blur the movement-and-velocity definition in the project brief.

## How many components

**Chosen.** Add components until BIC improves by less than 0.1% when one more is added. That gives
four on the pooled 2023-24 data.

**Evidence.** BIC keeps falling slowly past four, to eight, by tiny amounts. Held-out log
likelihood (fit 2023, scored 2024) improves by less than 0.011 per pitch between four and eight
components (`component_count.csv`). The number of components
is not sharply identified. The *labelling* is nonetheless stable: the share of pitches
called sweeper-like is 34% to 40% for k = 3 to 8, and agreement with the k = 4 labelling is 0.86 to
0.95 (adjusted Rand index; `component_count_sensitivity.csv`), lowest at k = 5.

## Splitting components into sweeper-like or not

**Chosen.** Weighted 2-means on the components' mean glove-side break; components in the
high-break group are sweeper-like.

**Rejected.** Picking the single highest-break component. With k = 4, two components share
similar horizontal break and differ in vertical break, and one of them would be left out arbitrarily.

## Pooled over both hands, after mirroring

A single mixture is fit to mirrored data. A mixture fit separately for each hand gives labels that
agree with the pooled ones at an adjusted Rand index of 0.96, so the pooling is not hiding a
hand-specific difference in shape.

## Stability checks

- Ten bootstrap refits on 60,000-pitch samples, each used to label all pitches: adjusted Rand
  index against the full fit averages 0.983 (minimum 0.973).
- Fit on 2023 only, applied to 2024, compared with the pooled-fit labels for 2024: 0.954.

## Agreement with Statcast's `ST` label

38.2% of slider-class pitches are sweeper-like and 30.5% are labelled `ST`. Of `ST` pitches,
90% are sweeper-like; of sweeper-like pitches, 72% are `ST`. Adjusted Rand index 0.52.

The largest disagreements are in `st_label_disagreements.csv`. Two kinds stand out: pitchers
whose sliders have a lot of glove-side break but no `ST` label, which includes some submarine and
low-slot pitchers with very high vertical break, and pitchers labelled `ST` at moderate break
(5 to 9 inches), which the mixture does not call sweeper-like.
Neither is obviously an error by the mixture or the label; they are why the review queue exists.

## Outcomes and the within-pitcher comparison

**Chosen.** Compare each pitcher's sweeper-like pitches with his own other pitches and average the
differences, weighted by n_s * n_o / (n_s + n_o) (sweeper-like and other pitch counts), with a
pitcher-level bootstrap interval.

**Why.** Pitchers who throw a sweeper are not random pitchers. Comparing league rates for
sweepers with league rates for everything else mixes pitch quality with pitcher quality.

**Weakness.** Within a pitcher, the sweeper is not randomly assigned to counts, batters or
locations either; a pitcher may save it for particular situations. The comparison is
split by batter hand and by two strikes to show how the pattern changes, not to remove it.

**What it shows.** Compared with *all* of a pitcher's other pitches the sweeper-like pitch gets
more whiffs per swing (+9.8 points) and more CSW (+3.0 points). Compared with his *other
sliders*, the whiff and CSW differences vanish (+0.5 and +0.6 points, both intervals include zero). What remains is
less damage on contact (xwOBAcon -0.023). So most of the apparent sweeper advantage is a
breaking-ball advantage.

## Shrinkage

Rates are shrunk with a beta-binomial prior estimated by method of moments across pitcher-seasons;
run value and xwOBAcon use a normal prior. Method-of-moments is crude (it ignores the
uncertainty in the prior), but the prior is estimated from several hundred pitcher-seasons, which
makes the error small compared with the sampling noise on a few hundred pitches.

## Usage and under-use

**Expected usage** is a fractional logistic regression on hand, arm angle, platoon exposure and the
number of pitch types a pitcher uses at least 5% of the time. It is a comparison with peers
of similar profile, not a model of optimal usage.

**Diminishing returns.** The slope of a pitcher's sweeper run value on his usage, differenced
across seasons so that stable pitcher quality drops out. With 128 pitchers who had at least 75
sweeper-like pitches in both seasons, it is -0.09 runs per 100 pitches per +10 points of usage
(95% interval -0.45 to +0.29). That is no evidence either way. The flag uses the pessimistic end,
-0.45, as a stress test.

**Weaknesses.**
- Usage changes between years are not random. A pitcher who throws more sweepers may be reacting to
  something that also changes quality (a grip change, an injury, a new catcher).
- Run value per 100 sweepers is noisy at this sample size (posterior standard deviation about 0.6 runs
  against a league mean around 0.7), so very few pitchers clear 80% posterior probability of being
  better than league.
- Quality is runs saved, which includes the count and sequence the pitch was thrown in.

## What a flag means

8 of 377 pitcher-seasons with at least 100 sweeper-like pitches are flagged. A flag
says: this pitch looks better than league, this usage looks low for the profile, and a
pessimistic usage penalty would not erase the advantage. It does not say what would
happen if usage were raised; a pitcher, a catcher and a pitching coach know things this data does
not.
