# Requirements

## Technical

- T1. Statcast pitch data for the 2023 and 2024 regular seasons, downloaded one day at a time (Savant caps an
  export at 25,000 rows) and cached; an interrupted download resumes.
- T2. The shape model uses only tracking fields (glove-side break, induced vertical break,
  velocity). The Statcast pitch-type label is an external check, never an input to the fit.
- T3. Every filter is counted. Implausible speeds or movements are removed and reported.
- T4. The mixture's component count, sweeper-like cut, stability and agreement with the Statcast
  label are all reported with numbers; none is assumed.
- T5. Raw Statcast data is never committed. A small sample and result tables are.

## User

- U1. A reader can see which pitchers throw a sweeper-like slider and how confident the label is.
- U2. A reader can compare the sweeper-like pitch with the *same pitcher's* other pitches, so a good
  pitcher's overall quality is not mistaken for a pitch's quality.
- U3. A reader can see rates with sample size and shrinkage, not raw rates on 40 pitches.
- U4. Pitchers whose label is doubtful (near the boundary, or the Statcast label disagrees) are
  listed for a human to look at; they are not silently decided.
- U5. Under-use flags carry intervals and the assumptions they rest on, and say they are hypotheses.

## Functional

- F1. `sweeper-finder fetch` downloads and caches; `run` labels pitches, builds the outcome tables
  and writes tables and figures.
- F2. The mixture exposes a pitch-level probability, a per-pitcher-season mean probability, and a
  hard label at 0.5.
- F3. Pitcher-level rates use beta-binomial (rates) or normal (run value) shrinkage toward the
  league.
- F4. A pitcher appearing to add or drop a sweeper mid-season is detected from the sequence of
  labels and reported.
- F5. Under-use flagging needs only the pitcher-season table, an expected-usage predictor and the
  usage-quality slope, so each part can be replaced.
