# Natural-gap evaluation: plan

Written before the evaluation was run. Exploratory; validation split only. Run after the test
split was sealed. The test split is not used, and whether to use it is a decision for the whole
group.

## Why

Some windows already have missing frames in the original recordings: somewhere in the window, two
neighbouring frames are more than 0.2 s apart. In the main experiment we remove frames at random,
but here the frames were really lost, which is closer to the question we are asking. The main
evaluation leaves these windows out, so its score tells us nothing about them. We want to know two
things:

1. How do the four models score on windows with a real break?
2. Does leaving those windows out make the reported score look better than it should?

## Which windows

Same as the formal evaluation except for the largest step in the window:

| Rule | Formal windows | Natural-gap windows |
|---|---|---|
| Target frame | speed above 3 m/s, finite angle | same |
| Window | 30 frames ending at the target | same |
| Largest step between neighbouring frames | at most 0.2 s | above 0.2 s |
| Every step | not checked | positive and finite (a repeated or reversed timestamp is a clock fault, not a gap) |
| Whole window span | not capped | at most 10 s |

The two sets cannot overlap. The span cap removes windows that stretch over a very long
interruption; the number removed is reported. Splitting is unchanged: validation sessions only.

## What is scored

* Models: LSTM, CfC, LTC, Transformer; seeds 0 to 7 each.
* Checkpoints: the ones the main experiment saved under `results/sparse/ckpt/`, at the learning rate in
  `<model>_lr_chosen.json`, using `best_state`. LTC uses 24 unfolds (file name ends `_u24`).
* Angle scaling: mean and standard deviation of the train windows' labels, as in training.
* Retention 1.0: no frame is dropped on top of the real break.
* Metric: macro MAE (degrees), unchanged.
* Before anything is scored, the restored checkpoint is checked against the main experiment's stored
  validation predictions on one session (maximum difference must be below 0.05 degrees).
* The independent LSTM reproduction can be run the same way (`--lr lstm=1e-4 --results-dir` pointing at
  its output) and is reported separately from the main experiment's checkpoints.

## How the results are cut

* **Largest step in the window**, three bins: above 0.2 up to 0.5 s, above 0.5 up to 1 s, above 1 s.
  A bin with fewer than 100 windows gets its count only, no score.
* **Sample-selection check**, three comparisons (all differences in degrees; positive means the
  second item is worse):
  1. natural-gap windows minus ordinary windows, as they are;
  2. natural-gap windows minus ordinary windows re-weighted over speed (3 to 10, 10 to 20, above
     20 m/s) to the gap windows' speed mix, inside each angle bin (angle bins are averaged
     equally by macro MAE, so the angle mix needs no correction);
  3. all eligible windows (ordinary and natural-gap) minus ordinary windows only: what the formal
     number would have been had the gap windows been kept.
* **Uncertainty**: 95% intervals from resampling whole route groups with replacement (2000 draws,
  fixed random seed). The validation split has 7 route groups, so intervals are expected to be wide.
  Scores are the mean over the 8 seeds, with the standard deviation across seeds.
* Nothing else is tuned or selected from these numbers.

## What would count as a finding

* Comparison 3 close to zero, with a narrow interval, means leaving the gap windows out did not
  change the reported number.
* Comparison 2 clearly above zero (interval excludes zero) would suggest real gaps hurt beyond
  what their speed and angle mix explains. An interval that spans zero is reported as inconclusive.
* A model ranking that differs on gap windows from the ranking on ordinary windows is noted, not
  explained.

## Outputs and confidentiality

* Script: `experiments/natural_gap_eval.py`; tests: `tests/test_natural_gap.py`.
* Output folder outside any git tree. Only summary numbers are written (counts, scores, intervals).
  No per-window prediction, session name or timestamp leaves the server.
* Result file: `natural_gap_val.json`. The results document must say this was carried out after the
  seal, and is exploratory.
* If a bin or the whole set is too small to score, only the count is reported.

## Command

    python experiments/natural_gap_eval.py --cache /srv/csc413/hdd/cache/dinov2_10hz \
        --results-dir /srv/csc413/hdd/results/sparse --out ~/workspace/natural-gap-4models
