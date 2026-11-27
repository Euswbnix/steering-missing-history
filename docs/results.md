# Reported results

These tables summarise completed experiments. They were not rerun by the core-code
refactor. Errors are macro MAE in steering-wheel degrees. The original current-angle
and anticipation test results average eight training seeds. The last two sections
are separate, exploratory validation analyses: a training comparison with three
paired seeds per model, and a check on windows with naturally missing frames. See [experiments.md](experiments.md) for the original experiment settings
and statistical qualifications.

## Current-angle prediction

Test evaluation completed on 2026-09-25, on 21 sessions in 16 route groups.

| Model | Full history | Keep 50% | Keep 25% |
|---|---:|---:|---:|
| LSTM with elapsed time | 16.000 | 16.020 | 16.315 |
| CfC | 17.702 | 16.665 | 16.459 |
| LTC, 24 integration substeps | 18.857 | 18.965 | 19.376 |
| Transformer, descriptive | 14.906 | 14.998 | 15.312 |

The primary CfC-minus-LSTM difference at keep 25% is +0.144 degrees, with a 95%
interval [-0.170, +0.440], inside the specified +/-0.5 degree equivalence bound.
This supports practical equivalence for that comparison, not superiority of CfC.
CfC is worse under full and half history, and LTC is worse across conditions in
the Holm-corrected secondary comparisons. Transformer results are descriptive
because its configuration was selected after seeing validation comparisons.

Validation-only retraining with elapsed times divided by the median frame interval
did not meet the predeclared criterion for a material improvement. That exploratory
check does not establish that time units are irrelevant in general. A fixed-weight
LTC 24-versus-48-substep diagnostic was within 0.1 degree in the checked conditions;
this is not a universal numerical-convergence claim.

## Anticipation

Test evaluation completed on 2026-10-08, using the same 21 sessions and common
target frames valid at all four horizons.

| Model | Retention | 0 s | 0.5 s | 1 s | 2 s |
|---|---|---:|---:|---:|---:|
| Single-frame | No history | 21.770 | 20.600 | 20.761 | 23.370 |
| LSTM | Full | 17.805 | 18.152 | 19.454 | 23.278 |
| LSTM | 50% | 17.857 | 18.221 | 19.511 | 23.278 |
| LSTM | 25% | 18.255 | 18.524 | 19.718 | 23.308 |
| CfC | Full | 19.397 | 19.103 | 20.096 | 23.815 |
| CfC | 50% | 18.423 | 18.578 | 19.770 | 23.644 |
| CfC | 25% | 18.080 | 18.460 | 19.739 | 23.652 |
| LTC | Full | 20.912 | 21.007 | 21.926 | 24.897 |
| LTC | 50% | 21.014 | 20.995 | 21.913 | 24.721 |
| LTC | 25% | 21.324 | 21.115 | 21.962 | 24.707 |
| Transformer | Full | 16.261 | 16.778 | 18.084 | 22.228 |
| Transformer | 50% | 16.325 | 16.784 | 18.066 | 22.222 |
| Transformer | 25% | 16.624 | 16.958 | 18.185 | 22.329 |

The history gain, single-frame MAE minus full-history LSTM MAE, is +3.965 degrees
at 0 s and +1.307 degrees at 1 s. The primary difference in gains is -2.658 degrees,
95% interval [-3.577, -1.854]: history helps 1 s anticipation less than current-angle
prediction. At 2 s the gain is +0.092 degrees.

For anticipation, CfC versus LSTM is inconclusive at 1 s after multiple-comparison
correction; this is not evidence of equivalence. LTC is worse and the Transformer
is better in the corresponding secondary comparisons. The extension reused
sessions already tested for current-angle prediction, which limits claims about
an untouched holdout.

## History-dropping training comparison (exploratory validation)

This comparison contains 24 completed current-angle runs: four models, two
training conditions and seeds 0, 1 and 2 in each condition. Random-drop training
samples a history-retention rate for each minibatch; full-history training always
retains the history. Both conditions were evaluated on the same 188,961 validation
windows. The test split was not evaluated. These results are separate from the
eight-seed test results above. The [training protocol](no_drop_training.md)
documents the settings, original source revision and post-run additions.

### Evaluation

Both training conditions use the same checkpoint-selection rule: the lowest
qualifying validation macro MAE averaged over 100%, 50% and 25% history retention.
Selection uses one fixed mask, seed 7, at each sparse retention rate. Training
stops after eight epochs without a score decrease greater than 1e-6 degrees,
with a maximum of 60 epochs.

Final scoring evaluates the selected checkpoint once at 100% retention and with
five masks at each sparse rate, using seeds 7, 1007, 2007, 3007 and 4007. Scores
are averaged across masks within a training seed before averaging the three
training seeds. Retention is the probability of keeping each history frame;
the current frame is always retained. Macro MAE averages the MAEs of the nonempty
absolute-angle bins [0, 5), [5, 15) and [15, infinity) degrees with equal weight.

The primary retention condition for this exploratory comparison is 25%; 50% and
100% are secondary conditions. In the tables, delta is full-history training
minus random-drop training. A positive delta means random-drop training had
lower validation macro MAE under the specified learning-rate settings. All
differences are computed before rounding.

### Mean validation macro MAE

| Model | History retained | Random-drop training (degrees) | Full-history training (degrees) | Delta (degrees) |
|---|---:|---:|---:|---:|
| LSTM | 100% | 15.178 | 14.994 | -0.184 |
| LSTM | 50% | 15.283 | 15.666 | +0.382 |
| LSTM | 25% | 15.706 | 17.104 | +1.398 |
| CfC | 100% | 17.851 | 17.643 | -0.208 |
| CfC | 50% | 16.575 | 18.279 | +1.704 |
| CfC | 25% | 16.360 | 20.999 | +4.639 |
| LTC | 100% | 19.099 | 19.896 | +0.797 |
| LTC | 50% | 19.199 | 21.554 | +2.355 |
| LTC | 25% | 19.631 | 23.569 | +3.938 |
| Transformer | 100% | 14.744 | 15.040 | +0.296 |
| Transformer | 50% | 14.861 | 15.283 | +0.423 |
| Transformer | 25% | 15.218 | 15.933 | +0.715 |

### Paired differences by training seed

| Model | History retained | Seed 0 delta (degrees) | Seed 1 delta (degrees) | Seed 2 delta (degrees) |
|---|---:|---:|---:|---:|
| LSTM | 100% | +0.017 | -0.135 | -0.435 |
| LSTM | 50% | +0.559 | +0.537 | +0.051 |
| LSTM | 25% | +1.620 | +1.750 | +0.824 |
| CfC | 100% | -0.247 | -0.153 | -0.222 |
| CfC | 50% | +1.517 | +1.753 | +1.842 |
| CfC | 25% | +4.252 | +4.582 | +5.083 |
| LTC | 100% | +3.417 | -0.627 | -0.399 |
| LTC | 50% | +4.715 | +1.506 | +0.843 |
| LTC | 25% | +6.181 | +3.093 | +2.541 |
| Transformer | 100% | +0.642 | +0.296 | -0.050 |
| Transformer | 50% | +0.812 | +0.390 | +0.067 |
| Transformer | 25% | +1.216 | +0.606 | +0.322 |

### Error increase from 100% to 25% retention

Each increase is macro MAE at 25% retention minus macro MAE at 100%, averaged
over the three seeds. A negative value means the error decreased. The final
column compares those increases between the two training conditions.

| Model | Random-drop training (degrees) | Full-history training (degrees) | Full minus random (degrees) |
|---|---:|---:|---:|
| LSTM | +0.528 | +2.110 | +1.582 |
| CfC | -1.491 | +3.356 | +4.847 |
| LTC | +0.532 | +3.673 | +3.141 |
| Transformer | +0.474 | +0.893 | +0.419 |

### Interpretation and limitations

At 25% retention, random-drop training reduced mean validation macro MAE by
1.398 degrees for LSTM, 4.639 for CfC, 3.938 for LTC and 0.715 for Transformer.
It had lower error in all 12 same-seed model comparisons at 25% retention and
also in all 12 at 50% retention.

At 100% retention, random-drop training was slightly worse on average for LSTM
(MAE higher by 0.184 degrees) and CfC (higher by 0.208 degrees). LSTM favoured
full-history training in two of three seeds, and CfC in all three. LTC's mean
improvement of 0.797 degrees was driven by seed 0, which improved by 3.417 degrees;
seeds 1 and 2 instead became worse by 0.627 and 0.399 degrees. Transformer improved
by 0.296 degrees on average, but its seed 2 result was slightly worse. Random-drop
training was therefore not uniformly better across retention rates or seeds.

Both conditions reused learning rates selected under random-drop training in the
original experiments: LSTM and Transformer 3e-4, CfC 1e-4 and LTC 1e-3.
Full-history training was not tuned independently, and no additional learning-rate
sweep was performed for this comparison. Differences may partly depend on that
choice, so these results compare the policies under fixed settings rather than
their best performance after separate tuning.

This is a three-seed exploratory validation comparison. No confidence intervals,
p-values or formal equivalence tests were computed for it. The original study's
0.5-degree margin is only a reference scale here; neither a small mean difference
nor three same-sign paired differences establishes statistical significance or
equivalence. The observed validation differences do not establish a test-set
improvement.

## Windows with naturally missing frames (exploratory validation)

Some windows already have missing frames in the original recordings: somewhere in
the 30-frame window, two neighbouring frames are more than 0.2 s apart. The main
evaluation leaves these windows out. This check scores them separately, using the
eight-seed current-angle checkpoints with full history, so no frames are dropped
on top of the real gaps. It uses the validation split only and was run after the
test split was sealed; the test split was not evaluated. It is exploratory. The
[plan](natural_gap_plan.md) was written before the run.

The validation split has 188,961 ordinary windows and 1,855 windows with missing
frames, about 1%. The windows with missing frames come from only 5 of the 7
validation route groups, and in every one of them the largest gap is between 0.2
and 0.5 s. Before scoring, each model's restored checkpoint reproduced its stored
validation predictions on one session to within 3.85e-4 degrees.

| Model (learning rate) | Missing-frame windows | Ordinary windows | Missing-frame minus reweighted ordinary | All windows minus ordinary |
|---|---:|---:|---:|---:|
| LSTM (3e-4) | 15.96 ± 0.73 | 15.44 ± 0.20 | +2.37 [-2.58, +6.45] | -0.001 [-0.076, +0.117] |
| CfC (1e-4) | 16.74 ± 0.73 | 17.42 ± 0.41 | +1.47 [-4.31, +5.00] | -0.019 [-0.096, +0.084] |
| LTC (1e-3) | 15.55 ± 1.42 | 18.55 ± 0.67 | -0.75 [-4.98, +1.18] | -0.059 [-0.122, -0.005] |
| Transformer (3e-4) | 15.46 ± 0.49 | 14.76 ± 0.24 | +2.40 [-1.39, +5.29] | -0.000 [-0.051, +0.068] |

Scores are macro MAE in degrees, the mean ± standard deviation over the eight
seeds. Brackets are 95% intervals from resampling whole route groups (2,000
draws). "Reweighted ordinary" gives the ordinary windows the same speed mix as the
missing-frame windows within each angle bin. "All windows minus ordinary" is how
much the reported score would change if the missing-frame windows were kept.

Leaving these windows out did not make the reported score look better: adding them
back changes it by at most 0.06 degrees for every model, because they are so few.
Whether real missing frames make the error larger is unclear, since every interval
in the reweighted comparison includes zero. The model order also differs: LTC is
last on ordinary windows but second on missing-frame windows. We do not read
anything into this, because LTC's score on those windows varies the most between
seeds (standard deviation 1.42 degrees).

The check covers only 5 route groups and gaps of at most 0.5 s, so it says nothing
about longer interruptions. The Transformer results stay descriptive, as above.
