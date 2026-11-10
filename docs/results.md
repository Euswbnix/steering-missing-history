# Reported results

These tables summarise completed experiments. They were not rerun by the core-code
refactor. All values are macro MAE in steering-wheel degrees, averaged over eight
training seeds. See [experiments.md](experiments.md) for configuration, scope and
statistical qualifications.

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
