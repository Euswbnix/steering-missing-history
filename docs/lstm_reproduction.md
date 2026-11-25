# LSTM reproduction check

On November 24, 2026, a current-angle LSTM run was repeated to investigate the
roughly 0.36-degree difference between the original experiment and an earlier
reproduction. This note records that completed diagnostic; preparing this PR did
not start another experiment.

## Code and settings

The rerun used a separate snapshot based on commit
`def284fec102cbe99f8747ed29b66e8023e5bd57`. Its `experiments/sparse_run.py`
SHA-256 was
`1a85e2fd866e37f105e67bd10c0db32f08afb80ae2eaeb7e10412423229650ea`.

| Setting | Value |
|---|---|
| Model / task | LSTM / current-angle prediction |
| Seed / learning rate | 0 / 3e-4 |
| Batch size / window length | 256 / 30 frames |
| Training retention probabilities | 1, 0.5, 0.25 |
| Maximum epochs / patience | 60 / 8 |
| Training / validation windows | 1,223,312 / 188,961 |
| Evaluation split | Validation only |

Checkpoint selection uses mean macro MAE over the three retention conditions,
with mask seed 7 for each sparse condition. The final validation scores below
average five masks per sparse condition, using seeds 7, 1007, 2007, 3007 and 4007.
No test data was evaluated, and none of the 24 history-training comparison runs
was repeated for this check.

The recorded rerun environment was Python 3.13.5, PyTorch 2.9.0+cu128,
NumPy 2.4.6, CUDA 12.8 and cuDNN 9.10.2, on an NVIDIA RTX 5090.

## Results

| Run | Stopping epoch | Best validation selection macro MAE (degrees) |
|---|---:|---:|
| Original experiment | 27 | 15.4193080679 |
| Earlier reproduction | 17 | 15.7780430181 |
| November 24 rerun | 17 | 15.7780430181 |

The rerun matched the earlier reproduction and was `0.3587349502` degrees above
the original selection score. Training and final validation took about 6.87
minutes in total.

| History retention | Original validation macro MAE (degrees) | Earlier reproduction and rerun (degrees) |
|---|---:|---:|
| 100% | 15.322690 | 15.657829 |
| 50% | 15.283589 | 15.628851 |
| 25% | 15.609624 | 16.009083 |

## Diagnostic checks

- The two reproductions had exactly matching best and last model weights,
  prediction arrays, and all 17 logged epoch scores.
- Evaluating the original checkpoint in the current code and environment
  recovered its `15.4193080679`-degree selection score. Predictions for all 11
  validation condition/mask combinations matched the original stored arrays.
  The largest macro-MAE difference was `1.78e-15` degrees, consistent with
  floating-point summation roundoff.
- The original checkpoint and result files were unchanged. An AST comparison
  found the same implementations of `Cached`, `SteeringModel`, `masked`,
  `forward_masked`, `predict`, `score_all`, `condition_metrics` and `train` in the
  original and published current-angle driver snapshots.

The discrepancy is between the weights produced by training; it is not explained
by evaluating the original checkpoint in the current setup. The specific cause
of the different training outcome remains unresolved. Old dependency lists and
timing records are historical documents, not an environment snapshot of the
original seed-0 run. Identifying the cause requires its actual environment,
launch settings and any continuation history.

The full diagnostic logs, checkpoints and prediction arrays remain on the
workstation. This repository contains only the settings and aggregate findings.
