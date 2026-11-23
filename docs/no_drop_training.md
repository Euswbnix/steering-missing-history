# Does dropping history frames during training improve steering predictions?

Initial exploratory validation protocol: 11 November 2026. The source-revision
audit and learning-rate limitations were added after the experiments, on
16 November 2026. Both are marked as post-run additions below; they were not
part of the initial protocol.

Train two current-angle
models from scratch for each architecture and seed with the public
`training.fit` API. One uses random history dropping and one always sees the
full history. Both runs in a pair start from the same seed. All 24 runs use the
same train and validation route split. The test split stays sealed.

## Source revision (post-run addition, 16 November 2026)

The 24 runs used a working tree based on commit
`12992866a491cce6c770979b3cfeb65ca8d953bf` (`Add README (#3)`), with changes to
`training.py` and the new `scripts/run_history_comparison.py` driver. Those
changes had not been committed when the runs were performed.

The run manifest records source-file SHA-256 hashes, not a Git commit ID.
The seven runtime files in this checkout match those recorded hashes.
`data.py`, `inference.py`, `metrics.py`, `models/cfc.py` and
`models/steering.py` are unchanged from the base commit. The two experiment
source files are preserved byte-for-byte:

| File | SHA-256 recorded for the completed runs |
|---|---|
| `training.py` | `9630404c2cb7f9cdbc2744e26e53fa0bb66a49707338e74e15480e1c43a3aff4` |
| `scripts/run_history_comparison.py` | `0702f1abaade106a2a47108c16b9b633d6c90f9f14076f0f437a136e04fe8274` |

## Running the comparison

Use Linux or macOS; the driver uses `fcntl` to lock its output directory.
Install the repository requirements and prepare the train/validation cache
described in [data.md](data.md). From the repository root, use paths to your
cache and a new output directory outside all Git working trees:

```console
python scripts/run_history_comparison.py --cache /path/to/cache --out /path/to/comparison --device cuda
```

The driver runs the 24 configurations below and opens only the train and
validation splits. It writes a source/environment manifest, progress status,
checkpoints, per-run validation results and a summary to the output directory.
These runtime files stay outside the repository.

Repeating the same command skips runs that have both a checkpoint and a result.
If only the final checkpoint exists, it scores that checkpoint without training
again. A changed manifest is rejected. An interrupted fit without a final
checkpoint starts again from its seed; this driver does not resume optimiser
state or an unfinished epoch. The `on_epoch` callback reports a copy of each
completed history entry for progress monitoring, not a resumable checkpoint.

## Training

| Architecture | Learning rate | Seeds in each condition |
|---|---:|---|
| LSTM | 3e-4 | 0, 1, 2 |
| CfC | 1e-4 | 0, 1, 2 |
| LTC | 1e-3 | 0, 1, 2 |
| Transformer | 3e-4 | 0, 1, 2 |

**Post-run addition, 16 November 2026: learning-rate limitations.**
For each model, both training conditions used the learning rate selected under
random-drop training in the original experiments. Full-history training was not
tuned independently, and no additional learning-rate sweep was performed for
this comparison. The observed differences may therefore partly depend on the
learning-rate choice. These results compare the two training policies under the
stated fixed settings; they do not establish the best performance each policy
could achieve after separate tuning.

For both conditions, keep the default training rates `(1.0, 0.5, 0.25)` and set
`validation_keep_rates=(1.0, 0.5, 0.25)`. In the random-drop run, use the
retention rate sampled for each minibatch. In the full-history run, set
`train_keep_override=1.0`. Apply the override after the usual NumPy draw so
that minibatch shuffling follows the same sequence at a given seed. The current
frame is always retained, and surviving frames keep their original timestamps.

Use 30-frame windows, train-only target scaling, standardised MSE and AdamW
with weight decay 0.01 and batch size 256. For LTC, use 24 integration substeps.
At each epoch, evaluate full, 50% and 25% history on validation, with seed 7
for each sparse mask. Select the checkpoint with the lowest mean macro MAE
across these three rates, with a 60-epoch limit and patience of 8.

For other callers of `fit`, omitting `validation_keep_rates` retains the old
behaviour of validating at `keep_rates`; omitting `train_keep_override` retains
random sampling from those training rates.

## Validation and reporting

Score each selected checkpoint on the same validation windows in the same
order, with evaluation batch size 512. Score full history once. At each sparse
rate, use the five mask seeds 7, 1007, 2007, 3007 and 4007 for both conditions.
Keep checkpoints, histories, validation scores and run settings in a fresh
directory under the workstation user's home, outside Git. Record the code
revision, model, seed, learning rate and training condition for each run.

Compute macro MAE in degrees as the unweighted average of the MAE in each
nonempty absolute steering-angle bin. The bins are below 5°, 5° to below 15°,
and 15° or more. Average masks within a seed before averaging the three seeds. For each
architecture and retention rate, show both condition means and all three
same-seed differences (full-history training minus random-drop training).
A positive difference means random-drop training had lower validation macro MAE
under the specified learning-rate settings.

Use 25% retention for the primary comparison within each architecture. Report
50% and 100% as secondary comparisons, and show how much each condition's
error increases from 100% to 25%. Interpret the mean paired difference
alongside its three seed values and the original study's 0.5° practical
margin. List incomplete seed pairs explicitly. Historical checkpoints and
published scores provide context only.
