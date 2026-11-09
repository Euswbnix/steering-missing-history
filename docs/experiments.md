# Experiment settings

This is a standalone description of the experiments behind the reported results,
and a configuration guide for writing a reproduction driver. It is a release
summary, not a newly timestamped preregistration.

## Data and models

Honda HDD driving sessions are held out in whole route groups, so train, validation
and test do not share a route group. Do not replace this with random windows.
The sparse-observation test set contains 21 sessions in 16 route groups, with
402,711 valid target frames. The anticipation test set retains the common targets
valid at all four horizons: 400,457 frames in those same sessions.

Inputs are the frozen features and 30-step windows in [data.md](data.md).
Architectures and parameter counts are in [models.md](models.md). A current-angle
model has one output; an anticipation model has four. No past steering labels
are provided as model inputs.

## Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.01 |
| Loss | Mean squared error of train-standardised angles |
| Target scaling | Mean and population standard deviation from train targets only, separately per output |
| Batch size | 256, grouped by session |
| Epoch limit / early-stopping patience | 60 / 8 |
| History augmentation | One retention rate drawn uniformly per batch from 1, 0.5, 0.25 |
| Current frame | Always kept |
| Learning-rate candidates | 1e-3, 3e-4, 1e-4, 3e-5, each at seed 0 |
| Final seeds | 0 through 7, retrained at each model's selected rate |
| Elapsed-time unit | Seconds |
| LTC integration substeps | 24, with constraints applied after every optimiser step |

Choose the best validation checkpoint using mean macro MAE over the three
retention conditions, with one shared mask per sparse condition (mask seed 7).
For anticipation, average over the 0.5, 1 and 2 s heads as well; exclude the
0 s head from selection. Select each model's learning rate by that same score.
Never use test data for early stopping or learning-rate selection.

Selected learning rates:

| Task | Single-frame | LSTM | CfC | LTC | Transformer |
|---|---|---|---|---|---|
| Current-angle prediction | Not used | 3e-4 | 1e-4 | 1e-3 | 3e-4 |
| Anticipation | 1e-3 | 1e-3 | 1e-4 | 3e-4 | 1e-4 |

`training.fit` performs one seed and learning rate. Write the surrounding loops
and logging as needed. Its selected checkpoint records all architecture arguments,
target scaling, window settings, hyperparameters and validation history. Inference
restores these values directly from the checkpoint.

## Evaluation

Score a finished checkpoint under retention 1, 0.5 and 0.25. For each sparse
condition use five shared masks with seeds 7, 1007, 2007, 3007 and 4007; full
history needs one evaluation. Keep the original timestamps after dropping frames.
Macro MAE is the equal average of MAEs in nonempty ground-truth absolute-angle
bins `[0,5)`, `[5,15)` and `[15,infinity)` degrees. Report mask averages first,
then seed averages. Keep session/route identifiers aligned with each prediction.

The reported uncertainty analysis resampled seeds and route groups in two levels,
with 10,000 bootstrap replicates and sensitivity checks grouped by session and
date. The current-angle primary comparison was CfC minus LSTM under 75% history
removal, using a predeclared practical-equivalence bound of +/-0.5 degrees.
Secondary comparisons used Holm correction. Analysis and experiment drivers are
written by the reproducing researcher.

## Interpretation and reproduction limits

The Transformer configuration for the current-angle experiment was selected after
the other models' validation results were available. Its test result is descriptive
and is excluded from the confirmatory family. Anticipation was designed after the
current-angle test on the same held-out sessions, so those sessions are not a
new untouched holdout for the extension.

The core model equations and state-dictionary layout are retained. Training APIs
and checkpoint packaging are simplified for this release. No result table was
regenerated during that refactor. Byte-identical reruns are not claimed: the
original training used a server-specific environment and some compiled execution
paths. The selected-model checkpoint here is an inference artifact, not the
complete optimiser/RNG state of an interrupted experiment.

The licensed data, frozen feature arrays, exact route split manifest and trained
weights are not bundled. Exact reproduction of the tables requires those artifacts
as well as the settings above. Creating a different route split yields a new
experiment, even if model and training hyperparameters match.
