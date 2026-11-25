# Transformer training curves

The figures were prepared on November 24, 2026 from existing final-run logs.
They cover current-angle prediction and anticipation, with seeds 0–7 for each
task. They are from the original experiments, not the later comparison of
training with and without history dropping.

| Task | Selected learning rate | Runs | Recorded epochs across runs |
|---|---:|---:|---:|
| Current-angle prediction | 3e-4 | 8 | 207 |
| Anticipation | 1e-4 | 8 | 206 |

## Figures and metrics

- Current angle: [PNG](../figures/transformer_current_training.png),
  [SVG](../figures/transformer_current_training.svg).
- Anticipation: [PNG](../figures/transformer_anticipation_training.png),
  [SVG](../figures/transformer_anticipation_training.svg).

Seed 0 is highlighted in blue as a fixed display choice; seeds 1–7 are gray.
Each line ends at its last logged epoch. No smoothing, interpolation or averaging
across seeds is applied. The circle marks seed 0's selected checkpoint using the
original improvement threshold of `1e-6`.

`train/loss` is the average of minibatch MSE values on train-standardised targets.
It has no degree units. For anticipation it includes the four outputs at
0, 0.5, 1 and 2 s.

`val/mean_macro_mae` is in degrees. It averages macro MAE over retention
probabilities 1, 0.5 and 0.25. For anticipation it also averages the three future
horizons, excluding the 0 s output. Per-epoch selection uses one fixed mask
(seed 7) per sparse condition. These are selection scores, not the five-mask
final evaluation scores. The panels use different metrics, so the vertical
distance between training and validation curves is not a generalisation gap.

## Sources and checks

The source files are `swanlab_scalars.csv` and `val_curves_final.csv` in the
workstation's `exports/training_curves_2026-11-01/` directory. The script matches
each task, selected learning rate and seed to one run identity. It aligns metrics
using the scalar export's `step` field, as logged by the original training loop.

All 413 validation points matched the independent validation export within its
three-decimal rounding: the largest absolute difference was
`0.0004991139281464996` degrees, below the `0.00050001` tolerance. Both metrics
have complete matching epoch sequences for all 16 runs, with no duplicate steps
or non-finite values.

| Source | SHA-256 |
|---|---|
| `swanlab_scalars.csv` | `9e897b15f2f7ebb51ca0ed4dffeb2dcb8b3a948b64eb7c45a11ba02fa875c074` |
| `val_curves_final.csv` | `4f0832942198969b7078ca5fc128df43487d224acca40b1b06f41b37ae5ca0b3` |

## Plotting from the existing logs

Install `requirements-dev.txt`, then run the following on the workstation with
access to the original exports. Set `--out` to a new directory outside any Git
working tree; the script checks this and refuses to overwrite existing outputs.

```console
python experiments/plot_transformer_curves.py \
  --scalars /srv/csc413/hdd/exports/training_curves_2026-11-01/swanlab_scalars.csv \
  --validation /srv/csc413/hdd/exports/training_curves_2026-11-01/val_curves_final.csv \
  --out "$HOME/transformer-curves-export"
```

This only reads CSV files and plots their recorded values. It does not train a
model, run inference or evaluate a data split. The output includes PNG/SVG figures,
a summary of checks and source hashes, and a ZIP for local review. Only the four
figures and the plotting script are included here; the original CSV files and
generated summary/ZIP remain outside the repository.
