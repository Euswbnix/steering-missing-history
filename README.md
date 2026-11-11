# Steering prediction with missing history frames

CSC413 project code for LSTM with elapsed-time input, CfC, LTC and a causal
Transformer. A single-frame MLP provides a no-history baseline. Inputs are
precomputed visual features and camera timestamps; outputs are steering-wheel
angles in degrees, for the current frame or at 0, 0.5, 1 and 2 seconds ahead.

The reported results were produced with the original scripts in
[`experiments/`](experiments/). For a single run, [`training.py`](training.py)
and [`inference.py`](inference.py) provide equivalent training and prediction
interfaces with a simpler checkpoint format.

## Installation

Use Python 3.10 or newer. Install a PyTorch build for your platform, then:

```console
python -m pip install -r requirements.txt
```

Run the examples from this directory. The numerical reference for CfC and the
LTC cell both use `ncps==1.0.1`.

## Train one model

Prepare the cache described in [docs/data.md](docs/data.md). Replace the paths
below with local paths to licensed data and an output directory.

```python
from data import FeatureCache
from training import TrainConfig, fit

cache = "/path/to/features"
train = FeatureCache(cache, "train", device="cpu")
val = FeatureCache(cache, "val", device="cpu")

checkpoint = fit(
    "lstm",  # lstm, cfc, ltc, transformer, or single_frame
    train,
    val,
    config=TrainConfig(lr=3e-4, seed=0),
    checkpoint_path="/path/to/outputs/lstm.pt",
)
print(checkpoint["best_epoch"], checkpoint["best_score"])
```

Use `device="cuda"` for both datasets to train on a CUDA device. For anticipation,
set `task="anticipation"` on both datasets; the same call then trains a four-output
readout. The temporal architecture is unchanged. Training uses mixed history
retention rates, standardised targets, AdamW and validation-based early stopping.
`fit` accepts only the train and validation splits. It saves the best model and
its normalisation/configuration; it does not save a resumable optimiser state.

## Infer from a checkpoint

```python
from inference import Predictor

predictor = Predictor("/path/to/outputs/lstm.pt", device="cpu")

# Supply actual feature windows and their original timestamps from your input pipeline.
# features: (batch, time, feature_dim), float; times: (batch, time), seconds.
angles = predictor.predict(features, times)
print(angles.shape)  # (batch,) for current angles, (batch, 4) for anticipation

# Optional controlled removal of history; the current frame is always retained.
sparse_angles = predictor.predict(features, times, keep=0.25, mask_seed=7)
```

Inference needs no steering labels, speed or data split. To predict a whole
prepared split, use `FeatureCache.batches()` in your own driver and keep the same
batch order as `FeatureCache.labels()` when scoring.

## Layout

- `models/cfc.py`: handwritten CfC cell and sequence wrapper.
- `models/steering.py`: shared projection/readout, temporal models and history masking.
- `data.py`: feature-cache loading and current/future target construction.
- `training.py`: train one configuration and select its best validation weights.
- `inference.py`: restore the model and return predictions in degrees.
- `metrics.py`: the macro MAE used for checkpoint selection.
- `experiments/`: original training, evaluation, statistical analysis and timing scripts.
- `docs/`: [input format](docs/data.md), [architectures](docs/models.md),
  [experiment settings](docs/experiments.md) and [reported results](docs/results.md).

## Verification

```console
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use synthetic inputs. They check the CfC equations and reference outputs,
model parameter counts, causal attention, timestamp handling, labels, training
and checkpoint inference. No driving data or GPU is needed.

The Honda HDD videos, CAN records, feature caches, split manifest and trained
weights are not distributed here. The experiment scripts take frozen features
in the [documented cache format](docs/data.md). See
[docs/experiments.md](docs/experiments.md) for the scripts and settings used
to produce the reported results and the data needed to reproduce them.
