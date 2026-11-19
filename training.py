"""Train one model at one learning rate and seed, without experiment orchestration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from metrics import macro_mae
from models import SteeringModel, forward_masked

MODEL_NAMES = ("single_frame", "lstm", "cfc", "ltc", "transformer")


def _plain(value):
    """Replace NumPy scalars with Python numbers, inside dicts, lists and tuples too.

    torch.load(weights_only=True) rejects NumPy scalars, so a checkpoint saved with
    them, for example a learning rate taken from a NumPy grid, could not be loaded.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class TrainConfig:
    """Omitted validation rates use keep_rates. A training override fixes the
    applied rate while preserving the batch-order RNG draws for a paired run.
    """
    lr: float = 3e-4
    seed: int = 0
    epochs: int = 60
    patience: int = 8
    batch_size: int = 256
    keep_rates: tuple = (1.0, 0.5, 0.25)
    ode_unfolds: int = 24
    validation_keep_rates: tuple | None = None
    train_keep_override: float | None = None

    def __post_init__(self):
        for name in ("lr", "seed", "epochs", "patience", "batch_size", "ode_unfolds", "train_keep_override"):
            object.__setattr__(self, name, _plain(getattr(self, name)))
        for name in ("keep_rates", "validation_keep_rates"):
            rates = getattr(self, name)
            if rates is not None:
                object.__setattr__(self, name, tuple(_plain(k) for k in rates))

    def validate(self):
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        for name in ("epochs", "patience", "batch_size", "ode_unfolds"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, rates in (("keep_rates", self.keep_rates), ("validation_keep_rates", self.validation_keep_rates)):
            if name == "validation_keep_rates" and rates is None:
                continue
            if not rates or any(not np.isfinite(k) or not 0 <= k <= 1 for k in rates):
                raise ValueError(f"{name} must contain finite values in [0,1]")
        if self.train_keep_override is not None and not 0 <= self.train_keep_override <= 1:
            raise ValueError("train_keep_override must be a finite value in [0,1]")


def predict_data(model, data, mean, std, *, keep=1.0, mask_seed=7):
    """Predict every cache window in labels()/sessions() order, without updating weights."""
    generator = torch.Generator(device=data.device).manual_seed(mask_seed)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, timestamps, _ in data.batches(512):
            out = forward_masked(model, features, timestamps, keep, generator)
            predictions.append((out * std + mean).cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def validation_score(model, data, mean, std, keep_rates):
    """Mean macro MAE across conditions and future horizons, using one shared mask."""
    target = data.labels()
    values = []
    for keep in keep_rates:
        prediction = predict_data(model, data, mean, std, keep=keep)
        if target.ndim == 1:
            values.append(macro_mae(prediction, target))
        else:
            # Anticipation selection uses 0.5, 1 and 2 s, excluding the 0 s head.
            values.append(np.mean([macro_mae(prediction[:, k], target[:, k])
                                   for k in range(1, target.shape[1])]))
    return float(np.mean(values))


def fit(model_name, train_data, val_data, *, config=None, checkpoint_path=None, on_epoch=None):
    """Fit once and return a self-contained checkpoint containing the best validation weights.

    The caller manages learning-rate sweeps, repeated seeds and evaluation. The
    optional checkpoint_path saves only the selected model for inference; it is
    not a resumable training state. See README.md for a complete minimal call.
    An optional on_epoch callback receives a copy of each completed epoch's
    history entry, including the final epoch when early stopping triggers.
    """
    config = config or TrainConfig()
    config.validate()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}")
    if train_data.split != "train" or val_data.split != "val":
        raise ValueError("fit requires train and val splits; the test split cannot train or select weights")
    for name in ("feature_dim", "horizons", "steps", "max_step", "device"):
        if getattr(train_data, name) != getattr(val_data, name):
            raise ValueError(f"train and validation must share {name}")
    training_sessions = {item["session"] for item in train_data.items}
    if training_sessions.intersection(item["session"] for item in val_data.items):
        raise ValueError("train and validation sessions must be disjoint")
    path = Path(checkpoint_path) if checkpoint_path is not None else None
    if path is not None and path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    target = train_data.labels()
    mean = torch.tensor(target.mean(0), dtype=torch.float32, device=train_data.device)
    std = torch.tensor(target.std(0), dtype=torch.float32, device=train_data.device)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("every training target must have finite statistics and nonzero variance")
    torch.manual_seed(config.seed)
    model_config = dict(kind=model_name, dim=train_data.feature_dim, hidden=64, seed=config.seed,
                        ode_unfolds=config.ode_unfolds, dt_unit=1.0, time_code="real",
                        n_out=len(train_data.horizons))
    model = SteeringModel(**model_config).to(train_data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=train_data.device).manual_seed(config.seed)
    best_score, best_state, best_epoch, stale = np.inf, None, 0, 0
    history = []
    validation_rates = config.validation_keep_rates or config.keep_rates
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for features, timestamps, y in train_data.batches(config.batch_size, shuffle=True, rng=rng):
            keep = float(rng.choice(config.keep_rates))
            if config.train_keep_override is not None:
                # The draw above keeps the shared shuffle RNG aligned between training conditions.
                keep = config.train_keep_override
            prediction = forward_masked(model, features, timestamps, keep, generator)
            loss = ((prediction - (y - mean) / std) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if model_name == "ltc":
                model.rnn.cell.apply_weight_constraints()
            loss_sum += float(loss.detach())
            batches += 1
        score = validation_score(model, val_data, mean, std, validation_rates)
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation score at epoch {epoch}")
        history.append(dict(epoch=epoch, train_loss=loss_sum / batches, val_macro_mae=score))
        if score < best_score - 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if on_epoch is not None:
            on_epoch(history[-1].copy())
        if stale >= config.patience:
            break
    checkpoint = dict(format_version=1, model_config=model_config, model=best_state,
                      target_mean=mean.cpu(), target_std=std.cpu(),
                      data_config=_plain(dict(task=train_data.task, horizons=list(train_data.horizons),
                                             steps=train_data.steps, max_step=train_data.max_step)),
                      train_config=asdict(config), best_epoch=best_epoch,
                      best_score=best_score, history=history, parameter_counts=model.blocks())
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    return checkpoint
