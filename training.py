"""Train one model at one learning rate and seed, without experiment orchestration."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from metrics import macro_mae
from models import SteeringModel, forward_masked

MODEL_NAMES = ("single_frame", "lstm", "cfc", "ltc", "transformer")


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 3e-4
    seed: int = 0
    epochs: int = 60
    patience: int = 8
    batch_size: int = 256
    keep_rates: tuple = (1.0, 0.5, 0.25)
    ode_unfolds: int = 24

    def validate(self):
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        for name in ("epochs", "patience", "batch_size", "ode_unfolds"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.keep_rates or any(not np.isfinite(k) or not 0 <= k <= 1 for k in self.keep_rates):
            raise ValueError("keep_rates must contain finite values in [0,1]")


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


def fit(model_name, train_data, val_data, *, config=None, checkpoint_path=None):
    """Fit once and return a self-contained checkpoint containing the best validation weights.

    The caller manages learning-rate sweeps, repeated seeds and evaluation. The
    optional checkpoint_path saves only the selected model for inference; it is
    not a resumable training state. See README.md for a complete minimal call.
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
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, batches = 0.0, 0
        for features, timestamps, y in train_data.batches(config.batch_size, shuffle=True, rng=rng):
            keep = float(rng.choice(config.keep_rates))
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
        score = validation_score(model, val_data, mean, std, config.keep_rates)
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation score at epoch {epoch}")
        history.append(dict(epoch=epoch, train_loss=loss_sum / batches, val_macro_mae=score))
        if score < best_score - 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    checkpoint = dict(format_version=1, model_config=model_config, model=best_state,
                      target_mean=mean.cpu(), target_std=std.cpu(),
                      data_config=dict(task=train_data.task, horizons=list(train_data.horizons),
                                       steps=train_data.steps, max_step=train_data.max_step),
                      train_config=asdict(config), best_epoch=best_epoch,
                      best_score=best_score, history=history, parameter_counts=model.blocks())
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    return checkpoint
