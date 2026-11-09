"""Load a model checkpoint and predict steering angles from feature windows."""

from pathlib import Path

import numpy as np
import torch

from models import SteeringModel, forward_masked


class Predictor:
    """Inference in degrees, including the target normalisation saved by fit()."""

    def __init__(self, checkpoint, *, device="cpu"):
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("expected a version-1 checkpoint produced by training.fit")
        self.device = str(torch.device(device))
        self.model_config = dict(checkpoint["model_config"])
        self.steps = checkpoint["data_config"]["steps"]
        self.horizons = tuple(checkpoint["data_config"]["horizons"])
        self.model = SteeringModel(**self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.mean = checkpoint["target_mean"].to(self.device)
        self.std = checkpoint["target_std"].to(self.device)

    def predict(self, features, timestamps, *, keep=1.0, mask_seed=7, batch_size=512):
        """Return (B,) current angles or (B,4) future angles in degrees.

        features is (B,T,D), timestamps is (B,T) in seconds. Frames must be
        chronological, unpadded and end at the current frame. keep controls
        optional independent dropping of history; the current frame survives.
        """
        if not np.isfinite(keep) or not 0 <= keep <= 1 or batch_size < 1:
            raise ValueError("keep must be in [0,1] and batch_size must be positive")
        f = torch.as_tensor(features, dtype=torch.float32)
        t = torch.as_tensor(timestamps, dtype=torch.float64)
        if f.ndim != 3 or t.shape != f.shape[:2] or f.shape[2] != self.model_config["dim"]:
            raise ValueError("expected features (B,T,D) and matching timestamps (B,T)")
        if not len(f) or f.shape[1] < 1 or f.shape[1] > self.steps:
            raise ValueError("provide a nonempty batch with 1 <= T <= the training window length")
        if not torch.isfinite(f).all() or not torch.isfinite(t).all() or not (t[:, 1:] > t[:, :-1]).all():
            raise ValueError("features and times must be finite, with strictly increasing timestamps")
        generator = torch.Generator(device=self.device).manual_seed(mask_seed)
        predictions = []
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(f), batch_size):
                current_f = f[start:start + batch_size].to(self.device)
                current_t = t[start:start + batch_size].to(self.device)
                out = forward_masked(self.model, current_f, current_t, keep, generator)
                predictions.append((out * self.std + self.mean).cpu().numpy())
        return np.concatenate(predictions).astype(np.float32)
