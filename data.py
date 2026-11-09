"""Read prepared per-session features and construct steering windows."""

import json
from pathlib import Path

import numpy as np
import torch

HORIZONS = (0.0, 0.5, 1.0, 2.0)


def future_labels(t, steer, targets, horizons=HORIZONS, max_gap=0.2):
    """Interpolate future angles; invalid, nonmonotone or unsupported labels become NaN.

    Search at most 64 subsequent frames, sufficient for the supported horizons
    on a 10 Hz cache. No interpolation crosses a session boundary.
    """
    t = np.asarray(t, dtype=np.float64)
    steer = np.asarray(steer, dtype=np.float64)
    targets = np.asarray(targets)
    out = np.full((len(targets), len(horizons)), np.nan)
    back = np.concatenate([[0], np.cumsum(np.diff(t) <= 0)])
    ahead = np.minimum(targets[:, None] + np.arange(64)[None], len(t) - 1)
    for k, h in enumerate(horizons):
        if h == 0:
            out[:, k] = steer[targets]
            continue
        tau = t[targets] + h
        reached = t[ahead] >= tau[:, None]
        j1 = ahead[np.arange(len(targets)), reached.argmax(1)]
        j0 = j1 - 1
        gap = t[j1] - t[j0]
        ok = (reached.any(1) & (j1 > targets) & (gap > 0) & (gap <= max_gap)
              & np.isfinite(steer[j0]) & np.isfinite(steer[j1])
              & (back[j1] == back[targets]))
        w = np.divide(tau - t[j0], gap, out=np.zeros_like(gap), where=gap > 0)
        out[ok, k] = (steer[j0] * (1 - w) + steer[j1] * w)[ok]
    return out


class FeatureCache:
    """Memory-mapped features for one split, with session-local windows.

    task='current' produces one angle; task='anticipation' produces four angles
    at the horizons in HORIZONS. See docs/data.md for the file schema.
    """

    def __init__(self, cache, split, *, task="current", steps=30, max_step=0.2,
                 device="cpu"):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if task not in ("current", "anticipation"):
            raise ValueError("task must be current or anticipation")
        if not isinstance(steps, int) or steps < 2 or not np.isfinite(max_step) or max_step <= 0:
            raise ValueError("steps must be an integer >= 2 and max_step must be positive")
        cache = Path(cache)
        with (cache / "index.json").open() as handle:
            index = json.load(handle)
        if not np.isclose(index.get("base_hz", 10.0), 10.0):
            raise ValueError("FeatureCache expects a 10 Hz base-rate cache")
        self.split, self.task = split, task
        self.steps, self.max_step, self.device = steps, max_step, str(torch.device(device))
        self.horizons = (0.0,) if task == "current" else HORIZONS
        self.items, self.feature_dim = [], None
        offsets = np.arange(steps - 1, -1, -1)[None]
        for session, meta in sorted(index["sessions"].items()):
            if meta["split"] != split:
                continue
            feats = np.load(cache / f"{session}.feats.npy", mmap_mode="r", allow_pickle=False)
            rows = np.load(cache / f"{session}.meta.npy", mmap_mode="r", allow_pickle=False)
            if feats.ndim != 2 or rows.ndim != 2 or rows.shape != (len(feats), 3) or feats.shape[1] == 0:
                raise ValueError(f"inconsistent feature/metadata shapes for session {session}")
            if self.feature_dim is not None and self.feature_dim != feats.shape[1]:
                raise ValueError("all sessions must have the same feature dimension")
            self.feature_dim = feats.shape[1]
            t, steer, speed = rows[:, 0], rows[:, 1], rows[:, 2]
            ok = np.isfinite(steer) & np.isfinite(speed) & (speed > 3.0)
            targets = np.flatnonzero(ok)
            targets = targets[targets >= steps - 1]
            if not len(targets):
                continue
            times = t[targets[:, None] - offsets]
            gaps = np.diff(times, axis=1)
            valid = np.isfinite(times).all(1) & (gaps > 0).all(1) & (gaps <= max_step).all(1)
            targets = targets[valid]
            if not len(targets):
                continue
            labels = (steer[targets].astype(np.float32) if task == "current"
                      else future_labels(t, steer, targets, max_gap=max_step).astype(np.float32))
            if task == "anticipation":
                valid = np.isfinite(labels).all(1)
                targets, labels = targets[valid], labels[valid]
            if len(targets):
                self.items.append(dict(session=session, cluster=meta.get("cluster", session),
                                       feats=feats, t=t, targets=targets, labels=labels))
        self.n = sum(len(it["targets"]) for it in self.items)
        if not self.n:
            raise ValueError(f"no valid {task} windows in split {split}")

    def labels(self):
        return np.concatenate([it["labels"] for it in self.items]).astype(np.float64)

    def sessions(self):
        return np.concatenate([[it["session"]] * len(it["targets"]) for it in self.items])

    def clusters(self):
        return np.concatenate([[it["cluster"]] * len(it["targets"]) for it in self.items])

    def batches(self, batch_size, *, shuffle=False, rng=None):
        """Yield features (B,T,D), float64 timestamps (B,T), and labels on device."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if shuffle and rng is None:
            raise ValueError("shuffle requires a NumPy random generator")
        order = np.arange(len(self.items))
        if shuffle:
            rng.shuffle(order)
        for i in order:
            item = self.items[i]
            order_t = (rng.permutation(len(item["targets"])) if shuffle
                       else np.arange(len(item["targets"])))
            for j in range(0, len(order_t), batch_size):
                selected = order_t[j:j + batch_size]
                windows = item["targets"][selected, None] - np.arange(self.steps - 1, -1, -1)[None]
                f = torch.from_numpy(np.ascontiguousarray(item["feats"][windows])).to(self.device).float()
                t = torch.from_numpy(item["t"][windows]).to(self.device)
                y = torch.from_numpy(item["labels"][selected]).to(self.device)
                yield f, t, y
