"""Compare random-drop and full-history training on current-angle validation windows.

Run once from the release checkout, for example:
    python scripts/run_history_comparison.py --cache /data/cache --out /data/comparison --device cuda

Re-run the same command to continue. Completed fits can be rescored without
training them again. An interruption during fit restarts that fit from its seed.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import FeatureCache
from inference import Predictor
from metrics import macro_mae
from training import TrainConfig, fit, predict_data


LEARNING_RATES = {"lstm": 3e-4, "cfc": 1e-4, "transformer": 3e-4, "ltc": 1e-3}
POLICIES = ("random_drop", "full_history")
SEEDS = (0, 1, 2)
KEEPS = (1.0, 0.5, 0.25)
MASK_SEEDS = (7, 1007, 2007, 3007, 4007)


def log(message):
    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def manifest(cache, train, val, device):
    # Record train/val array identities without reading the licensed features a second time.
    files = hashlib.sha256()
    for data in (train, val):
        for item in data.items:
            for suffix in ("feats.npy", "meta.npy"):
                file = cache / f"{item['session']}.{suffix}"
                stat = file.stat()
                files.update(f"{file.name} {stat.st_size} {stat.st_mtime_ns}\n".encode())
    source_files = ("data.py", "training.py", "inference.py", "metrics.py",
                    "models/cfc.py", "models/steering.py", "scripts/run_history_comparison.py")
    source = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
    return dict(
        task="current", split="val", cache=str(cache), cache_files_metadata_sha256=files.hexdigest(),
        index_sha256=hashlib.sha256((cache / "index.json").read_bytes()).hexdigest(),
        train_targets=train.n, validation_targets=val.n, feature_dim=train.feature_dim,
        steps=train.steps, max_step=train.max_step, learning_rates=LEARNING_RATES, policies=POLICIES,
        seeds=SEEDS, keep_rates=KEEPS, mask_seeds=MASK_SEEDS, default_training=asdict(TrainConfig()),
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        device=device, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        source_sha256=source,
    )


def score_checkpoint(checkpoint, val, targets):
    predictor = Predictor(checkpoint, device=val.device)
    conditions = {}
    for keep in KEEPS:
        rows = []
        for mask_seed in MASK_SEEDS[:1] if keep == 1 else MASK_SEEDS:
            pred = predict_data(predictor.model, val, predictor.mean, predictor.std,
                                keep=keep, mask_seed=mask_seed)
            rows.append(dict(mask_seed=mask_seed, macro_mae=macro_mae(pred, targets)))
        conditions[f"{keep:g}"] = dict(masks=rows, macro_mae=float(np.mean([r["macro_mae"] for r in rows])))
        log(f"validation keep {keep:g} complete")
    return conditions


def run_one(out, model, seed, policy, train, val, targets):
    directory = out / model / f"seed_{seed}" / policy
    directory.mkdir(parents=True, exist_ok=True)
    ckpt = directory / "checkpoint.pt"
    result = directory / "result.json"

    def status(state, **fields):
        write_json(out / "status.json", dict(model=model, seed=seed, policy=policy, state=state,
                   pid=os.getpid(), updated=datetime.now().astimezone().isoformat(), **fields))

    def log_epoch(row):
        status("training", **row)
        log(f"{model} seed {seed} {policy} epoch {row['epoch']}, val {row['val_macro_mae']:.4f}")

    config = TrainConfig(lr=LEARNING_RATES[model], seed=seed, keep_rates=KEEPS,
                         validation_keep_rates=KEEPS,
                         train_keep_override=1.0 if policy == "full_history" else None)
    if ckpt.is_file():
        log(f"resume validation {model} seed {seed} {policy}")
        checkpoint = torch.load(ckpt, weights_only=True, map_location="cpu")
    else:
        log(f"start fit {model} seed {seed} {policy}")
        status("training", epoch=0)
        checkpoint = fit(model, train, val, config=config, checkpoint_path=ckpt, on_epoch=log_epoch)
    if checkpoint["model_config"]["kind"] != model or checkpoint["train_config"] != asdict(config):
        raise ValueError(f"checkpoint configuration mismatch in {ckpt}")
    log(f"score {model} seed {seed} {policy}, selected epoch {checkpoint['best_epoch']}")
    status("scoring", best_epoch=checkpoint["best_epoch"])
    conditions = score_checkpoint(checkpoint, val, targets)
    row = dict(model=model, seed=seed, policy=policy, learning_rate=config.lr,
               best_epoch=checkpoint["best_epoch"], epochs=len(checkpoint["history"]),
               best_validation_score=checkpoint["best_score"], conditions=conditions)
    write_json(result, row)
    log(f"complete {model} seed {seed} {policy}")
    return row


def write_summary(out, results):
    comparisons = {}
    for model in LEARNING_RATES:
        by_seed = {(r["seed"], r["policy"]): r for r in results if r["model"] == model}
        if not all((seed, policy) in by_seed for seed in SEEDS for policy in POLICIES):
            continue
        scores = {}
        for keep in KEEPS:
            key = f"{keep:g}"
            means = {}
            for policy in POLICIES:
                means[policy] = np.array([by_seed[seed, policy]["conditions"][key]["macro_mae"]
                                          for seed in SEEDS])
            diffs = means["full_history"] - means["random_drop"]
            scores[key] = dict(random_drop=float(means["random_drop"].mean()),
                               full_history=float(means["full_history"].mean()),
                               full_minus_random=float(diffs.mean()), differences_by_seed=diffs.tolist())
        increases = {policy: scores["0.25"][policy] - scores["1"][policy] for policy in POLICIES}
        increases["full_minus_random"] = increases["full_history"] - increases["random_drop"]
        scores["increase_full_to_25_percent"] = increases
        comparisons[model] = scores
    write_json(out / "summary.json",
               dict(completed_runs=len(results), planned_runs=len(LEARNING_RATES) * len(SEEDS) * len(POLICIES),
                    runs=results, comparisons=comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out, cache = args.out.resolve(), args.cache.resolve(strict=True)
    if (ROOT == out or ROOT in out.parents or cache == out or cache in out.parents
            or any((p / ".git").exists() for p in (out, *out.parents))):
        parser.error("--out must be outside the source checkout, Git working trees and the input cache")
    os.umask(0o077)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "another process is already using this output directory\n")
        log(f"load train and validation cache on {args.device}")
        train = FeatureCache(cache, "train", task="current", device=args.device)
        val = FeatureCache(cache, "val", task="current", device=args.device)
        meta = manifest(cache, train, val, args.device)
        path = out / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(meta)):
                raise ValueError("output manifest differs from this run; use a new --out directory")
        else:
            if any((out / model).exists() for model in LEARNING_RATES):
                raise ValueError("found previous run directories without a manifest; use a new --out")
            write_json(path, meta)
        targets = val.labels()
        results = {}
        jobs = list(product(LEARNING_RATES, SEEDS, POLICIES))
        for model, seed, policy in jobs:
            directory = out / model / f"seed_{seed}" / policy
            if (directory / "checkpoint.pt").is_file() and (directory / "result.json").is_file():
                row = json.loads((directory / "result.json").read_text())
                if (row["model"], row["seed"], row["policy"]) != (model, seed, policy):
                    raise ValueError(f"result coordinates do not match {directory}")
                results[model, seed, policy] = row
        write_summary(out, list(results.values()))
        log(f"resume with {len(results)} of {len(jobs)} runs complete")
        for model, seed, policy in jobs:
            key = model, seed, policy
            if key in results:
                continue
            try:
                results[key] = run_one(out, model, seed, policy, train, val, targets)
            except (Exception, KeyboardInterrupt) as exc:
                write_json(out / "status.json", dict(state="interrupted", model=model, seed=seed,
                           policy=policy, pid=os.getpid(), error=f"{type(exc).__name__}: {exc}",
                           updated=datetime.now().astimezone().isoformat()))
                raise
            write_summary(out, list(results.values()))
        write_json(out / "status.json", dict(state="complete", completed_runs=len(results),
                   updated=datetime.now().astimezone().isoformat()))
        log(f"all {len(results)} runs complete")


if __name__ == "__main__":
    main()
