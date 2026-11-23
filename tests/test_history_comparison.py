"""Restart and output handling for the validation comparison driver."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_history_comparison as comparison
from training import TrainConfig


def forbid_training(*args, **kwargs):
    pytest.fail("a completed checkpoint must not be trained again")


def saved_checkpoint(out, *, kind="lstm", seed=0):
    config = TrainConfig(lr=3e-4, seed=seed, keep_rates=comparison.KEEPS,
                         validation_keep_rates=comparison.KEEPS, train_keep_override=1.0)
    checkpoint = dict(model_config={"kind": kind}, train_config=asdict(config),
                      best_epoch=1, best_score=2.0,
                      history=[{"epoch": 1, "train_loss": 1.0, "val_macro_mae": 2.0}])
    directory = out / "lstm" / "seed_0" / "full_history"
    directory.mkdir(parents=True)
    torch.save(checkpoint, directory / "checkpoint.pt")
    return directory


def test_completed_runs_are_skipped_without_training_or_scoring(tmp_path, monkeypatch):
    cache, out = tmp_path / "cache", tmp_path / "outputs"
    cache.mkdir()
    out.mkdir()
    manifest = {"task": "current", "split": "val"}
    (out / "manifest.json").write_text(json.dumps(manifest))
    for model, lr in comparison.LEARNING_RATES.items():
        for seed in comparison.SEEDS:
            for policy in comparison.POLICIES:
                directory = out / model / f"seed_{seed}" / policy
                directory.mkdir(parents=True)
                # Completed jobs should not even attempt to deserialize their checkpoint.
                (directory / "checkpoint.pt").write_bytes(b"already completed")
                row = dict(model=model, seed=seed, policy=policy, learning_rate=lr,
                           conditions={f"{keep:g}": {"macro_mae": 2.0}
                                       for keep in comparison.KEEPS})
                (directory / "result.json").write_text(json.dumps(row))

    splits = []

    def load_cache(path, split, *, task, device):
        assert path == cache and task == "current" and device == "cpu"
        splits.append(split)
        return SimpleNamespace(labels=lambda: np.array([0.0, 10.0, 20.0]))

    monkeypatch.setattr(comparison, "FeatureCache", load_cache)
    monkeypatch.setattr(comparison, "manifest", lambda *args: manifest)
    monkeypatch.setattr(comparison.os, "umask", lambda mask: 0o022)
    monkeypatch.setattr(comparison, "fit", forbid_training)
    monkeypatch.setattr(comparison, "run_one", forbid_training)
    monkeypatch.setattr(comparison.sys, "argv", ["run_history_comparison.py", "--cache", str(cache),
                                               "--out", str(out), "--device", "cpu"])
    comparison.main()

    status = json.loads((out / "status.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert splits == ["train", "val"]
    assert status["state"] == "complete" and status["completed_runs"] == 24
    assert summary["completed_runs"] == summary["planned_runs"] == len(summary["runs"]) == 24
    assert set(summary["comparisons"]) == set(comparison.LEARNING_RATES)


def test_checkpoint_without_result_is_scored_without_training(tmp_path, monkeypatch):
    directory = saved_checkpoint(tmp_path)
    val, targets = object(), np.array([0.0, 10.0, 20.0])
    conditions = {"1": {"macro_mae": 1.5}, "0.5": {"macro_mae": 2.0}, "0.25": {"macro_mae": 2.5}}
    scored = []

    def score(checkpoint, data, labels):
        assert data is val and labels is targets
        scored.append(checkpoint["best_epoch"])
        return conditions

    monkeypatch.setattr(comparison, "fit", forbid_training)
    monkeypatch.setattr(comparison, "score_checkpoint", score)
    row = comparison.run_one(tmp_path, "lstm", 0, "full_history", None, val, targets)

    assert scored == [1]
    assert row["conditions"] == conditions
    assert row["best_epoch"] == row["epochs"] == 1
    assert json.loads((directory / "result.json").read_text()) == row


@pytest.mark.parametrize("settings", [{"kind": "cfc"}, {"seed": 1}])
def test_mismatched_checkpoint_is_rejected_before_scoring(tmp_path, monkeypatch, settings):
    directory = saved_checkpoint(tmp_path, **settings)
    monkeypatch.setattr(comparison, "fit", forbid_training)
    monkeypatch.setattr(comparison, "score_checkpoint", forbid_training)
    with pytest.raises(ValueError, match="checkpoint configuration mismatch"):
        comparison.run_one(tmp_path, "lstm", 0, "full_history", None, None, None)
    assert not (directory / "result.json").exists()


@pytest.mark.parametrize("location", ["source", "git_tree", "cache"])
def test_output_guard_rejects_protected_paths_before_loading_data(tmp_path, monkeypatch, capsys, location):
    cache, source = tmp_path / "cache", tmp_path / "source"
    cache.mkdir()
    source.mkdir()
    git_tree = tmp_path / "other_checkout"
    (git_tree / ".git").mkdir(parents=True)
    out = {"source": source, "git_tree": git_tree, "cache": cache}[location] / "outputs"
    monkeypatch.setattr(comparison, "ROOT", source)
    monkeypatch.setattr(comparison, "FeatureCache", forbid_training)
    monkeypatch.setattr(comparison.sys, "argv", ["run_history_comparison.py", "--cache", str(cache),
                                               "--out", str(out), "--device", "cpu"])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2
    assert "--out must be outside" in capsys.readouterr().err
    assert not out.exists()
