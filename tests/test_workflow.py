"""Training-to-inference integration and split isolation on artificial data."""

import json

import numpy as np
import pytest
import torch

import training
from data import FeatureCache
from inference import Predictor
from metrics import macro_mae
from models import SteeringModel
from training import MODEL_NAMES, TrainConfig, fit


def make_cache(root, *, dim=8, n=64):
    root.mkdir()
    rng = np.random.default_rng(12)
    sessions = {}
    for i, split in enumerate(("train", "val", "test")):
        session = f"session_{i}"
        t = 100 * i + 0.1 * np.arange(n)
        steer = 25 * np.sin(0.9 * t) + np.linspace(-3, 3, n)
        f = rng.normal(size=(n, dim)).astype(np.float16)
        np.save(root / f"{session}.feats.npy", f)
        np.save(root / f"{session}.meta.npy", np.column_stack([t, steer, np.full(n, 5.0)]))
        sessions[session] = dict(split=split, cluster=i)
    (root / "index.json").write_text(json.dumps(dict(base_hz=10, sessions=sessions)))
    return root


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("task", ["current", "anticipation"])
def test_train_save_reload_and_sparse_predict(tmp_path, model_name, task):
    cache = make_cache(tmp_path / "cache")
    train = FeatureCache(cache, "train", task=task, steps=6)
    val = FeatureCache(cache, "val", task=task, steps=6)
    test = FeatureCache(cache, "test", task=task, steps=6)
    path = tmp_path / "outputs" / f"{model_name}.pt"
    config = TrainConfig(epochs=2, patience=2, batch_size=64)
    checkpoint = fit(model_name, train, val, config=config, checkpoint_path=path)
    loaded = Predictor(path)
    memory = Predictor(checkpoint)
    f, t, y = next(test.batches(64))
    for keep in (1.0, 0.25, 0.0):
        a = loaded.predict(f, t, keep=keep)
        b = memory.predict(f, t, keep=keep)
        np.testing.assert_array_equal(a, b)
        assert a.shape == tuple(y.shape)
        assert np.isfinite(a).all()
    assert loaded.horizons == test.horizons
    assert checkpoint["best_epoch"] in (1, 2)
    assert len(checkpoint["history"]) == 2
    assert np.isfinite(checkpoint["best_score"])
    np.testing.assert_allclose(checkpoint["target_mean"], train.labels().mean(0), rtol=1e-6)
    np.testing.assert_allclose(checkpoint["target_std"], train.labels().std(0), rtol=1e-6)
    if model_name == "ltc":
        assert checkpoint["model_config"]["ode_unfolds"] == 24
        for name in ("w", "sensory_w", "cm", "gleak"):
            assert (loaded.model.rnn.cell._params[name] >= 0).all()


def test_fit_rejects_test_selection_and_existing_checkpoint(tmp_path):
    cache = make_cache(tmp_path / "cache")
    train, val, test = [FeatureCache(cache, s, steps=6) for s in ("train", "val", "test")]
    with pytest.raises(ValueError, match="test split"):
        fit("lstm", train, test)
    with pytest.raises(ValueError, match="test split"):
        fit("lstm", test, val)
    path = tmp_path / "existing.pt"
    path.write_bytes(b"preserve existing file")
    with pytest.raises(FileExistsError):
        fit("lstm", train, val, checkpoint_path=path)
    assert path.read_bytes() == b"preserve existing file"


def test_full_history_override_keeps_mixed_run_batch_order_and_validation(tmp_path, monkeypatch):
    cache = make_cache(tmp_path / "cache", n=25)
    train, val = [FeatureCache(cache, split, steps=6) for split in ("train", "val")]
    rates = (1.0, 0.5, 0.25)
    original = training.forward_masked

    def trace_run(override):
        batches, train_rates, val_rates = [], [], []

        def trace_forward(model, features, timestamps, keep, generator):
            if model.training:
                batches.append(timestamps[:, -1].tolist())
                train_rates.append(keep)
            else:
                val_rates.append(keep)
            return original(model, features, timestamps, keep, generator)

        config = TrainConfig(epochs=2, batch_size=5, keep_rates=rates, validation_keep_rates=rates,
                             train_keep_override=override)
        with monkeypatch.context() as patch:
            patch.setattr(training, "forward_masked", trace_forward)
            fit("lstm", train, val, config=config)
        return batches, train_rates, val_rates

    mixed_batches, mixed_rates, mixed_val = trace_run(None)
    full_batches, full_rates, full_val = trace_run(1.0)
    assert len(mixed_batches) == 8  # Four batches in each of two shuffled epochs.
    assert full_batches == mixed_batches
    assert min(mixed_rates) < 1.0
    assert set(full_rates) == {1.0}
    assert mixed_val == full_val == list(rates) * 2


@pytest.mark.parametrize("validation_rates, expected", [(None, (1.0,)), ((1.0, 0.5, 0.25), (1.0, 0.5, 0.25))])
def test_selection_uses_configured_rates_or_falls_back_to_training_rates(tmp_path, monkeypatch,
                                                                        validation_rates, expected):
    cache = make_cache(tmp_path / "cache", n=25)
    train, val = [FeatureCache(cache, split, steps=6) for split in ("train", "val")]
    observed = []

    def score(model, data, mean, std, keep_rates):
        observed.append(keep_rates)
        return 1.0

    monkeypatch.setattr(training, "validation_score", score)
    config = TrainConfig(epochs=1, keep_rates=(1.0,), validation_keep_rates=validation_rates)
    fit("lstm", train, val, config=config)
    assert observed == [expected]


@pytest.mark.parametrize("settings", [{"validation_keep_rates": ()}, {"validation_keep_rates": (np.nan,)},
                                      {"train_keep_override": -0.1}, {"train_keep_override": np.nan}])
def test_invalid_retention_settings_are_rejected(settings):
    with pytest.raises(ValueError, match=next(iter(settings))):
        TrainConfig(**settings).validate()


def test_epoch_callback_observes_early_stop_without_mutating_history(tmp_path, monkeypatch):
    cache = make_cache(tmp_path / "cache", n=25)
    train, val = [FeatureCache(cache, split, steps=6) for split in ("train", "val")]
    monkeypatch.setattr(training, "validation_score", lambda *args: 2.0)
    observed = []

    def record(row):
        observed.append(row.copy())
        row.clear()

    config = TrainConfig(epochs=5, patience=1)
    checkpoint = fit("lstm", train, val, config=config, on_epoch=record)
    assert observed == checkpoint["history"]
    assert [row["epoch"] for row in observed] == [1, 2]
    assert all(row["val_macro_mae"] == 2.0 and np.isfinite(row["train_loss"]) for row in observed)


def test_cache_common_targets_order_and_no_cross_session_windows(tmp_path):
    cache = make_cache(tmp_path / "cache")
    data = FeatureCache(cache, "train", task="anticipation", steps=6)
    labels = []
    for f, t, y in data.batches(7):
        assert f.shape[1:] == (6, 8)
        assert t.dtype == torch.float64
        assert (t[:, -1] - t[:, 0] < 0.6).all()
        labels.append(y.numpy())
    np.testing.assert_array_equal(np.concatenate(labels), data.labels())
    assert data.n == 64 - 5 - 20
    assert len(data.sessions()) == len(data.clusters()) == data.n


def test_bad_history_times_are_filtered_and_empty_split_is_explicit(tmp_path):
    cache = make_cache(tmp_path / "cache")
    path = cache / "session_0.meta.npy"
    rows = np.load(path)
    rows[20:23, 0] = [rows[19, 0], np.nan, rows[23, 0] + 2]
    np.save(path, rows)
    data = FeatureCache(cache, "train", steps=6)
    for _, times, _ in data.batches(64):
        gaps = times[:, 1:] - times[:, :-1]
        assert torch.isfinite(times).all() and (gaps > 0).all() and (gaps <= 0.2).all()
    index = json.loads((cache / "index.json").read_text())
    del index["sessions"]["session_2"]
    (cache / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no valid"):
        FeatureCache(cache, "test")


def test_macro_mae_gives_equal_weight_to_nonempty_ground_truth_bins():
    target = np.array([0, 1, 2, 10, 20], dtype=float)
    pred = target + np.array([1, 1, 1, 4, 7])
    assert macro_mae(pred, target) == 4
    assert macro_mae(np.array([2.0]), np.array([1.0])) == 1


@pytest.mark.parametrize("kind, count", [("lstm", 33536), ("cfc", 33024), ("ltc", 33216),
                                         ("transformer", 33472), ("single_frame", 33088)])
def test_all_model_parameter_counts(kind, count):
    counts = SteeringModel(kind, 1536, ode_unfolds=24).blocks()
    assert counts == dict(projection=98368, temporal=count, readout=65,
                          temporal_frozen=8192 if kind == "ltc" else 0)


def test_numpy_settings_still_save_a_loadable_checkpoint(tmp_path):
    cache = make_cache(tmp_path / "cache")
    max_step = np.float64(0.2)
    train = FeatureCache(cache, "train", steps=6, max_step=max_step)
    val = FeatureCache(cache, "val", steps=6, max_step=max_step)
    config = TrainConfig(lr=np.float64(3e-4), seed=np.int64(3), epochs=1, patience=1,
                         batch_size=64, keep_rates=tuple(np.array([1.0, 0.5, 0.25])),
                         validation_keep_rates=(np.float64(0.5),), train_keep_override=np.float64(1.0))
    path = tmp_path / "outputs" / "lstm.pt"
    checkpoint = fit("lstm", train, val, config=config, checkpoint_path=path)
    loaded = Predictor(path)
    f, t, _ = next(FeatureCache(cache, "test", steps=6).batches(64))
    np.testing.assert_array_equal(loaded.predict(f, t), Predictor(checkpoint).predict(f, t))
    assert type(checkpoint["train_config"]["lr"]) is float
    assert type(checkpoint["train_config"]["seed"]) is int
    assert type(checkpoint["train_config"]["validation_keep_rates"][0]) is float
    assert type(checkpoint["train_config"]["train_keep_override"]) is float
    assert type(checkpoint["data_config"]["max_step"]) is float
