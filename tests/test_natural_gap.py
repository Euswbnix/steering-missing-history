"""Natural-gap evaluation on synthetic data: the selected windows really contain a break, they never
overlap the formal evaluation's windows, the cell statistics reproduce the registered macro MAE, and
the whole script runs end to end on small made-up checkpoints."""
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import experiment_metrics as metrics  # noqa: E402
import natural_gap_eval as ng  # noqa: E402
import sparse_run  # noqa: E402

DIM, N, STEPS = 8, 320, 6


def make_cache(root, with_bad_clock=True):
    """Three splits, two sessions each. Breaks of 0.5 s, 2 s and 15 s are cut into the clock."""
    root.mkdir()
    rng = np.random.default_rng(3)
    sessions = {}
    for i, split in enumerate(("train", "val", "test")):
        for j in range(2):
            name = f"s{i}{j}"
            t = 0.1 * np.arange(N)
            t[60:] += 0.4          # one step of 0.5 s
            t[150:] += 1.9         # one step of 2.0 s
            t[250:] += 14.9        # one step of 15 s
            if with_bad_clock and j == 1:
                t[200] = t[199]    # a repeated timestamp: a clock fault, not a gap
            t = t + 1000 * (3 * i + j)
            ang = 40 * np.sin(0.05 * np.arange(N)) + rng.normal(0, 2, N)
            speed = 4 + 8 * (np.arange(N) % 4)
            np.save(root / f"{name}.feats.npy", rng.normal(size=(N, DIM)).astype(np.float16))
            np.save(root / f"{name}.meta.npy", np.column_stack([t, ang, speed]))
            sessions[name] = dict(split=split, cluster=2 * i + j)
    (root / "index.json").write_text(json.dumps(dict(base_hz=10, sessions=sessions)))
    return str(root)


def test_gap_windows_have_a_break_and_never_overlap_the_formal_windows(tmp_path):
    cache = make_cache(tmp_path / "c")
    formal = sparse_run.Cached(cache, "val", STEPS, "cpu")
    ordinary = ng.Windows(cache, "val", STEPS, "cpu", "ordinary")
    gap = ng.Windows(cache, "val", STEPS, "cpu", "gap", max_span=10.0)
    assert ordinary.n == formal.n                                     # ordinary == Cached, window for window
    assert np.array_equal(ordinary.labels(), formal.labels())
    assert gap.n > 0 and (gap.column("maxstep") > 0.2).all()
    by_session = {it["session"]: set(it["tgt"]) for it in ordinary.items}
    for it in gap.items:
        assert not (set(it["tgt"]) & by_session.get(it["session"], set()))


def test_span_cap_drops_the_long_break_and_counts_it(tmp_path):
    cache = make_cache(tmp_path / "c")
    capped = ng.Windows(cache, "val", STEPS, "cpu", "gap", max_span=10.0)
    free = ng.Windows(cache, "val", STEPS, "cpu", "gap", max_span=1e9)
    assert free.n > capped.n and capped.n_over_span == free.n - capped.n > 0
    assert free.column("maxstep").max() > 10 > capped.column("maxstep").max()


def test_repeated_timestamp_is_not_a_gap(tmp_path):
    cache = make_cache(tmp_path / "c")
    t = np.load(os.path.join(cache, "s11.meta.npy"))[:, 0]
    tgt = np.arange(STEPS - 1, N)
    keep, _, _ = ng.select_windows(t, tgt, STEPS, "gap", 0.2, 1e9)
    win = tgt[keep][:, None] - np.arange(STEPS - 1, -1, -1)[None]
    assert (np.diff(t[win], axis=1).min(1) > 0).all()
    bad = [k for k in range(N) if k >= STEPS - 1 and (np.diff(t[k - STEPS + 1:k + 1]) <= 0).any()]
    assert bad and not set(bad) & set(tgt[keep])


def test_cell_tables_reproduce_macro_mae():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(0, 3, 400), rng.normal(0, 30, 300)])
    p = y + rng.normal(0, 4, len(y))
    speed = rng.uniform(3.1, 30, len(y))
    cl = rng.integers(0, 4, len(y))
    S, N_ = ng.cell_tables(np.abs(p - y), y, speed, cl, 4)
    assert ng.macro(S.sum(0), N_.sum(0)) == pytest.approx(metrics.macro_mae(p, y))
    assert N_.sum() == len(y)


def test_reweighting_moves_ordinary_to_the_gap_speed_mix():
    # one angle bin; ordinary error 2 at low speed and 10 at high speed; gap windows are all high speed
    So = np.zeros((3, 3)); No = np.zeros((3, 3))
    So[0, 0], No[0, 0] = 2 * 100, 100
    So[0, 2], No[0, 2] = 10 * 100, 100
    Ng = np.zeros((3, 3)); Ng[0, 2] = 5
    assert ng.reweighted(So, No, Ng) == pytest.approx(10.0)
    Ng[0, 0] = 5
    assert ng.reweighted(So, No, Ng) == pytest.approx(6.0)


def test_bootstrap_point_estimate_matches_the_pooled_numbers():
    rng = np.random.default_rng(1)
    So, No = rng.uniform(5, 50, (5, 3, 3)), rng.integers(10, 50, (5, 3, 3)).astype(float)
    Sg, Ng = rng.uniform(1, 20, (5, 3, 3)), rng.integers(0, 5, (5, 3, 3)).astype(float)
    out = ng.bootstrap(So, No, Sg, Ng, draws=50)
    direct = ng.quantities(So.sum(0), No.sum(0), Sg.sum(0), Ng.sum(0))
    for k, v in direct.items():
        assert out[k]["estimate"] == pytest.approx(v)
        assert out[k]["ci95"][0] <= out[k]["ci95"][1]


@pytest.mark.parametrize("kinds", [("lstm", "ltc")])
def test_script_end_to_end_on_small_checkpoints(tmp_path, monkeypatch, kinds):
    cache = make_cache(tmp_path / "c")
    res = tmp_path / "res"
    (res / "ckpt").mkdir(parents=True)
    # the script reads the window length from --steps; use the same short window everywhere
    train = sparse_run.Cached(cache, "train", STEPS, "cpu")
    val = sparse_run.Cached(cache, "val", STEPS, "cpu")
    ytr = train.labels(); mu, sd = float(ytr.mean()), float(ytr.std())
    for kind in kinds:
        json.dump(dict(chosen_lr=1e-4), open(res / f"{kind}_lr_chosen.json", "w"))
        for seed in (0, 1):
            torch.manual_seed(seed)
            m = sparse_run.SteeringModel(kind, DIM, seed=seed, ode_unfolds=24 if kind == "ltc" else 6).eval()
            name = f"{kind}_lr0.0001_s{seed}{'_u24' if kind == 'ltc' else ''}.pt"
            torch.save(dict(best_state={k: v.clone() for k, v in m.state_dict().items()}), res / "ckpt" / name)
            p = sparse_run.predict(m, val, mu, sd, 1.0, 7)
            np.savez(res / f"pred_{kind}_s{seed}_val.npz", y=val.labels().astype(np.float32),
                     cluster=val.clusters(), session=val.sessions(), k1_m0=p)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--cache", cache, "--results-dir", str(res), "--out", str(out),
                                      "--models", *kinds, "--seeds", "0", "1", "--steps", str(STEPS),
                                      "--boot", "30"])
    ng.main()
    r = json.load(open(out / "natural_gap_val.json"))
    assert r["windows"]["gap"] > 0 and r["windows"]["gap_cut_by_span_cap"] > 0
    for kind in kinds:
        m = r["models"][kind]
        assert m["comparisons"]["gap"]["valid_draws"] > 0
        assert np.isfinite(m["gap_macro_mae"]["mean"]) and len(m["per_seed"]) == 2
        assert r["self_check_max_abs_diff_deg"][kind] < 0.05
    text = (out / "natural_gap_val.json").read_text()
    assert "s10" not in text and "s11" not in text                    # no session names in the output


def test_refuses_to_write_inside_a_git_tree(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(sys, "argv", ["x", "--cache", "c", "--results-dir", "r", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        ng.main()
