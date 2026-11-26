import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import experiment_metrics as metrics
import sparse_run

MODELS = ("lstm", "cfc", "ltc", "transformer")
SPEED_EDGES = np.array([sparse_run.MOVING, 10.0, 20.0, np.inf])
MIN_WINDOWS_FOR_BIN = 100


class Windows(sparse_run.Cached):
    def __init__(self, cache, which, steps, device, kind, max_step=0.2, max_span=10.0):
        idx = json.load(open(os.path.join(cache, "index.json")))
        self.steps, self.device, self.items, counts = steps, device, [], []
        self.n_over_span = 0
        for s, meta in sorted(idx["sessions"].items()):
            if meta["split"] != which:
                continue
            f = np.load(os.path.join(cache, f"{s}.feats.npy"), mmap_mode="r")
            m = np.load(os.path.join(cache, f"{s}.meta.npy"), mmap_mode="r")
            t, steer, speed = m[:, 0], m[:, 1], m[:, 2]
            ok = np.isfinite(steer) & np.isfinite(speed) & (speed > sparse_run.MOVING)
            tgt = np.flatnonzero(ok)
            tgt = tgt[tgt >= steps - 1]
            if not len(tgt):
                continue
            keep, big, span = select_windows(t, tgt, steps, kind, max_step, max_span)
            self.n_over_span += int(big)
            tgt = tgt[keep]
            if not len(tgt):
                continue
            win = tgt[:, None] - np.arange(steps - 1, -1, -1)[None]
            self.items.append(dict(session=s, cluster=meta["cluster"], feats=f, t=t,
                                   y=steer.astype(np.float32), tgt=tgt,
                                   maxstep=np.diff(t[win], axis=1).max(1),
                                   speed=np.asarray(speed[tgt], dtype=np.float64)))
            counts.append(len(tgt))
        self.n = int(sum(counts))

    def column(self, name):
        return np.concatenate([it[name] for it in self.items])


def select_windows(t, tgt, steps, kind, max_step=0.2, max_span=10.0):
    win = tgt[:, None] - np.arange(steps - 1, -1, -1)[None]
    tw = np.asarray(t[win], dtype=np.float64)
    d = np.diff(tw, axis=1)
    span = tw[:, -1] - tw[:, 0]
    if kind == "ordinary":
        return d.max(1) <= max_step, 0, span
    gap = np.isfinite(d).all(1) & (d.min(1) > 0) & (d.max(1) > max_step)
    within = span <= max_span
    return gap & within, int((gap & ~within).sum()), span


def angle_bin_index(angle):
    return np.digitize(np.abs(angle), metrics.BIN_EDGES[1:-1])


def speed_bin_index(speed):
    return np.digitize(speed, SPEED_EDGES[1:-1])


def cell_tables(abs_error, angle, speed, route_group_index, number_of_route_groups):
    number_of_angle_bins = len(metrics.BIN_NAMES)
    number_of_speed_bins = len(SPEED_EDGES) - 1
    shape = (number_of_route_groups, number_of_angle_bins, number_of_speed_bins)
    error_sum = np.zeros(shape)
    count = np.zeros(shape)
    cell = (route_group_index, angle_bin_index(angle), speed_bin_index(speed))
    np.add.at(error_sum, cell, abs_error)
    np.add.at(count, cell, 1)
    return error_sum, count


def macro(error_sum, count):
    count_per_angle_bin = count.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_error_per_angle_bin = np.where(count_per_angle_bin > 0,
                                            error_sum.sum(1) / count_per_angle_bin, np.nan)
    if not np.isfinite(mean_error_per_angle_bin).any():
        return float("nan")
    return float(np.nanmean(mean_error_per_angle_bin))


def reweighted(error_sum_ordinary, count_ordinary, count_gap):
    mean_error_per_angle_bin = []
    for angle_bin in range(error_sum_ordinary.shape[0]):
        weight = count_gap[angle_bin]
        usable = (count_ordinary[angle_bin] > 0) & (weight > 0)
        if not usable.any():
            continue
        mean_error_per_speed_bin = error_sum_ordinary[angle_bin][usable] / count_ordinary[angle_bin][usable]
        mean_error_per_angle_bin.append((weight[usable] * mean_error_per_speed_bin).sum() / weight[usable].sum())
    if not mean_error_per_angle_bin:
        return float("nan")
    return float(np.mean(mean_error_per_angle_bin))


def quantities(error_sum_ordinary, count_ordinary, error_sum_gap, count_gap):
    ordinary = macro(error_sum_ordinary, count_ordinary)
    gap = macro(error_sum_gap, count_gap)
    ordinary_reweighted = reweighted(error_sum_ordinary, count_ordinary, count_gap)
    all_eligible = macro(error_sum_ordinary + error_sum_gap, count_ordinary + count_gap)
    return dict(ordinary=ordinary, gap=gap, ordinary_reweighted=ordinary_reweighted, all_eligible=all_eligible,
                gap_minus_ordinary=gap - ordinary,
                gap_minus_reweighted=gap - ordinary_reweighted,
                all_minus_ordinary=all_eligible - ordinary)


def bootstrap(error_sum_ordinary, count_ordinary, error_sum_gap, count_gap, draws, seed=0):
    point = quantities(error_sum_ordinary.sum(0), count_ordinary.sum(0),
                       error_sum_gap.sum(0), count_gap.sum(0))
    rng = np.random.default_rng(seed)
    number_of_route_groups = error_sum_ordinary.shape[0]
    draw_results = []
    for _ in range(draws):
        picked = rng.integers(0, number_of_route_groups, number_of_route_groups)
        draw_results.append(quantities(error_sum_ordinary[picked].sum(0), count_ordinary[picked].sum(0),
                                       error_sum_gap[picked].sum(0), count_gap[picked].sum(0)))
    out = {}
    for name, estimate in point.items():
        values = np.array([result[name] for result in draw_results])
        values = values[np.isfinite(values)]
        if len(values):
            low, high = np.percentile(values, [2.5, 97.5])
        else:
            low, high = float("nan"), float("nan")
        out[name] = dict(estimate=estimate, ci95=[float(low), float(high)], valid_draws=int(len(values)))
    return out


def load_model(kind, results_dir, lr, seed, dim, device):
    import torch
    name = f"{kind}_lr{lr:g}_s{seed}{'_u24' if kind == 'ltc' else ''}.pt"
    path = os.path.join(results_dir, "ckpt", name)
    state = torch.load(path, map_location="cpu", weights_only=False)["best_state"]
    model = sparse_run.SteeringModel(kind, dim, seed=seed, ode_unfolds=24 if kind == "ltc" else 6).to(device)
    model.load_state_dict(state)
    return model.eval(), name


def chosen_lr(kind, results_dir, override):
    if kind in override:
        return override[kind]
    return json.load(open(os.path.join(results_dir, f"{kind}_lr_chosen.json")))["chosen_lr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--results-dir", required=True, help="holds ckpt/, <model>_lr_chosen.json, pred_*_val.npz")
    ap.add_argument("--out", required=True, help="outside any git working tree")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    ap.add_argument("--lr", nargs="*", default=[], metavar="MODEL=RATE",
                    help="override a model's learning rate, e.g. lstm=1e-4 for the independent reproduction")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min-step", type=float, default=0.2)
    ap.add_argument("--max-span", type=float, default=10.0)
    ap.add_argument("--gap-bins", type=float, nargs="+", default=[0.5, 1.0],
                    help="upper edges (s) of the largest-step bins above --min-step; the last bin is open")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    sparse_run.refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    override = {k: float(v) for k, v in (x.split("=") for x in a.lr)}
    t0 = time.perf_counter()

    train = sparse_run.Cached(a.cache, "train", a.steps, device, a.min_step)
    ytr = train.labels()
    mu, sd = float(ytr.mean()), float(ytr.std())
    dim = train.items[0]["feats"].shape[1]
    ordn = Windows(a.cache, "val", a.steps, device, "ordinary", a.min_step, a.max_span)
    gap = Windows(a.cache, "val", a.steps, device, "gap", a.min_step, a.max_span)
    y_o, y_g = ordn.labels(), gap.labels()
    sp_o, sp_g = ordn.column("speed"), gap.column("speed")
    ms_g = gap.column("maxstep")
    clusters = np.unique(np.concatenate([ordn.clusters(), gap.clusters()]).astype(str))
    ci_o = np.searchsorted(clusters, ordn.clusters().astype(str))
    ci_g = np.searchsorted(clusters, gap.clusters().astype(str))
    edges = [a.min_step] + list(a.gap_bins) + [np.inf]
    gb = np.digitize(ms_g, edges[1:-1])
    gb_names = [f"({edges[i]:g}, {edges[i + 1]:g}] s" for i in range(len(edges) - 1)]
    print(f"ordinary {len(y_o):,} windows, gap {len(y_g):,} (cut by the {a.max_span:g} s span cap: "
          f"{gap.n_over_span:,}), {len(clusters)} route groups, device {device}")
    if len(y_g) == 0:
        sys.exit("no gap windows in validation")

    result = dict(
        status="exploratory; validation only; run after the test split was sealed; plan docs/natural_gap_plan.md",
        rule=dict(min_step_s=a.min_step, max_span_s=a.max_span, steps=a.steps, retention=1.0),
        windows=dict(ordinary=len(y_o), gap=len(y_g), gap_cut_by_span_cap=gap.n_over_span,
                     route_groups=len(clusters),
                     gap_route_groups=int(len(np.unique(ci_g))),
                     gap_by_largest_step={n: int((gb == i).sum()) for i, n in enumerate(gb_names)}),
        models={})

    for kind in a.models:
        lr = chosen_lr(kind, a.results_dir, override)
        errs_o, errs_g, per_seed, files = [], [], [], []
        for seed in a.seeds:
            model, ck = load_model(kind, a.results_dir, lr, seed, dim, device)
            npz = np.load(os.path.join(a.results_dir, f"pred_{kind}_s{seed}_val.npz"))
            if not np.allclose(npz["y"], y_o, atol=1e-4):
                sys.exit(f"{kind} s{seed}: pred file is not aligned with the rebuilt ordinary windows")
            if seed == a.seeds[0]:
                one = sparse_run.Cached.__new__(sparse_run.Cached)
                one.__dict__.update(ordn.__dict__)
                one.items = ordn.items[:1]
                n1 = len(one.items[0]["tgt"])
                p1 = sparse_run.predict(model, one, mu, sd, 1.0, 7)
                diff = float(np.abs(p1 - npz["k1_m0"][:n1]).max())
                print(f"[{kind}] self-check on one session ({n1:,} windows): max |diff| {diff:.2e} deg")
                if diff > 0.05:
                    sys.exit(f"{kind}: restored checkpoint does not reproduce the stored predictions "
                             f"(max diff {diff:.3f} deg); check the learning rate and unfolds")
                result.setdefault("self_check_max_abs_diff_deg", {})[kind] = diff
            p_o = npz["k1_m0"].astype(np.float64)
            p_g = sparse_run.predict(model, gap, mu, sd, 1.0, 7).astype(np.float64)
            e_o, e_g = np.abs(p_o - y_o), np.abs(p_g - y_g)
            errs_o.append(e_o)
            errs_g.append(e_g)
            files.append(ck)
            row = dict(seed=seed, gap_macro_mae=metrics.macro_mae(p_g, y_g),
                       ordinary_macro_mae=metrics.macro_mae(p_o, y_o))
            row["by_largest_step"] = {
                n: (metrics.macro_mae(p_g[gb == i], y_g[gb == i]) if (gb == i).sum() >= MIN_WINDOWS_FOR_BIN else None)
                for i, n in enumerate(gb_names)}
            per_seed.append(row)
            print(f"[{kind}] seed {seed}: gap {row['gap_macro_mae']:.3f}  ordinary {row['ordinary_macro_mae']:.3f}")
            del model
        eo, eg = np.mean(errs_o, 0), np.mean(errs_g, 0)
        error_sum_ordinary, count_ordinary = cell_tables(eo, y_o, sp_o, ci_o, len(clusters))
        error_sum_gap, count_gap = cell_tables(eg, y_g, sp_g, ci_g, len(clusters))
        gm = np.array([r["gap_macro_mae"] for r in per_seed])
        om = np.array([r["ordinary_macro_mae"] for r in per_seed])
        assert abs(gm.mean() - macro(error_sum_gap.sum(0), count_gap.sum(0))) < 1e-6, "seed-mean identity failed"
        bins = {}
        for i, n in enumerate(gb_names):
            m = gb == i
            vals = [r["by_largest_step"][n] for r in per_seed]
            bins[n] = dict(windows=int(m.sum()), route_groups=int(len(np.unique(ci_g[m]))),
                           macro_mae_mean=(float(np.mean(vals)) if None not in vals else None),
                           macro_mae_sd_over_seeds=(float(np.std(vals, ddof=1)) if None not in vals and len(vals) > 1 else None))
        result["models"][kind] = dict(
            lr=lr, checkpoints=files, seeds=a.seeds,
            gap_macro_mae=dict(mean=float(gm.mean()), sd_over_seeds=float(gm.std(ddof=1)) if len(gm) > 1 else None),
            ordinary_macro_mae=dict(mean=float(om.mean()), sd_over_seeds=float(om.std(ddof=1)) if len(om) > 1 else None),
            comparisons=bootstrap(error_sum_ordinary, count_ordinary, error_sum_gap, count_gap, a.boot), by_largest_step=bins, per_seed=per_seed)
        c = result["models"][kind]["comparisons"]
        print(f"[{kind}] all-eligible - ordinary {c['all_minus_ordinary']['estimate']:+.3f} "
              f"{c['all_minus_ordinary']['ci95']}; gap - reweighted {c['gap_minus_reweighted']['estimate']:+.3f} "
              f"{c['gap_minus_reweighted']['ci95']}")

    path = os.path.join(a.out, "natural_gap_val.json")
    json.dump(result, open(path, "w"), indent=1)
    os.chmod(path, 0o600)
    print(f"wrote {path} in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
