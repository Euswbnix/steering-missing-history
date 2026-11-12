"""The analysis fixed by the anticipation plan.

Every statistic is a function of mean macro MAEs, resampled as in the sparse-observation experiment: route clusters jointly (the
frames are shared) and each model's seeds independently, with a model's horizons and conditions
kept together because one trained model produces all of them.

  Main question (no correction): does history help more for anticipation than for the current
      frame?  [single_frame - LSTM at 1 s] - [single_frame - LSTM at 0 s], full history. Interval
      above 0 -> yes.
  Model comparisons: CfC, LTC and Transformer minus LSTM at 1 s, with full history and with 75%
      dropped; three-way verdict against delta = 0.5.
  Effect of dropping frames: does dropping history hurt anticipation more?  For each sequence
      model, [75% dropped - full at 1 s] - [75% dropped - full at 0 s].
  The model comparisons and the frame-dropping effects form the secondary family, Holm-corrected
  together (difference and equivalence claims each, as in sparse_analyze).
  Descriptive: macro MAE by model, horizon and condition, and three untrained baselines --
      the best constant, the model's own 0 s output used as the forecast, and persistence of the
      true current angle (it uses CAN, so it is a difficulty reference, not a vision model).

    python experiments/anticipation_analyze.py --pred results/anticipation --split val --out results/anticipation/analysis/val.json
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment_metrics as metrics
from sparse_analyze import ALPHA, DELTA, holm, p_values, verdict

HORIZONS = (0.0, 0.5, 1.0, 2.0)
NOW, MAIN = 0, 2                                   # columns: the current frame, and 1 s ahead
MODEL_NAMES = ("single_frame", "lstm", "cfc", "ltc", "transformer")
SEQUENCE = ("lstm", "cfc", "ltc", "transformer")


def load(pred_dir, model_name, split):
    out, ref = {}, None
    for f in sorted(glob.glob(os.path.join(pred_dir, f"pred_{model_name}_s*_{split}.npz"))):
        seed = int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1))
        z = np.load(f, allow_pickle=False)
        cur = (z["y"].astype(np.float64), z["cluster"], z["session"])
        if ref is None:
            ref = cur
        elif not all(np.array_equal(p, q) for p, q in zip(ref, cur)):
            sys.exit(f"{f}: different target frames from the other runs; refusing to pair")
        out[seed] = {k: z[k].astype(np.float64) for k in z.files if re.fullmatch(r"k[0-9.]+_m\d+", k)}
    return out, ref


def matrix(runs, y, groups, keep, col):
    """(seeds, groups): each group's share of the macro MAE at one horizon, mean over mask draws."""
    w = metrics.macro_weights(y[:, col])
    keys = sorted(k for k in next(iter(runs.values())) if k.startswith(f"k{keep:g}_m"))
    ids = np.unique(groups)
    rows = []
    for s in sorted(runs):
        err = np.mean([w * np.abs(runs[s][k][:, col] - y[:, col]) for k in keys], axis=0)
        rows.append([err[groups == g].sum() for g in ids])
    return np.array(rows)


def resample(mats, fn, n_boot, rng):
    """mats: {model_name: [matrix, ...]}; fn maps {model_name: [mean macro MAE, ...]} to the statistic."""
    estimate = fn({a: [m.sum(1).mean() for m in ms] for a, ms in mats.items()})
    g = next(iter(mats.values()))[0].shape[1]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        cg = rng.integers(g, size=g)
        vals = {}
        for a, ms in mats.items():
            si = rng.integers(len(ms[0]), size=len(ms[0]))          # one seed draw for all of an model's cells
            vals[a] = [m[si][:, cg].sum(1).mean() for m in ms]
        draws[i] = fn(vals)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(estimate), [float(lo), float(hi)], draws


def analyze(pred, split, n_boot=10000, delta=DELTA, require_all=False):
    runs, ref = {}, None
    for model_name in MODEL_NAMES:
        r, cur = load(pred, model_name, split)
        if not r:
            continue
        if ref is not None and not all(np.array_equal(p, q) for p, q in zip(ref, cur)):
            sys.exit(f"{model_name} was scored on different target frames")
        runs[model_name], ref = r, cur
    missing = [a for a in MODEL_NAMES if a not in runs]
    if missing and require_all:
        sys.exit(f"no predictions for {missing} on {split}")
    if "lstm" not in runs:
        sys.exit("the LSTM is the reference for every question; it has no predictions yet")
    y, cluster, session = ref
    date = np.array([s[:8] for s in session])
    keeps = sorted({float(k.split("_")[0][1:]) for k in next(iter(runs["lstm"].values()))}, reverse=True)
    rng = np.random.default_rng(0)
    res = dict(split=split, target_frames=int(len(y)), clusters=int(len(np.unique(cluster))), delta=delta,
               horizons=list(HORIZONS), model_names={a: sorted(r) for a, r in runs.items()}, missing=missing)
    print(f"{split}: {len(y):,} target frames, {res['clusters']} route clusters; models "
          + ", ".join(f"{a} ({len(r)} seeds)" for a, r in runs.items())
          + (f"; missing {missing}" if missing else ""))

    def M(model_name, keep, col, groups=cluster):
        return matrix(runs[model_name], y, groups, keep, col)

    # ---- descriptive table and baselines
    table = {}
    for model_name in runs:
        table[model_name] = {}
        for keep in keeps:
            per_seed = np.array([M(model_name, keep, c).sum(1) for c in range(len(HORIZONS))])      # (H, seeds)
            table[model_name][f"keep_{keep:g}"] = dict(mean=per_seed.mean(1).tolist(), seed_sd=per_seed.std(1, ddof=1).tolist())
        as_forecast = np.array([[metrics.macro_mae(runs[model_name][s]["k1_m0"][:, NOW], y[:, c]) for s in sorted(runs[model_name])]
                            for c in range(len(HORIZONS))])
        table[model_name]["current_as_forecast"] = as_forecast.mean(1).tolist()
    base = dict(constant=[metrics.macro_mae(np.full(len(y), metrics.best_constant(y[:, c])), y[:, c])
                          for c in range(len(HORIZONS))],
                true_persistence=[metrics.macro_mae(y[:, NOW], y[:, c]) for c in range(len(HORIZONS))])
    res["table"], res["baselines"] = table, base
    print("\nDescriptive: macro MAE (deg) at " + " / ".join(f"{h:g} s" for h in HORIZONS))
    for model_name in runs:
        for keep in keeps:
            print(f"  {model_name:12s} keep {keep:4g}: " + "  ".join(f"{v:7.3f}" for v in table[model_name][f"keep_{keep:g}"]["mean"]))
        print(f"  {model_name:12s} 0 s output as forecast: " + "  ".join(f"{v:7.3f}" for v in table[model_name]["current_as_forecast"]))
    print("  best constant:            " + "  ".join(f"{v:7.3f}" for v in base["constant"]))
    print("  true-angle persistence:   " + "  ".join(f"{v:7.3f}" for v in base["true_persistence"]))

    # ---- main question: does history help anticipation more than the current frame?
    if "single_frame" in runs:
        main = {}
        for name, groups in (("route_cluster", cluster), ("session", session), ("date", date)):
            est, ci, _ = resample(
                {"single_frame": [M("single_frame", 1.0, MAIN, groups), M("single_frame", 1.0, NOW, groups)],
                 "lstm": [M("lstm", 1.0, MAIN, groups), M("lstm", 1.0, NOW, groups)]},
                lambda v: (v["single_frame"][0] - v["lstm"][0]) - (v["single_frame"][1] - v["lstm"][1]), n_boot, rng)
            main[name] = dict(estimate=est, ci=ci,
                            verdict="yes" if ci[0] > 0 else "no, less" if ci[1] < 0 else "not distinguished")
        gain = {f"{h:g}s": float(M("single_frame", 1.0, c).sum(1).mean() - M("lstm", 1.0, c).sum(1).mean())
                for c, h in enumerate(HORIZONS)}
        res["main_question"] = dict(main, history_gain_by_horizon=gain)
        print(f"\nMain question: history gain (single_frame - LSTM, full history) by horizon: "
              + ", ".join(f"{k} {v:+.3f}" for k, v in gain.items()))
        print(f"    gain at 1 s minus gain at 0 s: {main['route_cluster']['estimate']:+.3f} "
              f"[{main['route_cluster']['ci'][0]:+.3f}, {main['route_cluster']['ci'][1]:+.3f}] -> {main['route_cluster']['verdict']}"
              f"   (by session {main['session']['verdict']}, by date {main['date']['verdict']})")

    # ---- the secondary family: model comparisons and frame-dropping effects, Holm-corrected together
    tests = []
    for model_name in ("cfc", "ltc", "transformer"):
        if model_name not in runs:
            continue
        for keep in (1.0, 0.25):
            est, ci, d = resample({model_name: [M(model_name, keep, MAIN)], "lstm": [M("lstm", keep, MAIN)]},
                                  lambda v, a=model_name: v[a][0] - v["lstm"][0], n_boot, rng)
            p_diff, p_equiv = p_values(d, delta)
            sens = {}
            for name, groups in (("session", session), ("date", date)):
                _, ci2, _ = resample({model_name: [M(model_name, keep, MAIN, groups)], "lstm": [M("lstm", keep, MAIN, groups)]},
                                     lambda v, a=model_name: v[a][0] - v["lstm"][0], n_boot, rng)
                sens[name] = dict(ci=ci2, verdict=verdict(ci2[0], ci2[1], delta))
            tests.append(dict(test=f"{model_name}-lstm 1s keep_{keep:g}", estimate=est, ci=ci, p_diff=p_diff,
                              p_equiv=p_equiv, nominal=verdict(ci[0], ci[1], delta), sensitivity=sens))
    for model_name in SEQUENCE:
        if model_name not in runs:
            continue
        est, ci, d = resample(
            {model_name: [M(model_name, 0.25, MAIN), M(model_name, 1.0, MAIN), M(model_name, 0.25, NOW), M(model_name, 1.0, NOW)]},
            lambda v, a=model_name: (v[a][0] - v[a][1]) - (v[a][2] - v[a][3]), n_boot, rng)
        tests.append(dict(test=f"{model_name} extra degradation at 1s", estimate=est, ci=ci,
                          p_diff=p_values(d, None)[0], p_equiv=None,
                          nominal="different" if ci[0] > 0 or ci[1] < 0 else "not distinguished"))
    if tests and 2 * len(tests) / (n_boot + 1) > ALPHA / 5:
        sys.exit(f"{n_boot} bootstrap draws cannot resolve Holm over {len(tests)} tests; use more")
    adj = holm([t["p_diff"] for t in tests]) if tests else []
    eq = [i for i, t in enumerate(tests) if t["p_equiv"] is not None]
    adj_eq = dict(zip(eq, holm([tests[i]["p_equiv"] for i in eq]))) if eq else {}
    print(f"\nSecondary family, Holm over m = {len(tests)}")
    for i, t in enumerate(tests):
        t["p_diff_holm"] = float(adj[i])
        t["p_equiv_holm"] = float(adj_eq[i]) if i in adj_eq else None
        t["verdict_holm"] = ("practically equivalent" if i in adj_eq and adj_eq[i] <= ALPHA else
                             "different" if adj[i] <= ALPHA else "not distinguished")
        print(f"  {t['test']:36s} {t['estimate']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]  "
              f"Holm p {t['p_diff_holm']:.4f}  {t['verdict_holm']}")
    res["secondary"], res["secondary_m"] = tests, len(tests)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--require-all", action="store_true", help="fail unless all five models have predictions")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = analyze(a.pred, a.split, a.boot, a.delta, a.require_all)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        os.chmod(a.out, 0o600)


if __name__ == "__main__":
    main()
