import argparse
import json
import os
import sys

import numpy as np

MOVING, STEPS, MAX_STEP, TURN = 3.0, 30, 0.2, 15.0


def refuse_inside_git(path):
    d = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def targets(cache, split, steps=STEPS, max_step=MAX_STEP):
    idx = json.load(open(os.path.join(cache, "index.json")))
    out = []
    for s, meta in sorted(idx["sessions"].items()):
        if meta["split"] != split:
            continue
        m = np.load(os.path.join(cache, f"{s}.meta.npy"), mmap_mode="r")
        t, steer, speed = np.asarray(m[:, 0]), np.asarray(m[:, 1]), np.asarray(m[:, 2])
        ok = np.isfinite(steer) & np.isfinite(speed) & (speed > MOVING)
        tgt = np.flatnonzero(ok)
        tgt = tgt[tgt >= steps - 1]
        if not len(tgt):
            continue
        win = tgt[:, None] - np.arange(steps - 1, -1, -1)[None]
        tgt = tgt[np.diff(t[win], axis=1).max(1) <= max_step]
        if not len(tgt):
            continue
        out.append(dict(session=s, t=t[tgt], y=steer[tgt].astype(np.float32), speed=speed[tgt]))
    return out


def segments(items, pred, seconds, min_targets, turn_share):
    rows, pos = [], 0
    for k, it in enumerate(items):
        n = len(it["t"])
        p = pred[pos:pos + n]
        pos += n
        t0 = it["t"][0]
        edge = np.floor((it["t"] - t0) / seconds).astype(int)
        for e in np.unique(edge):
            sel = np.flatnonzero(edge == e)
            if len(sel) < min_targets:
                continue
            y, q = it["y"][sel], p[sel]
            err = np.abs(q - y)
            rows.append(dict(item=k, sel=sel, mae=float(err.mean()), n=len(sel),
                             turn=float((np.abs(y) >= TURN).mean()) >= turn_share))
    return rows


def describe(it, r, p):
    sel = r["sel"]
    y, q, sp, t = it["y"][sel], p[sel], it["speed"][sel], it["t"][sel]
    a = np.abs(y)
    bins = {"[0,5)": a < 5, "[5,15)": (a >= 5) & (a < 15), "[15,inf)": a >= 15}
    per_bin = {k: (float(np.abs(q - y)[m].mean()) if m.any() else None) for k, m in bins.items()}
    big = a >= TURN
    return dict(targets=int(len(sel)), seconds=round(float(t[-1] - t[0]), 1), mae=round(r["mae"], 3),
                mean_speed_mps=round(float(sp.mean()), 1),
                share_turning=round(float(big.mean()), 3), max_abs_true=round(float(a.max()), 1),
                mae_by_true_bin={k: (None if v is None else round(v, 3)) for k, v in per_bin.items()},
                pred_over_true_when_turning=(round(float(np.abs(q[big]).mean() / a[big].mean()), 3) if big.any() else None),
                same_sign_when_turning=(round(float((np.sign(q[big]) == np.sign(y[big])).mean()), 3) if big.any() else None),
                longest_gap_between_targets_s=round(float(np.diff(t).max()), 2) if len(t) > 1 else 0.0)


def plot(path, it, r, p, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = r["sel"]
    t = it["t"][sel] - it["t"][sel][0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(t, it["y"][sel], label="true", lw=1.6)
    ax.plot(t, p[sel], label="predicted", lw=1.6, ls="--")
    ax.set_xlabel("time within the segment (s)")
    ax.set_ylabel("steering-wheel angle (deg)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pred_for(items, pred, k):
    pos = sum(len(it["t"]) for it in items[:k])
    return pred[pos:pos + len(items[k]["t"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--pred", required=True, help="pred_<model>_s<seed>_test.npz from the finished runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--key", default="k1_m0", help="full-history predictions")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--min-targets", type=int, default=100)
    ap.add_argument("--turn-share", type=float, default=0.2)
    ap.add_argument("--percentile", type=float, default=95.0, help="percentile for the representative failure")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)

    z = np.load(a.pred, allow_pickle=False)
    pred = z[a.key].astype(np.float64)
    items = targets(a.cache, a.split)
    y = np.concatenate([it["y"] for it in items]) if items else np.empty(0, np.float32)
    if len(z["y"]) != len(y) or len(pred) != len(y) or not np.allclose(z["y"], y, atol=1e-4):
        sys.exit("pred file is not aligned with the rebuilt target windows")
    rows = segments(items, pred, a.seconds, a.min_targets, a.turn_share)
    turning = [r for r in rows if r["turn"]]
    good = min(turning, key=lambda r: r["mae"])
    bad = max(rows, key=lambda r: r["mae"])
    maes = np.array([r["mae"] for r in rows])
    target_mae = float(np.percentile(maes, a.percentile))
    rep = min(rows, key=lambda r: abs(r["mae"] - target_mae))

    res = dict(rule=dict(seconds=a.seconds, min_targets=a.min_targets, turn_share=a.turn_share, turn_deg=TURN,
                         key=a.key, percentile=a.percentile), segments=len(rows), segments_with_turn=len(turning),
               segment_mae_median=round(float(np.median(maes)), 3),
               segment_mae_percentile=round(target_mae, 3))
    for name, r, title in (("success", good, "Successful example"),
                           ("failure", bad, "Unsuccessful example (most extreme)"),
                           ("failure_p95", rep, f"Unsuccessful example ({a.percentile:g}th percentile)")):
        it, p = items[r["item"]], pred_for(items, pred, r["item"])
        res[name] = describe(it, r, p)
        plot(os.path.join(a.out, f"{name}.png"), it, r, p, title)
    json.dump(res, open(os.path.join(a.out, "examples.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
