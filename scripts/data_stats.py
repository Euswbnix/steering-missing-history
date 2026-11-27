"""Summary statistics and distribution figures of the data the models actually see.

Reads only the feature cache's index.json and the per-session metadata (<session>.meta.npy:
time, steering angle, speed). It reads no features, no video and no predictions. Rules are the
same as experiments/sparse_run.py (Cached):

  * a frame is usable if its angle and speed are finite and speed > 3 m/s
  * a target window is 30 frames ending at a usable frame whose largest step is at most 0.2 s
  * a natural-gap window has largest step > 0.2 s, every step positive and finite, span <= 10 s
    (the rule of experiments/natural_gap_eval.py)

Per split it reports sessions, route groups (and how many also occur in train), recorded hours
(first to last timestamp, stationary time included), frames, the share dropped as stationary,
target windows, natural-gap windows, the share of targets in each |angle| bin used by macro MAE
([0,5), [5,15), [15,inf) deg), and quantiles of |angle| and speed over the targets.

In the figures, |angle| below 0.1 deg is counted at 0.1 and speed above 40 m/s at 40.

Only aggregates are written; no session id, timestamp or per-frame value leaves this script.
Output must be outside any git working tree.

    python scripts/data_stats.py --cache /srv/csc413/hdd/cache/dinov2_10hz --out ~/workspace/data-stats
"""
import argparse
import json
import os
import sys

import numpy as np

MOVING, STEPS, MAX_STEP, MAX_SPAN = 3.0, 30, 0.2, 10.0
BIN_EDGES = (5.0, 15.0)
BIN_NAMES = ("[0,5)", "[5,15)", "[15,inf)")
SPLITS = ("train", "val", "test")
COLORS = {"train": "#1f5fa8", "val": "#d9822b", "test": "#3a9a5b"}


def refuse_inside_git(path):
    d = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write HDD-derived output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def session_windows(t, steer, speed, steps=STEPS, max_step=MAX_STEP, max_span=MAX_SPAN):
    """(usable mask, ordinary target idx, natural-gap target idx)."""
    ok = np.isfinite(steer) & np.isfinite(speed) & (speed > MOVING)
    tgt = np.flatnonzero(ok)
    tgt = tgt[tgt >= steps - 1]
    if not len(tgt):
        return ok, tgt, tgt
    win = tgt[:, None] - np.arange(steps - 1, -1, -1)[None]
    tw = np.asarray(t[win], dtype=np.float64)
    d = np.diff(tw, axis=1)
    ordinary = d.max(1) <= max_step
    gap = np.isfinite(d).all(1) & (d.min(1) > 0) & (d.max(1) > max_step) & (tw[:, -1] - tw[:, 0] <= max_span)
    return ok, tgt[ordinary], tgt[gap]


def collect(cache):
    idx = json.load(open(os.path.join(cache, "index.json")))
    acc = {s: dict(sessions=0, groups=set(), seconds=0.0, frames=0, finite=0, usable=0,
                   targets=0, gap=0, abs_y=[], speed=[]) for s in SPLITS}
    for name, meta in sorted(idx["sessions"].items()):
        a = acc.get(meta["split"])
        if a is None:
            continue
        m = np.load(os.path.join(cache, f"{name}.meta.npy"), mmap_mode="r")
        t, steer, speed = np.asarray(m[:, 0]), np.asarray(m[:, 1]), np.asarray(m[:, 2])
        ok, tgt, gap = session_windows(t, steer, speed)
        a["sessions"] += 1
        a["groups"].add(str(meta["cluster"]))
        if len(t) > 1:
            a["seconds"] += float(t[-1] - t[0])
        a["frames"] += len(t)
        a["finite"] += int((np.isfinite(steer) & np.isfinite(speed)).sum())
        a["usable"] += int(ok.sum())
        a["targets"] += len(tgt)
        a["gap"] += len(gap)
        a["abs_y"].append(np.abs(steer[tgt]).astype(np.float32))
        a["speed"].append(speed[tgt].astype(np.float32))
    return acc


def summarise(acc):
    out = {}
    for s, a in acc.items():
        y = np.concatenate(a["abs_y"]) if a["abs_y"] else np.zeros(0)
        v = np.concatenate(a["speed"]) if a["speed"] else np.zeros(0)
        b = np.digitize(y, BIN_EDGES)
        q = [5, 25, 50, 75, 95, 99]
        out[s] = dict(
            sessions=a["sessions"], route_groups=len(a["groups"]),
            route_groups_also_in_train=(len(a["groups"] & acc["train"]["groups"]) if s != "train" else None),
            recorded_hours_incl_stationary=round(a["seconds"] / 3600, 2),
            frames=a["frames"],
            share_finite_dropped_as_stationary=(round(1 - a["usable"] / a["finite"], 4) if a["finite"] else None),
            target_windows=a["targets"], natural_gap_windows=a["gap"],
            natural_gap_share_of_eligible=(round(a["gap"] / (a["targets"] + a["gap"]), 4) if a["targets"] + a["gap"] else None),
            target_share_by_abs_angle_bin={n: (round(float((b == i).mean()), 4) if len(y) else None)
                                           for i, n in enumerate(BIN_NAMES)},
            abs_angle_quantiles_deg={f"p{p}": round(float(np.percentile(y, p)), 2) for p in q} if len(y) else None,
            speed_quantiles_mps={f"p{p}": round(float(np.percentile(v, p)), 2) for p in q} if len(v) else None)
    return out


def plot(acc, summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    # (a) share of targets per |angle| bin, by split
    ax = axes[0]
    w = 0.26
    x = np.arange(len(BIN_NAMES))
    for k, s in enumerate(SPLITS):
        sh = [summary[s]["target_share_by_abs_angle_bin"][n] or 0 for n in BIN_NAMES]
        ax.bar(x + (k - 1) * w, sh, w, color=COLORS[s], label=s)
    ax.set_xticks(x, [f"{n} deg" for n in BIN_NAMES])
    ax.set_ylabel("share of target frames")
    ax.set_title("|steering angle| bins used by macro MAE", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    # (b) |angle| density on a log x axis, edges of the bins marked
    ax = axes[1]
    edges = np.logspace(-1, np.log10(600), 60)
    for s in SPLITS:
        y = np.concatenate(acc[s]["abs_y"]) if acc[s]["abs_y"] else np.zeros(0)
        if len(y):
            ax.hist(np.clip(y, 0.1, None), bins=edges, weights=np.full(len(y), 1 / len(y)),
                    histtype="step", lw=1.5, color=COLORS[s], label=s)
    for e in BIN_EDGES:
        ax.axvline(e, color="0.5", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("|steering-wheel angle| (deg, log scale)")
    ax.set_ylabel("share of target frames per bar")
    ax.set_title("|steering angle| of target frames", fontsize=10)

    # (c) speed density of target frames
    ax = axes[2]
    edges = np.linspace(MOVING, 40, 60)
    for s in SPLITS:
        v = np.concatenate(acc[s]["speed"]) if acc[s]["speed"] else np.zeros(0)
        if len(v):
            ax.hist(np.clip(v, None, 40), bins=edges, weights=np.full(len(v), 1 / len(v)),
                    histtype="step", lw=1.5, color=COLORS[s], label=s)
    ax.set_xlabel("speed (m/s; > 3 m/s only)")
    ax.set_ylabel("share of target frames per bar")
    ax.set_title("speed of target frames", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "data_distributions.png"), dpi=200)
    fig.savefig(os.path.join(out, "data_distributions.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refuse_inside_git(a.out)
    os.makedirs(a.out, exist_ok=True)
    os.chmod(a.out, 0o700)

    acc = collect(a.cache)
    summary = summarise(acc)
    path = os.path.join(a.out, "data_stats.json")
    json.dump(summary, open(path, "w"), indent=1)
    plot(acc, summary, a.out)

    hdr = f"{'split':6} {'sess':>5} {'groups':>6} {'rec_h':>7} {'targets':>9} {'gap':>6} {'stationary':>10}  " + \
          "  ".join(f"{n:>9}" for n in BIN_NAMES)
    print(hdr)
    for s in SPLITS:
        r = summary[s]
        sh = r["target_share_by_abs_angle_bin"]
        print(f"{s:6} {r['sessions']:>5} {r['route_groups']:>6} {r['recorded_hours_incl_stationary']:>7.2f} {r['target_windows']:>9,} "
              f"{r['natural_gap_windows']:>6,} {r['share_finite_dropped_as_stationary'] or 0:>10.3f}  "
              + "  ".join(f"{(sh[n] or 0):>9.3f}" for n in BIN_NAMES))
    print("\nrec_h = first-to-last timestamp per session, stationary time included (split.json's hours are moving hours)")
    print("route groups are a clustering unit for statistics, not the split unit: the split holds out whole sessions,")
    print("so a group can appear in more than one split. Groups also in train: "
          + ", ".join(f"{s} {summary[s]['route_groups_also_in_train']}" for s in SPLITS[1:]))
    print(f"\nwrote {path} and data_distributions.png/.pdf to {a.out}")


if __name__ == "__main__":
    main()
