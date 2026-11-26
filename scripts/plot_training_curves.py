import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

KEYS = ("train/loss", "val/mean_macro_mae")
COLOR = "#1f5fa8"


def refuse_inside_git(path):
    d = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(d, ".git")):
            sys.exit(f"refusing to write output inside a git working tree ({d})")
        parent = os.path.dirname(d)
        if parent == d:
            return
        d = parent


def load(csv_path, project, names):
    data = {n: {k: {} for k in KEYS} for n in names}
    ids = defaultdict(set)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["project"] != project or r["run_name"] not in data or r["key"] not in KEYS:
                continue
            ids[r["run_name"]].add(r["run_id"])
            data[r["run_name"]][r["key"]][int(float(r["step"]))] = float(r["value"])
    multi = {n: sorted(v) for n, v in ids.items() if len(v) > 1}
    if multi:
        sys.exit("a run name has more than one run id, stopping so no curves are mixed: "
                 + "; ".join(f"{n}: {len(v)} ids" for n, v in multi.items()))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default="CSC413-D4")
    ap.add_argument("--prefix", default="transformer_lr0.0003_s")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    refuse_inside_git(out)
    os.makedirs(out, exist_ok=True)

    names = [f"{a.prefix}{s}" for s in a.seeds]
    data = load(os.path.expanduser(a.csv), a.project, names)

    missing = [(n, k) for n in names for k in KEYS if not data[n][k]]
    if missing:
        sys.exit("no rows found for: " + ", ".join(f"{n} {k}" for n, k in missing))

    print(f"{'run':>26} {'metric':>20} {'points':>7} {'first':>6} {'last':>6}")
    for n in names:
        for k in KEYS:
            steps = sorted(data[n][k])
            print(f"{n:>26} {k:>20} {len(steps):7d} {steps[0]:6d} {steps[-1]:6d}")
    print("\nvalidation: best mean macro MAE per run (deg) and the step it occurs at")
    for n in names:
        v = data[n]["val/mean_macro_mae"]
        s = min(v, key=v.get)
        print(f"{n:>26} {v[s]:8.3f}  step {s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    titles = {"train/loss": "Training loss (MSE, standardized steering)", "val/mean_macro_mae": "Validation mean macro MAE (deg)"}
    for ax, k in zip(axes, KEYS):
        for n, s in zip(names, a.seeds):
            steps = sorted(data[n][k])
            y = [data[n][k][t] for t in steps]
            if s == a.seeds[0]:
                continue
            ax.plot(steps, y, color=COLOR, alpha=0.25, lw=1.0, label="seeds 1-7" if s == a.seeds[1] else None)
        steps = sorted(data[names[0]][k])
        ax.plot(steps, [data[names[0]][k][t] for t in steps], color=COLOR, lw=2.0, label="seed 0")
        ax.set_xlabel("epoch")
        ax.set_ylabel(titles[k])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=200)
    fig.savefig(os.path.join(out, "training_curves.pdf"))
    print(f"\nfigures written to {out}")


if __name__ == "__main__":
    main()
