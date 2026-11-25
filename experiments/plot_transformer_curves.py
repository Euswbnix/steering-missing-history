"""Plot the final Transformer runs from existing scalar exports; no model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

TASKS = {
    "sparse": ("CSC413-D4", 3e-4, "Current-angle prediction", "current"),
    "anticipation": ("CSC413-anticipation", 1e-4, "Steering anticipation", "anticipation"),
}
KEYS = ("train/loss", "val/mean_macro_mae")
TOL = 0.00050001


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing required columns")
        yield from reader


def step_number(value):
    n = float(value)
    if not math.isfinite(n) or not n.is_integer() or not 1 <= n <= 60:
        raise ValueError("invalid epoch/step")
    return int(n)


def score_value(value):
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise ValueError("non-finite or negative metric")
    return n


def read_data(scalars, validation):
    data, identities = {}, {}
    required = {"project", "run_name", "run_id", "key", "step", "value"}
    for row in rows(scalars, required):
        if row["key"] not in KEYS:
            continue
        match = re.fullmatch(r"transformer_lr([^_]+)_s([0-7])", row["run_name"])
        if not match:
            continue
        task = next((t for t, (project, rate, _, _) in TASKS.items()
                     if row["project"] == project and math.isclose(float(match[1]), rate, rel_tol=1e-10)), None)
        if task is None:
            continue
        run = task, int(match[2])
        identity = row["run_id"].strip()
        if not identity:
            raise ValueError(f"{run}: missing run identity")
        identities.setdefault(run, set()).add(identity)
        values = data.setdefault(run, {}).setdefault(row["key"], {})
        step = step_number(row["step"])
        if step in values:
            raise ValueError(f"{run}, {row['key']}: duplicate step {step}")
        values[step] = score_value(row["value"])
    reference = {}
    for row in rows(validation, {"experiment", "model", "lr", "seed", "epoch", "val_selection_score"}):
        task = row["experiment"]
        if task not in TASKS or row["model"] != "transformer":
            continue
        if not math.isclose(float(row["lr"]), TASKS[task][1], rel_tol=1e-10):
            continue
        seed = int(row["seed"])
        if seed not in range(8):
            continue
        run, epoch = (task, seed), step_number(row["epoch"])
        values = reference.setdefault(run, {})
        if epoch in values:
            raise ValueError(f"{run}: duplicate validation reference epoch {epoch}")
        values[epoch] = score_value(row["val_selection_score"])
    summaries, differences = [], []
    for task in TASKS:
        for seed in range(8):
            run = task, seed
            if run not in data or set(data[run]) != set(KEYS) or len(identities[run]) != 1:
                raise ValueError(f"{run}: missing metrics or multiple run identities")
            loss, val = (data[run][k] for k in KEYS)
            expected = list(range(1, max(val) + 1))
            if sorted(loss) != expected or sorted(val) != expected or sorted(reference.get(run, {})) != expected:
                raise ValueError(f"{run}: missing or mismatched epochs")
            diff = [abs(val[e] - reference[run][e]) for e in expected]
            if max(diff) > TOL:
                raise ValueError(f"{run}: validation export mismatch {max(diff)}")
            differences.extend(diff)
            best, best_epoch = math.inf, None
            for epoch in expected:
                if val[epoch] < best - 1e-6:
                    best, best_epoch = val[epoch], epoch
            summaries.append(dict(task=task, seed=seed, learning_rate=TASKS[task][1],
                                  epochs=len(expected), best_epoch=best_epoch,
                                  best_validation_score=best, last_validation_score=val[expected[-1]],
                                  first_training_loss=loss[1], last_training_loss=loss[expected[-1]]))
    return data, summaries, differences


def plot(task, data, summaries, out):
    blue, gray = "#315f84", "#b2b2b2"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, KEYS, ("Training loss (standardized MSE)", "Validation selection score (°)")):
        for seed in [*range(1, 8), 0]:
            values = data[task, seed][metric]
            epochs = np.array(sorted(values))
            series = np.array([values[e] for e in epochs])
            line, = ax.plot(epochs, series, color=blue if seed == 0 else gray,
                            lw=1.8 if seed == 0 else 0.9, zorder=3 if seed == 0 else 2)
            assert np.array_equal(line.get_xdata(), epochs) and np.array_equal(line.get_ydata(), series)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.02)
    selected = next(r for r in summaries if r["task"] == task and r["seed"] == 0)
    axes[1].plot(selected["best_epoch"], selected["best_validation_score"], "o", color=blue, ms=5, zorder=4)
    fig.suptitle(f"Transformer — {TASKS[task][2]}", fontsize=13, y=0.98)
    handles = [Line2D([], [], color=blue, lw=1.8, label="Seed 0"),
               Line2D([], [], color=gray, lw=0.9, label="Seeds 1–7"),
               Line2D([], [], color=blue, marker="o", ls="", ms=5, label="Selected checkpoint (seed 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    name = f"transformer_{TASKS[task][3]}_training"
    fig.savefig(out / f"{name}.png", dpi=200, facecolor="white")
    fig.savefig(out / f"{name}.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    return [f"{name}.png", f"{name}.svg"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scalars", type=Path, required=True)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    if any((p / ".git").exists() for p in [out, *out.parents]):
        raise ValueError("output must be outside a Git working tree")
    names = [f"transformer_{label}_training.{ext}" for label in ("current", "anticipation") for ext in ("png", "svg")]
    names += ["summary.json", "README.txt", "plot_transformer_curves.py"]
    if any((out / name).exists() for name in names + ["transformer_training_curves.zip"]):
        raise FileExistsError("output files already exist; choose a new output directory")
    before = {"scalars": sha(args.scalars), "validation": sha(args.validation)}
    data, summaries, differences = read_data(args.scalars, args.validation)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "transformer-curves"}):
        for task in TASKS:
            plot(task, data, summaries, out)
    script = Path(__file__).resolve()
    shutil.copy2(script, out / script.name)
    summary = dict(runs=summaries, run_count=16, matched_validation_points=len(differences),
                   maximum_validation_difference=max(differences), validation_rounding_tolerance=TOL,
                   unique_run_identity_per_seed=True, continuous_aligned_epochs=True,
                   plotted_points_match_logs=True, smoothing=False, extrapolation=False,
                   sources={key: {"filename": path.name, "sha256": before[key]} for key, path in
                            (("scalars", args.scalars), ("validation", args.validation))},
                   script_sha256=sha(script))
    after = {"scalars": sha(args.scalars), "validation": sha(args.validation)}
    if before != after:
        raise RuntimeError("input files changed during plotting")
    summary["input_files_unchanged"] = True
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "README.txt").write_text(
        "Transformer training curves\n\n"
        "Existing final runs only: seeds 0–7; current-angle lr=3e-4; anticipation lr=1e-4.\n"
        "Blue: seed 0, selected in advance for display. Gray: seeds 1–7.\n"
        "Each line ends at its last recorded epoch. No smoothing, interpolation or seed averaging.\n"
        "The circle marks seed 0's selected checkpoint using the training improvement threshold of 1e-6.\n"
        "Epoch is the scalar export's step, as logged by the original training script.\n\n"
        "Training loss is minibatch-averaged MSE on standardized targets, without degree units.\n"
        "Anticipation training loss includes all four outputs: 0, 0.5, 1 and 2 seconds.\n"
        "Validation selection scores are macro MAE in degrees, averaging 100%, 50% and 25% retention.\n"
        "Anticipation validation also averages 0.5, 1 and 2 seconds; it excludes the 0-second output.\n"
        "Per-epoch selection uses one fixed mask (seed 7) per sparse condition, not the final five-mask scores.\n"
        "The two panels use different metrics and scales; their vertical separation is not a generalization gap.\n\n"
        "All validation points were checked against val_curves_final.csv within its three-decimal rounding.\n"
        "summary.json contains aggregate checks and file hashes; original CSV files stay on the server.\n"
        "PNG is for viewing; SVG is for resizing. No training, inference or test evaluation was run.\n")
    with zipfile.ZipFile(out / "transformer_training_curves.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(out / name, name)
    print(json.dumps(summary, indent=2))
    print(out / "transformer_training_curves.zip")


if __name__ == "__main__":
    main()
