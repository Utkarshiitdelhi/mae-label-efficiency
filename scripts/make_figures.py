"""Build every figure and table from runs.csv (+ per-run val logs) alone.

    python scripts/make_figures.py                         # reads results/runs.csv
    python scripts/make_figures.py --runs a/runs.csv b/runs.csv   # merge Kaggle sessions

Selection rules (test accuracy is never used to choose anything):
  fine-tuning: per (arm, encoder, fraction) the LR with the best seed-0 val accuracy;
               all seeds at that LR are reported.
  probing:     per (encoder, fraction, seed) the LR with the best val accuracy.
Outputs: figures/label_efficiency.png, figures/ablation.png, figures/val_curves.png,
         tables/label_efficiency.md, tables/ablation.md, summary.json, merged runs.csv.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.utils import RUN_FIELDS, read_runs, select_lr  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference palette (dataviz skill): categorical slots in fixed order + neutral for the control
C_MAE, C_SCRATCH, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_RANDOM = "#8c8b86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FRACS = [1, 10, 100]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def f(r, k):
    return float(r[k]) if r.get(k) not in (None, "") else float("nan")


def stats(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return None
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v)), "values": [float(x) for x in v]}


def finetune_cell(rows, arm, encoder, frac):
    """Rows for one (arm, encoder, fraction) at its seed-0-val-chosen LR."""
    lr = select_lr(rows, "finetune", encoder, frac, seed=0, arm=arm)
    if lr is None:
        return None, []
    cell = [r for r in rows if r["kind"] == "finetune" and r["arm"] == arm and r["encoder"] == encoder
            and int(r["fraction"]) == frac and abs(f(r, "lr") - lr) < 1e-12]
    return lr, sorted(cell, key=lambda r: int(r["seed"]))


def probe_cell(rows, encoder, frac):
    """Per seed: test acc at the LR with best val acc."""
    out = {}
    for r in rows:
        if r["kind"] == "probe" and r["encoder"] == encoder and int(r["fraction"]) == frac:
            s = int(r["seed"])
            if s not in out or f(r, "best_val_acc") > f(out[s], "best_val_acc"):
                out[s] = r
    return [out[s] for s in sorted(out)]


def pct(x):
    return "n/a" if x is None else (f"{100 * x['mean']:.1f} ± {100 * x['std']:.1f}"
                                     if x["n"] > 1 else f"{100 * x['mean']:.1f}")


# ---------------------------------------------------------------------------- main result
def label_efficiency(rows, main_enc, outdir, tag):
    res = {}
    for frac in FRACS:
        lr_m, mae = finetune_cell(rows, "mae", main_enc, frac)
        lr_s, scr = finetune_cell(rows, "scratch", "none", frac)
        sm = {int(r["seed"]): f(r, "test_acc") for r in mae}
        ss = {int(r["seed"]): f(r, "test_acc") for r in scr}
        paired = [sm[s] - ss[s] for s in sorted(set(sm) & set(ss))]
        res[frac] = {"mae": stats(list(sm.values())), "scratch": stats(list(ss.values())),
                     "gain": stats(paired), "lr_mae": lr_m, "lr_scratch": lr_s}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(len(FRACS))
    for key, color, label, dx in [("mae", C_MAE, f"MAE-75 init ({main_enc})", -0.05),
                                  ("scratch", C_SCRATCH, "From scratch", 0.05)]:
        pts = [(i + dx, res[fr][key]) for i, fr in enumerate(FRACS) if res[fr][key]]
        if not pts:
            continue
        xi = [p[0] for p in pts]
        m = [100 * p[1]["mean"] for p in pts]
        s = [100 * p[1]["std"] for p in pts]
        ax.errorbar(xi, m, yerr=s, color=color, lw=2, marker="o", ms=8, capsize=4,
                    mec="white", mew=1.5, label=label, zorder=3)
        ax.annotate(label.split(" (")[0], (xi[-1], m[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    for i, fr in enumerate(FRACS):
        g = res[fr]["gain"]
        if g and res[fr]["mae"] and res[fr]["scratch"]:
            top = 100 * max(res[fr][k]["mean"] + res[fr][k]["std"] for k in ("mae", "scratch"))
            sign = "+" if g["mean"] >= 0 else ""
            ax.annotate(f"gain {sign}{100 * g['mean']:.1f} pts", (i, top), xytext=(0, 10),
                        textcoords="offset points", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [f"{fr}%\n({[500, 5000, 50000][i]:,} labels)" for i, fr in enumerate(FRACS)])
    ax.set_xlim(-0.4, len(FRACS) - 0.4 + 0.5)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.margins(y=0.15)
    for i, fr in enumerate(FRACS):
        if not (res[fr]["mae"] or res[fr]["scratch"]):
            ax.annotate("not run yet", (i, 0.5), xycoords=("data", "axes fraction"), ha="center",
                        color=INK2, fontsize=8.5)
    ax.set_xlabel("Labelled fraction of the training set (val carved from it)")
    ax.set_title(f"Label efficiency: ViT-Tiny, MAE pretraining vs matched scratch{tag}",
                 fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower right")
    fig.text(0.01, 0.005, "Mean ± std over seeds (3 at 1%, 1 at 10%/100%); gain = paired MAE − scratch.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(outdir, "figures", "label_efficiency.png"), dpi=170)
    plt.close(fig)

    lines = ["| Label fraction | Labelled images | MAE-75 fine-tune (%) | Scratch (%) | Paired gain (pts) | LR (MAE / scratch) | Seeds |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for i, fr in enumerate(FRACS):
        r = res[fr]
        g = r["gain"]
        gain = "n/a" if not g else (f"{100 * g['mean']:+.1f} ± {100 * g['std']:.1f}" if g["n"] > 1
                                    else f"{100 * g['mean']:+.1f}")
        n = max((r[k]["n"] for k in ("mae", "scratch") if r[k]), default=0)
        lines.append(f"| {fr}% | {[500, 5000, 50000][i]:,} | {pct(r['mae'])} | {pct(r['scratch'])} | "
                     f"{gain} | {r['lr_mae'] or '-'} / {r['lr_scratch'] or '-'} | {n} |")
    return res, "\n".join(lines)


# ---------------------------------------------------------------------------- ablation
def ablation(rows, outdir, tag):
    probes_present = {r["encoder"] for r in rows if r["kind"] == "probe"}
    m75 = "m75e200_ep200" if "m75e200_ep200" in probes_present else "m75_ep200"
    encs = [("random", "Random init (frozen)", C_RANDOM), ("m50_ep200", "MAE-50", C_SCRATCH),
            (m75, "MAE-75" + (" (own 200-ep run)" if m75.startswith("m75e200") else " (ep200 of 400)"), C_MAE), ("m90_ep200", "MAE-90", C_AQUA)]
    probe = {e: {fr: stats([f(r, "test_acc") for r in probe_cell(rows, e, fr)]) for fr in FRACS}
             for e, _, _ in encs}
    ft = {}
    for e, _, _ in encs[1:]:
        _, cell = finetune_cell(rows, "mae", e, 1)
        ft[e] = stats([f(r, "test_acc") for r in cell])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    w = 0.19
    for j, (e, label, color) in enumerate(encs):
        xs, ms, ss = [], [], []
        for i, fr in enumerate(FRACS):
            if probe[e][fr]:
                xs.append(i + (j - 1.5) * (w + 0.01))
                ms.append(100 * probe[e][fr]["mean"])
                ss.append(100 * probe[e][fr]["std"])
        if xs:
            a1.bar(xs, ms, w, color=color, label=label, edgecolor="white", linewidth=2, zorder=3)
            a1.errorbar(xs, ms, yerr=ss, fmt="none", ecolor=INK2, capsize=2, lw=1, zorder=4)
    for i, fr in enumerate(FRACS):
        if not any(probe[e][fr] for e, _, _ in encs):
            a1.annotate("not run yet", (i, 0.5), xycoords=("data", "axes fraction"), ha="center",
                        color=INK2, fontsize=8.5)
    a1.set_xticks(range(len(FRACS)), [f"{fr}%" for fr in FRACS])
    a1.set_xlim(-0.5, len(FRACS) - 0.5)
    a1.set_xlabel("Labelled fraction")
    a1.set_ylabel("Linear-probe test accuracy (%)")
    a1.set_title("Linear probe, frozen encoder (200 pretraining epochs)", fontsize=10, loc="left", color=INK)
    a1.legend(loc="upper left", fontsize=8.5, ncol=2)
    a1.set_ylim(0, max([100 * probe[e][fr]["mean"] for e, _, _ in encs for fr in FRACS
                        if probe[e][fr]] + [10]) * 1.22)

    bars = [(label, color, ft[e]) for e, label, color in encs[1:] if ft.get(e)]
    for k, (label, color, st) in enumerate(bars):
        a2.bar(k, 100 * st["mean"], 0.6, color=color, edgecolor="white", linewidth=2, zorder=3)
        a2.errorbar(k, 100 * st["mean"], yerr=100 * st["std"], fmt="none", ecolor=INK2, capsize=3, lw=1, zorder=4)
        a2.annotate(f"{100 * st['mean']:.1f}", (k, 100 * (st["mean"] + st["std"])), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    a2.set_xticks(range(len(bars)), [b[0].split(" (")[0] for b in bars])
    a2.set_ylabel("Fine-tune test accuracy at 1% (%)")
    a2.set_title("Full fine-tune, 1% labels", fontsize=10, loc="left", color=INK)
    if bars:
        a2.set_ylim(0, 100 * max(b[2]["mean"] + b[2]["std"] for b in bars) * 1.15)
    fig.suptitle(f"Masking-ratio ablation (mean ± std over seeds at 1%){tag}", fontsize=10.5,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figures", "ablation.png"), dpi=170)
    plt.close(fig)

    lines = ["| Encoder | Probe 1% (%) | Probe 10% (%) | Probe 100% (%) | Fine-tune 1% (%) |",
             "| --- | --- | --- | --- | --- |"]
    for e, label, _ in encs:
        lines.append(f"| {label} (`{e}`) | {pct(probe[e][1])} | {pct(probe[e][10])} | "
                     f"{pct(probe[e][100])} | {pct(ft.get(e)) if e != 'random' else '-'} |")
    # main 400-epoch encoder probe, for reference
    if "m75_ep400" in probes_present:
        p = {fr: stats([f(r, "test_acc") for r in probe_cell(rows, "m75_ep400", fr)]) for fr in FRACS}
        lines.append(f"| MAE-75, 400 ep (`m75_ep400`, main model) | {pct(p[1])} | {pct(p[10])} | "
                     f"{pct(p[100])} | see label-efficiency table |")
    return {"probe": probe, "finetune_1pct": ft, "m75_ablation_encoder": m75}, "\n".join(lines)


# ---------------------------------------------------------------------------- val curves
def val_curves(rows, main_enc, logs_dirs, outdir, tag):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    drawn = False
    for ax, frac in zip(axes, FRACS):
        for arm, enc, color, label in [("mae", main_enc, C_MAE, "MAE-75 init"),
                                       ("scratch", "none", C_SCRATCH, "From scratch")]:
            _, cell = finetune_cell(rows, arm, enc, frac)
            seed0 = [r for r in cell if int(r["seed"]) == 0]
            if not seed0:
                continue
            path = next((os.path.join(d, f"{seed0[0]['run_id']}.csv") for d in logs_dirs
                         if os.path.exists(os.path.join(d, f"{seed0[0]['run_id']}.csv"))), None)
            if not path:
                continue
            with open(path) as fh:
                log = list(csv.DictReader(fh))
            ax.plot([int(r["step"]) for r in log], [100 * float(r["val_acc"]) for r in log],
                    color=color, lw=2, label=label)
            drawn = True
        if not ax.lines:
            ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes, ha="center", color=INK2, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{frac}% labels (seed 0, chosen LR)", fontsize=9.5, loc="left", color=INK)
        ax.set_xlabel("Fine-tuning step")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle(f"Validation curves: has the scratch baseline plateaued?{tag}", fontsize=10.5,
                 x=0.01, ha="left", color=INK)
    fig.tight_layout()
    if drawn:
        fig.savefig(os.path.join(outdir, "figures", "val_curves.png"), dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["results/runs.csv"])
    ap.add_argument("--logs", nargs="*", default=None, help="dirs with per-run logs (default: <runs dir>/logs)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--main-encoder", default="m75_ep400")
    ap.add_argument("--tag", default="", help="suffix for titles, e.g. ' [SMOKE TEST]'")
    args = ap.parse_args()

    rows = read_runs(args.runs)
    if not rows:
        sys.exit(f"no rows in {args.runs}")
    logs = args.logs if args.logs is not None else [os.path.join(os.path.dirname(p) or ".", "logs") for p in args.runs]
    for d in ("figures", "tables"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    merged = os.path.join(args.out, "runs.csv")
    if [os.path.abspath(p) for p in args.runs] != [os.path.abspath(merged)]:
        with open(merged, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    le, le_md = label_efficiency(rows, args.main_encoder, args.out, args.tag)
    ab, ab_md = ablation(rows, args.out, args.tag)
    val_curves(rows, args.main_encoder, logs, args.out, args.tag)
    with open(os.path.join(args.out, "tables", "label_efficiency.md"), "w") as fh:
        fh.write(le_md + "\n")
    with open(os.path.join(args.out, "tables", "ablation.md"), "w") as fh:
        fh.write(ab_md + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({"label_efficiency": {str(k): v for k, v in le.items()},
                   "ablation": {"probe": {e: {str(k): v for k, v in d.items()} for e, d in ab["probe"].items()},
                                "finetune_1pct": ab["finetune_1pct"],
                                "m75_ablation_encoder": ab["m75_ablation_encoder"]}}, fh, indent=1)

    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(f"{len(rows)} runs ({dict(kinds)})\n\n{le_md}\n\n{ab_md}")


if __name__ == "__main__":
    main()
