"""Merge runs.csv files from several Kaggle sessions into one (later rows win on run_id).

    python scripts/merge_runs.py --out results/runs.csv /kaggle/input/*/results/runs.csv results/runs.csv
Per-run val logs (logs/*.csv) next to each input are copied too.
"""
import argparse
import csv
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.utils import RUN_FIELDS, read_runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results/runs.csv")
    a = ap.parse_args()
    paths = [p for pat in a.inputs for p in sorted(glob.glob(pat))]
    rows = read_runs(paths)
    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    for p in paths:
        logs = os.path.join(os.path.dirname(p), "logs")
        if os.path.isdir(logs) and os.path.abspath(logs) != os.path.abspath(os.path.join(out_dir, "logs")):
            for fp in glob.glob(os.path.join(logs, "*.csv")):
                shutil.copy2(fp, os.path.join(out_dir, "logs", os.path.basename(fp)))
    tmp = a.out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, a.out)
    print(f"merged {len(paths)} files -> {len(rows)} runs in {a.out}")


if __name__ == "__main__":
    main()
