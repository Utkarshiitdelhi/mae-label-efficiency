"""Generate the class-balanced label subsets once and save them as committed JSON index files.

    python scripts/make_splits.py --data ./data

Default subsets (design doc): 1% x seeds {0,1,2}, 10% x seed 0, 100% x seed 0.
Every fine-tuning and probing run loads indices from these files, so both arms of a
comparison always see exactly the same images.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.data import cifar_arrays, make_split, split_path  # noqa: E402

DEFAULT = [(1, 0), (1, 1), (1, 2), (10, 0), (100, 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "splits"))
    ap.add_argument("--extra", nargs="*", default=[], help="extra frac:seed pairs, e.g. 10:1 10:2")
    args = ap.parse_args()

    _, labels = cifar_arrays(args.data, train=True)
    pairs = DEFAULT + [tuple(int(v) for v in s.split(":")) for s in args.extra]
    os.makedirs(args.out, exist_ok=True)
    for frac, seed in pairs:
        train, val = make_split(labels, frac, seed)
        counts_tr = np.bincount(labels[train], minlength=10).tolist()
        counts_va = np.bincount(labels[val], minlength=10).tolist()
        with open(split_path(args.out, frac, seed), "w") as f:
            json.dump({"fraction": frac, "seed": seed, "n_train": len(train), "n_val": len(val),
                       "train_per_class": counts_tr, "val_per_class": counts_va,
                       "train": train, "val": val}, f)
        print(f"frac{frac}_seed{seed}: train {len(train)} val {len(val)} "
              f"(per class {counts_tr[0]}/{counts_va[0]})")


if __name__ == "__main__":
    main()
