"""Linear probes on cached frozen features, for every encoder x subset x LR.

    python scripts/probe.py --random \
        --encoder m75_ep400=checkpoints/m75/encoder_m75_ep400.pt \
        --encoder m75_ep200=checkpoints/m75/encoder_m75_ep200.pt \
        --encoder m50_ep200=checkpoints/m50/encoder_m50_ep200.pt \
        --encoder m90_ep200=checkpoints/m90/encoder_m90_ep200.pt
Features are cached in --features (one pass per encoder), so each probe takes seconds.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.data import cifar_arrays, load_split  # noqa: E402
from mae.engine import extract_features, probe_run  # noqa: E402
from mae.utils import get_device, load_config, load_encoder, log_run, seed_everything  # noqa: E402
from mae.vit import vit_tiny  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/probe.yaml")
    ap.add_argument("--encoder", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--random", action="store_true", help="also probe a random-init frozen encoder")
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="./results")
    ap.add_argument("--features", default="./features")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.data:
        cfg["data_root"] = args.data
    device = get_device()
    xtr, ytr = cifar_arrays(cfg["data_root"], True, cfg.get("download", True))
    xte, yte = cifar_arrays(cfg["data_root"], False, cfg.get("download", True))
    ytr, yte = torch.from_numpy(ytr), torch.from_numpy(yte)

    encoders = [tuple(e.split("=", 1)) for e in args.encoder]
    if args.random:
        encoders = [("random", None)] + encoders
    os.makedirs(args.features, exist_ok=True)

    splits = {(fr, sd): load_split(cfg["splits_dir"], fr, sd) for fr, sd in cfg["subsets"]}
    needed = np.unique(np.concatenate([np.concatenate(v) for v in splits.values()]))
    if cfg.get("test_subset"):                          # smoke tests only
        xte, yte = xte[:cfg["test_subset"]], yte[:cfg["test_subset"]]

    for name, path in encoders:
        cache = os.path.join(args.features, f"{name}.pt")
        feats = torch.load(cache) if os.path.exists(cache) else None
        if feats is None or not feats["have"][needed].all() or len(feats["test"]) != len(xte):
            seed_everything(cfg.get("random_init_seed", 0))
            enc = vit_tiny()                         # random init unless weights are loaded
            meta = load_encoder(path, enc) if path else {"mask_ratio": ""}
            t0 = time.time()
            train = torch.zeros(len(xtr), enc.embed_dim)
            train[needed] = extract_features(enc, xtr[needed], device)   # only images a split uses
            have = torch.zeros(len(xtr), dtype=torch.bool)
            have[needed] = True
            feats = {"train": train, "have": have, "test": extract_features(enc, xte, device),
                     "ratio": meta.get("mask_ratio", "")}
            torch.save(feats, cache)
            print(f"[{name}] features for {len(needed)} train + {len(xte)} test images in "
                  f"{time.time() - t0:.0f}s")
        for frac, seed in cfg["subsets"]:
            tr, va = splits[(frac, seed)]
            for lr in cfg["lr_sweep"]:
                t0 = time.time()
                best_val, test = probe_run(feats["train"][tr], ytr[tr], feats["train"][va], ytr[va],
                                           feats["test"], yte, lr, device, epochs=cfg["epochs"],
                                           batch_size=cfg["batch_size"], seed=seed)
                log_run(args.results, {
                    "run_id": f"lp_{name}_f{frac}_s{seed}_lr{lr:g}", "config": cfg["config_name"],
                    "kind": "probe", "arm": "random" if name == "random" else "mae", "encoder": name,
                    "ratio": feats["ratio"], "fraction": frac, "seed": seed, "lr": lr,
                    "best_val_acc": round(best_val, 4), "test_acc": round(test, 4),
                    "wall_time_s": round(time.time() - t0, 1)})
                print(f"[{name}] {frac}% s{seed} lr {lr:g}: val {best_val:.4f} test {test:.4f}", flush=True)


if __name__ == "__main__":
    main()
