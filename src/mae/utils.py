"""Seeding, LR schedule, checkpointing with exact resume, CSV logging, config loading."""
import csv
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import yaml


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def load_config(path: str, overrides=None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:                 # "key=value", value parsed as YAML
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg["config_name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def warmup_cosine(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr=0.0):
    """Linear warm-up to peak, then half-cosine to min_lr. Stateless: a function of step,
    so resuming only needs the step counter."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------- checkpoints
def rng_state():
    s = {"python": random.getstate(), "numpy": np.random.get_state(),
         "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path, model, optimizer, scaler, epoch, step, extra=None):
    """Everything needed for an exact resume: model, optimiser, scaler, counters, RNG.
    (The LR schedule is a pure function of `step`, so it needs no state.)"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None,
             "epoch": epoch, "step": step, "rng": rng_state(), "extra": extra or {}}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic: a killed session never leaves a half-written file


def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    set_rng_state(state["rng"])
    return state


def save_encoder(path, encoder, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "meta": meta or {}}, path)


def load_encoder(path, encoder, map_location="cpu"):
    state = torch.load(path, map_location=map_location, weights_only=False)
    encoder.load_state_dict(state["encoder"])
    return state.get("meta", {})


# ---------------------------------------------------------------- logging
RUN_FIELDS = ["run_id", "git_commit", "config", "kind", "arm", "encoder", "ratio", "fraction",
              "seed", "lr", "best_val_acc", "test_acc", "final_loss", "steps", "wall_time_s",
              "timestamp"]


def append_csv(path, row: dict, fields=None):
    """Append one row; write the header if the file is new. Unknown keys are ignored."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    fields = fields or list(row.keys())
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def log_run(results_dir, row: dict):
    row = {**row, "git_commit": git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    append_csv(os.path.join(results_dir, "runs.csv"), row, RUN_FIELDS)


def gpu_mem_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    return 0.0


# ---------------------------------------------------------------- reading results
def read_runs(paths):
    """Read one or more runs.csv files; later rows with the same run_id win (reruns)."""
    rows = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["run_id"]] = r
    return list(rows.values())


def select_lr(rows, kind, encoder, fraction, seed=0, arm=None):
    """LR with the best validation accuracy among seed-`seed` runs of one (kind, encoder,
    fraction[, arm]) cell. Test accuracy is never looked at."""
    cands = [r for r in rows if r["kind"] == kind and r["encoder"] == encoder
             and int(r["fraction"]) == int(fraction) and int(r["seed"]) == int(seed)
             and (arm is None or r["arm"] == arm)]
    if not cands:
        return None
    best = max(cands, key=lambda r: (float(r["best_val_acc"]), -float(r["lr"])))
    return float(best["lr"])
