"""Training loops shared by the scripts and the tests.

  pretrain_mae   MAE pretraining with exact resume (scripts/pretrain.py)
  finetune_run   full fine-tuning, identical recipe for MAE and scratch arms (scripts/finetune.py)
  probe_*        cached-feature linear probing (scripts/probe.py)
"""
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (GPUImageBank, IndexedCIFAR, cifar_arrays, eval_transform, finetune_transform,
                   infinite_loader, load_cifar, load_split, CIFAR_MEAN, CIFAR_STD)
from .mae import MAE, reconstruct
from .utils import (append_csv, gpu_mem_mb, load_checkpoint, load_encoder, log_run,
                    save_checkpoint, save_encoder, seed_everything, set_lr, warmup_cosine)
from .vit import ViTClassifier, param_groups, vit_tiny


def _amp(device, enabled=True):
    on = enabled and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=on)


def _scaler(device, enabled=True):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def encoder_kwargs(cfg):
    """Optional architecture overrides (used only by the tiny CPU tests)."""
    return dict(cfg.get("encoder", {}) or {})


# ====================================================================== pretraining
def build_mae(cfg):
    enc = vit_tiny(**encoder_kwargs(cfg))
    return MAE(enc, mask_ratio=cfg["mask_ratio"], dec_dim=cfg.get("dec_dim", 128),
               dec_depth=cfg.get("dec_depth", 4), dec_heads=cfg.get("dec_heads", 4),
               norm_pix_loss=cfg.get("norm_pix_loss", True))


def save_recon_grid(model, imgs, path, title="", seed=0):
    """Rows: original / masked input / prediction pasted into the visible patches."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(imgs.shape[0], model.encoder.patch_embed.num_patches, generator=g)
    with torch.no_grad():
        masked, pasted, _ = reconstruct(model, imgs, noise=noise.to(imgs.device))
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    den = lambda t: (t.float().cpu() * std + mean).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    orig, msk, rec = den(imgs), den(masked), den(pasted)
    msk[(masked.cpu() == 0).all(1).numpy()] = 0.5          # grey for hidden patches
    n = imgs.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(1.2 * n, 4.0))
    for j in range(n):
        for i, (row, name) in enumerate([(orig, "original"), (msk, "masked input"),
                                          (rec, "reconstruction")]):
            ax = axes[i, j]
            ax.imshow(row[j], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    model.train()


def pretrain_mae(cfg, out_dir, results_dir, device, train_images=None, recon_images=None,
                 resume=True, stop_after_epoch=None, log_every_epoch=True):
    """MAE pretraining. `train_images`: uint8 NHWC array (default: CIFAR-10 train, labels unused).

    Checkpoints `ckpt_latest.pt` every `ckpt_every` epochs (and at the end) with model,
    optimiser, scaler, epoch, step and RNG state; resuming continues bit-for-bit on CPU.
    Saves encoder-only weights every `encoder_save_every` epochs for downstream use.
    """
    run = cfg["run_name"]
    seed_everything(cfg.get("seed", 0))
    if train_images is None:
        train_images, _ = cifar_arrays(cfg["data_root"], train=True, download=cfg.get("download", True))
        if cfg.get("train_subset"):                 # smoke tests only
            train_images = train_images[:cfg["train_subset"]]
    if recon_images is None and cfg.get("recon_every"):
        test_imgs, _ = cifar_arrays(cfg["data_root"], train=False, download=cfg.get("download", True))
        recon_images = test_imgs[cfg.get("recon_indices", [0, 1, 2, 3, 4, 5, 6, 7])]
    bank = GPUImageBank(train_images, device)
    recon_x = bank.normalise(torch.from_numpy(recon_images).permute(0, 3, 1, 2).to(device)) \
        if recon_images is not None else None

    model = build_mae(cfg).to(device)
    bs = cfg["batch_size"]
    peak = cfg.get("lr") or cfg["base_lr"] * bs / 256          # linear scaling rule
    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=peak,
                            betas=tuple(cfg.get("betas", (0.9, 0.95))))
    scaler = _scaler(device, cfg.get("amp", True))

    os.makedirs(out_dir, exist_ok=True)
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    start_epoch, step, wall = 0, 0, 0.0
    if resume and os.path.exists(latest):
        st = load_checkpoint(latest, model, opt, scaler, map_location=device)
        start_epoch, step = st["epoch"], st["step"]
        wall = st["extra"].get("wall_time_s", 0.0)
        print(f"[{run}] resumed from epoch {start_epoch} (step {step})")

    epochs = cfg["epochs"]
    spe = bank.n // bs
    total, warm = epochs * spe, cfg["warmup_epochs"] * spe
    log_path = os.path.join(results_dir, "logs", f"pt_{run}.csv")
    clip = cfg.get("clip_grad")
    loss_val = float("nan")

    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        for xb in bank.epoch_batches(bs, epoch_seed=cfg.get("seed", 0) * 100_003 + epoch):
            lr = warmup_cosine(step, total, warm, peak)
            set_lr(opt, lr)
            with _amp(device, cfg.get("amp", True)):
                loss, _, _ = model(xb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.detach().float()
            n_steps += 1
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        wall += dt
        loss_val = (loss_sum / max(n_steps, 1)).item()
        if not np.isfinite(loss_val):
            raise FloatingPointError(f"[{run}] loss is {loss_val} at epoch {epoch + 1}; "
                                     "set clip_grad: 1.0 and resume")
        ep = epoch + 1
        append_csv(log_path, {"epoch": ep, "train_loss": round(loss_val, 5), "lr": lr,
                              "epoch_time_s": round(dt, 2), "gpu_mem_mb": gpu_mem_mb()})
        if log_every_epoch:
            print(f"[{run}] ep {ep}/{epochs} loss {loss_val:.4f} lr {lr:.2e} {dt:.1f}s "
                  f"mem {gpu_mem_mb()}MB", flush=True)
        if recon_x is not None and cfg.get("recon_every") and (ep % cfg["recon_every"] == 0 or ep == epochs):
            save_recon_grid(model, recon_x, os.path.join(out_dir, "recon", f"{run}_ep{ep}.png"),
                            title=f"MAE {int(cfg['mask_ratio'] * 100)}% masking, epoch {ep} (held-out test images)")
        if ep % cfg.get("encoder_save_every", 50) == 0 or ep == epochs:
            save_encoder(os.path.join(out_dir, f"encoder_{run}_ep{ep}.pt"), model.encoder,
                         {"run": run, "epoch": ep, "mask_ratio": cfg["mask_ratio"],
                          "encoder_kwargs": encoder_kwargs(cfg)})
        if ep % cfg.get("ckpt_every", 25) == 0 or ep == epochs:
            save_checkpoint(latest, model, opt, scaler, ep, step, {"wall_time_s": wall})
        if stop_after_epoch is not None and ep >= stop_after_epoch:
            return model

    if start_epoch < epochs:   # only log a completed run once
        log_run(results_dir, {"run_id": f"pt_{run}", "config": cfg.get("config_name", ""),
                              "kind": "pretrain", "arm": "mae", "encoder": f"{run}_ep{epochs}",
                              "ratio": cfg["mask_ratio"], "final_loss": round(loss_val, 5),
                              "steps": step, "seed": cfg.get("seed", 0),
                              "wall_time_s": round(wall, 1)})
    return model


# ====================================================================== evaluation helpers
@torch.no_grad()
def evaluate(model, loader, device, amp=True):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with _amp(device, amp):
            logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / max(total, 1)


# ====================================================================== fine-tuning
def finetune_run(cfg, arm, fraction, seed, lr, device, results_dir, encoder_path=None,
                 encoder_name="none", max_steps=None, out_dir=None):
    """Full fine-tuning. The MAE and scratch arms differ ONLY in the encoder's starting
    weights: same subset/split, augmentation, step budget, optimiser and schedule."""
    assert arm in ("mae", "scratch")
    budget = cfg["budgets"][int(fraction)]
    bs, steps = budget["batch_size"], max_steps or budget["steps"]
    run_id = f"ft_{encoder_name if arm == 'mae' else 'scratch'}_f{fraction}_s{seed}_lr{lr:g}"
    seed_everything(seed)

    encoder = vit_tiny(drop_path_rate=cfg.get("drop_path", 0.1), **encoder_kwargs(cfg))
    if arm == "mae":
        load_encoder(encoder_path, encoder)
    model = ViTClassifier(encoder, num_classes=10).to(device)

    root, dl = cfg["data_root"], cfg.get("download", True)
    base_train, base_test = load_cifar(root, True, dl), load_cifar(root, False, dl)
    tr_idx, va_idx = load_split(cfg["splits_dir"], fraction, seed)
    test_idx = np.arange(len(base_test)) if not cfg.get("test_subset") else np.arange(cfg["test_subset"])
    nw = cfg.get("num_workers", 4)
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        IndexedCIFAR(base_train, tr_idx, finetune_transform()), batch_size=bs, shuffle=True,
        drop_last=len(tr_idx) >= bs, num_workers=nw, pin_memory=device.type == "cuda",
        persistent_workers=nw > 0, generator=g)
    mk = lambda idx, base: torch.utils.data.DataLoader(
        IndexedCIFAR(base, idx, eval_transform()), batch_size=500, num_workers=nw)
    val_loader, test_loader = mk(va_idx, base_train), mk(test_idx, base_test)

    opt = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]), lr=lr,
                            betas=tuple(cfg.get("betas", (0.9, 0.999))))
    scaler = _scaler(device, cfg.get("amp", True))
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    warm = int(cfg.get("warmup_frac", 0.05) * steps)
    eval_every = max(1, steps // cfg.get("num_evals", 30))
    log_path = os.path.join(results_dir, "logs", f"{run_id}.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    best_val, best_state, best_step = -1.0, None, 0
    t0 = time.time()
    it = infinite_loader(train_loader)
    run_loss = torch.zeros((), device=device)
    model.train()
    for step in range(steps):
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        set_lr(opt, warmup_cosine(step, steps, warm, lr))
        with _amp(device, cfg.get("amp", True)):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.get("clip_grad"):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        scaler.step(opt)
        scaler.update()
        run_loss += loss.detach().float()
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            val_acc = evaluate(model, val_loader, device, cfg.get("amp", True))
            tl = (run_loss / eval_every).item()
            run_loss.zero_()
            append_csv(log_path, {"step": step + 1, "train_loss": round(tl, 4),
                                  "val_acc": round(val_acc, 4), "lr": opt.param_groups[0]["lr"]})
            if val_acc > best_val:
                best_val, best_step = val_acc, step + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if not np.isfinite(tl):
                raise FloatingPointError(f"{run_id}: NaN loss at step {step + 1}")
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device, cfg.get("amp", True))
    wall = time.time() - t0
    if out_dir and cfg.get("save_model"):
        torch.save(best_state, os.path.join(out_dir, f"{run_id}.pt"))
    row = {"run_id": run_id, "config": cfg.get("config_name", ""), "kind": "finetune", "arm": arm,
           "encoder": encoder_name if arm == "mae" else "none",
           "ratio": cfg.get("_ratio", ""), "fraction": fraction, "seed": seed, "lr": lr,
           "best_val_acc": round(best_val, 4), "test_acc": round(test_acc, 4),
           "steps": steps, "wall_time_s": round(wall, 1)}
    log_run(results_dir, row)
    print(f"{run_id}: best val {best_val:.4f} @ step {best_step}, test {test_acc:.4f}, {wall:.0f}s",
          flush=True)
    return row


# ====================================================================== linear probing
@torch.no_grad()
def extract_features(encoder, images_nhwc, device, batch_size=1000):
    """Frozen encoder on all 64 tokens, no augmentation, mean-pooled -> (N, D) float32."""
    encoder.eval().to(device)
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images_nhwc), batch_size):
        x = torch.from_numpy(images_nhwc[i:i + batch_size]).permute(0, 3, 1, 2).to(device)
        x = (x.float() / 255 - mean) / std
        with _amp(device):
            out.append(encoder(x).mean(1).float().cpu())
    return torch.cat(out)


def probe_run(f_train, y_train, f_val, y_val, f_test, y_test, lr, device, epochs=100,
              batch_size=256, seed=0):
    """BatchNorm(affine=False) + Linear on cached features. Returns (best_val, test_at_best)."""
    seed_everything(seed)
    D = f_train.shape[1]
    head = nn.Sequential(nn.BatchNorm1d(D, affine=False, eps=1e-6), nn.Linear(D, 10)).to(device)
    nn.init.trunc_normal_(head[1].weight, std=0.01)
    nn.init.zeros_(head[1].bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    f_train, y_train = f_train.to(device), y_train.to(device)
    n = len(f_train)
    spe = max(1, -(-n // batch_size))
    total, warm = epochs * spe, max(1, int(0.05 * epochs * spe))
    g = torch.Generator().manual_seed(seed)
    step = 0
    best_val, best_test = -1.0, 0.0

    def acc(f, y):
        head.eval()
        with torch.no_grad():
            a = (head(f.to(device)).argmax(1).cpu() == y).float().mean().item()
        head.train()
        return a

    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            set_lr(opt, warmup_cosine(step, total, warm, lr))
            loss = F.cross_entropy(head(f_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        v = acc(f_val, y_val)
        if v > best_val:
            best_val, best_test = v, acc(f_test, y_test)
    return best_val, best_test
