"""MAE pretraining on the 50K CIFAR-10 training images (labels unused).

    python scripts/pretrain.py --config configs/pretrain_m75.yaml
    # timing run (Milestone 1):   --set epochs=5 run_name=m75_timing recon_every=0
    # LR sanity check (20 ep):    --set epochs=20 warmup_epochs=2 lr=3e-4 run_name=m75_lr3e-4
Resumes automatically from <out>/ckpt_latest.pt if it exists.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.engine import pretrain_mae  # noqa: E402
from mae.utils import get_device, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None, help="CIFAR-10 root (overrides config)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default ./checkpoints/<run_name>)")
    ap.add_argument("--results", default="./results")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.data:
        cfg["data_root"] = args.data
    out = args.out or os.path.join("checkpoints", cfg["run_name"])
    device = get_device()
    print(f"device: {device}  run: {cfg['run_name']}  mask {cfg['mask_ratio']}  epochs {cfg['epochs']}")
    if device.type == "cuda":
        import torch
        torch.backends.cudnn.benchmark = True
    pretrain_mae(cfg, out, args.results, device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
