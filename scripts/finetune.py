"""Full fine-tuning at a label fraction, MAE-initialised or from scratch.

    # LR sweep on seed 0 (both arms, each fraction):
    python scripts/finetune.py --arm scratch --fraction 1 --seeds 0
    python scripts/finetune.py --arm mae --fraction 1 --seeds 0 \
        --encoder checkpoints/m75/encoder_m75_ep400.pt --encoder-name m75_ep400
    # remaining seeds at the LR each arm chose on seed-0 validation accuracy:
    python scripts/finetune.py --arm mae --fraction 1 --seeds 1 2 --lr-from-sweep \
        --encoder ... --encoder-name m75_ep400
    # ablation: MAE-50 at 1% with the LR chosen for MAE-75 at 1%:
    python scripts/finetune.py --arm mae --fraction 1 --seeds 0 1 2 --lr-like m75_ep400 \
        --encoder checkpoints/m50/encoder_m50_ep200.pt --encoder-name m50_ep200
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.engine import finetune_run  # noqa: E402
from mae.utils import get_device, load_config, read_runs, select_lr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--arm", choices=["mae", "scratch"], required=True)
    ap.add_argument("--fraction", type=int, choices=[1, 10, 100], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--lrs", type=float, nargs="+", help="explicit LRs (default: config lr_sweep)")
    g.add_argument("--lr-from-sweep", action="store_true",
                   help="use the LR this arm/encoder chose on seed-0 val at this fraction")
    g.add_argument("--lr-like", metavar="ENCODER_NAME",
                   help="use the LR chosen for another encoder at this fraction (ablation)")
    ap.add_argument("--encoder", help="encoder-only checkpoint (MAE arm)")
    ap.add_argument("--encoder-name", default=None, help="e.g. m75_ep400")
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default="./results")
    ap.add_argument("--out", default="./checkpoints/finetune")
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if args.data:
        cfg["data_root"] = args.data
    if args.arm == "mae":
        assert args.encoder and os.path.exists(args.encoder), "--encoder checkpoint required for MAE arm"
        name = args.encoder_name or os.path.splitext(os.path.basename(args.encoder))[0].replace("encoder_", "")
        import torch
        cfg["_ratio"] = torch.load(args.encoder, map_location="cpu", weights_only=False)["meta"].get("mask_ratio", "")
    else:
        name = "none"

    runs_csv = os.path.join(args.results, "runs.csv")
    if args.lr_from_sweep or args.lr_like:
        rows = read_runs([runs_csv])
        src_enc = args.lr_like or name
        src_arm = "mae" if args.lr_like else args.arm
        lr = select_lr(rows, "finetune", src_enc, args.fraction, seed=0, arm=src_arm)
        assert lr is not None, f"no seed-0 sweep found for {src_enc} at {args.fraction}% in {runs_csv}"
        lrs = [lr]
        print(f"using LR {lr:g} chosen on seed-0 val for {src_enc} at {args.fraction}%")
    else:
        lrs = args.lrs or cfg["lr_sweep"]

    device = get_device()
    if device.type == "cuda":
        import torch
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    for seed in args.seeds:
        for lr in lrs:
            finetune_run(cfg, args.arm, args.fraction, seed, lr, device, args.results,
                         encoder_path=args.encoder, encoder_name=name,
                         max_steps=args.max_steps, out_dir=args.out)


if __name__ == "__main__":
    main()
