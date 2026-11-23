"""Parameter and compute split, encoder vs decoder (CV bullet 3).

    python scripts/model_stats.py [--mask-ratio 0.75] [--md results/tables/model_stats.md]

Compute is measured with torch's FLOP counter (matrix products + attention) on one image
and reported as multiply-accumulates (MACs = FLOPs / 2).
"""
import argparse
import os
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mae.mae import MAE, random_masking  # noqa: E402
from mae.vit import vit_tiny  # noqa: E402


def macs(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--md", default=None, help="also write a markdown table here")
    a = ap.parse_args()

    torch.manual_seed(0)
    m = MAE(vit_tiny(), mask_ratio=a.mask_ratio).eval()
    x = torch.randn(1, 3, 32, 32)
    ids_keep, _, ids_restore = random_masking(1, 64, a.mask_ratio)
    with torch.no_grad():
        latent = m.encoder(x, ids_keep)
        enc = macs(lambda: m.encoder(x, ids_keep))
        dec = macs(lambda: m.decoder(latent, ids_restore))
        full = macs(lambda: m.encoder(x))
    pe = sum(p.numel() for p in m.encoder.parameters())
    pd = sum(p.numel() for p in m.decoder.parameters())
    K = ids_keep.shape[1]
    rows = [
        ("Encoder (%d visible tokens)" % K, pe, enc, f"{100 * enc / (enc + dec):.0f}%"),
        ("Decoder (64 tokens)", pd, dec, f"{100 * dec / (enc + dec):.0f}%"),
        ("Encoder at fine-tune (64 tokens)", pe, full, "n/a"),
    ]
    md = ["| Part | Parameters | Compute per image | Share of pretraining compute |",
          "| --- | --- | --- | --- |"]
    md += [f"| {n} | {p / 1e6:.2f} M | {c / 1e6:.0f} M MACs | {s} |" for n, p, c, s in rows]
    md.append(f"\nDecoder / encoder parameters: {100 * pd / pe:.1f}%. "
              f"Pretraining encoder cost vs running it on all 64 tokens: {full / enc:.1f}x cheaper.")
    text = "\n".join(md)
    print(text)
    if a.md:
        os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
        with open(a.md, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
