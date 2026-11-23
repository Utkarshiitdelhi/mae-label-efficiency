# MAE for label-efficient vision (CIFAR-10, ViT-Tiny, from scratch)

A Masked Autoencoder written from scratch in PyTorch, pretrained on the 50K CIFAR-10 training
images **without labels**, then used to measure how many labels a ViT-Tiny needs compared with a
from-scratch baseline trained under exactly the same recipe. Design rationale:
[`docs/design_doc_v1.0.md`](docs/design_doc_v1.0.md).

## Result

> **To be filled after the Kaggle runs** (`python scripts/make_figures.py` prints the numbers).
> One paragraph: MAE-75 vs scratch test accuracy at 1% / 10% / 100% labels with the paired gain,
> where MAE did **not** help, the best masking ratio by linear probe, and the gap over the
> random-init floor.

![Label efficiency](results/figures/label_efficiency.png)

| Table | File |
| --- | --- |
| Label efficiency (fine-tune, both arms, mean ± std, paired gain) | [`results/tables/label_efficiency.md`](results/tables/label_efficiency.md) |
| Masking-ratio ablation (linear probe 1/10/100% + fine-tune 1%) | [`results/tables/ablation.md`](results/tables/ablation.md) |
| Encoder vs decoder parameters and compute | [`results/tables/model_stats.md`](results/tables/model_stats.md) |

Figures: `label_efficiency.png`, `ablation.png`, `val_curves.png` (scratch-baseline plateau check),
`recon_m75.png` (plus `recon_m50.png`, `recon_m90.png`), all in `results/figures/`.

## CV bullet → evidence

| CV bullet | Evidence in this repo |
| --- | --- |
| 1. MAE from scratch: ViT encoder–decoder, 75% masking, loss on masked patches only | `src/mae/vit.py`, `src/mae/mae.py` (no timm/HF/`nn.MultiheadAttention`); `tests/test_masking.py`, `tests/test_loss.py` (changing visible-patch targets leaves the loss unchanged; zero gradient from visible predictions); `results/figures/recon_m75.png` |
| 2. ViT-Tiny pretrained on 50K unlabelled images, fine-tuned at 1/10/100% vs matched scratch | `results/figures/label_efficiency.png`, `results/tables/label_efficiency.md`, `results/runs.csv`; matching enforced in `src/mae/engine.py::finetune_run` |
| 3. Shallow decoder so semantic capacity sits in the encoder | [Architecture](#architecture-and-the-shallow-decoder-bullet-3) below, `scripts/model_stats.py`; decoder never leaves pretraining (only `encoder_*.pt` is saved for downstream use) |
| 4. Masking-ratio ablation 50/75/90 + linear probe vs random-init frozen encoder | `results/figures/ablation.png`, `results/tables/ablation.md` |

## Architecture and the shallow decoder (bullet 3)

| Component | Setting |
| --- | --- |
| Patches | 4×4 px → 64 tokens of 48 values; fixed 2D sin-cos position embedding |
| Encoder | ViT-Tiny: 12 pre-norm blocks, width 192, 3 heads, MLP 768, final LayerNorm, no class token |
| Masking | per image: uniform noise → argsort → keep first 16 of 64 (75%); encoder sees only these |
| Decoder | 192→128, mask tokens, unshuffle, pos-embed, 4 blocks × width 128 (4 heads), 128→48 |
| Loss | MSE on per-patch-normalised pixels, **masked patches only** |
| Classifier | encoder on all 64 tokens → mean-pool → LayerNorm → Linear(192, 10), same for both arms |

Measured by `python scripts/model_stats.py` (torch FLOP counter, per image):

| Part | Parameters | Compute per image | Share of pretraining compute |
| --- | --- | --- | --- |
| Encoder (16 visible tokens) | 5.35 M | 87 M MACs | 61% |
| Decoder (64 tokens) | 0.82 M | 55 M MACs | 39% |
| Encoder at fine-tune (64 tokens) | 5.35 M | 359 M MACs | n/a |

The decoder has 15% of the encoder's parameters, and dropping masked tokens makes the encoder
4.1× cheaper during pretraining. The decoder is discarded after pretraining, and everything it knows
about an image has to pass through the encoder's 16 visible-token outputs, so the understanding has
to live in the encoder. The caveat (MAE paper): a 1-block decoder makes the last encoder blocks
specialise in pixels, which hurts linear probing. So the decoder here has 4 blocks: shallow relative
to the 12-block encoder, but not minimal.

## Reproduce

Everything runs from scripts. The Kaggle notebooks only fetch the code and call `scripts/run_block.sh`.

```bash
pip install -r requirements.txt
python scripts/make_splits.py --data ./data      # only if you want to regenerate; splits/ is committed
python -m pytest -q tests                        # 25 tests; the overfit test is ~3 min on CPU
bash scripts/smoke_test.sh ./data                # whole pipeline in miniature (numbers meaningless)
bash scripts/run_block.sh <block> [...]          # the real run matrix, one block per session
```

### Run plan on a Kaggle T4

Upload this repo as a Kaggle Dataset (or push to GitHub and set `REPO_URL` in the notebook). Turn on
**GPU T4** and **Internet** (for the CIFAR-10 download). Use **Save & Run All** for long blocks.
Attach each session's output as an input to the next one: checkpoints and earlier `runs.csv` files are
found and merged automatically, so LR choices carry across sessions.

| # | Notebook | `BLOCKS` | What it does | Gate before moving on |
| --- | --- | --- | --- | --- |
| 1 | `kaggle_pretrain` | `test timing lrcheck` | tests + smoke pipeline on GPU, 5-epoch timing, 20-epoch LR check (3e-4 vs 1e-3) | all tests pass; set `LR_PT=1e-3` only if it has lower loss |
| 2 | `kaggle_pretrain` | `pretrain_m75` | MAE-75, 400 epochs (encoders saved every 50 epochs) | recon grids look plausible |
| 3 | `kaggle_finetune` | `probe_ep100` | epoch-100 encoder probe vs random init | MAE clearly above random |
| 4 | `kaggle_finetune` | `ft_low` | 1% + 10% LR sweeps (both arms, seed 0), 1% seeds 1–2 at chosen LRs | |
| 5 | `kaggle_pretrain` | `pretrain_abl` (+ `pretrain_m75e200`, optional) | MAE-50 and MAE-90, 200 epochs | |
| 6 | `kaggle_finetune` | `ft_100` | 100% LR sweeps (both arms) | scratch val curve plateaued |
| 7 | `kaggle_finetune` | `ft_ablation probes figures` | ablation fine-tunes, all linear probes, figures + tables | every run in `runs.csv` |

Then download `results/` into this repo, and if you have several sessions' CSVs, run
`python scripts/merge_runs.py --out results/runs.csv <csvs...>` and
`python scripts/make_figures.py`.

### Protocol (what makes "matched" defensible)

* **Splits** are class-balanced and committed (`splits/frac{1,10,100}_seed*.json`). Validation is
  carved from the labelled fraction (1% = 400 train + 100 val), and both arms load the same files.
* **Fine-tuning** is the same for both arms: AdamW (0.9, 0.999), wd 0.05, 5% warm-up + cosine,
  stochastic depth 0.1, label smoothing 0.1, RandAugment(2, 9), fp16, the same step budget
  (3000 / 6000 / 17600 steps at batch 64 / 128 / 256) and **its own** LR sweep {5e-4, 1e-3, 2e-3}
  on seed 0. Layer-wise LR decay is deliberately left out, because it would make the arms differ in
  two things instead of one.
* **Selection**: the best-val checkpoint is tested once on the 10K test set. LRs are chosen on val,
  never on test. Reported as mean ± std over seeds (3 at 1%, 1 at 10% and 100%) plus the paired gain.
* **Linear probe**: frozen encoder on all 64 tokens, mean-pooled, cached; BatchNorm (no affine) +
  Linear, AdamW, 100 epochs, LR {1e-3, 1e-2, 1e-1} chosen on val per subset. The random-init frozen
  encoder is the floor.

## Deviations from design doc v1.0 (and why)

1. **Ablation fine-tune adds MAE-75@200 at 1% (3 runs, ~0.1 GPU-h).** The doc's run matrix lists only
   MAE-50/90 there. Without the matching MAE-75 runs, the fine-tune column would compare 200-epoch
   encoders with a 400-epoch one.
2. **Optional `pretrain_m75e200`.** The epoch-200 checkpoint of the 400-epoch run is mid-cosine (its LR
   has not annealed), while MAE-50/90 finish their own 200-epoch schedules. That is equal epochs but not
   an equal schedule, and it slightly favours 50/90. If you run this block, the ablation automatically
   uses it for the 75% row; otherwise it uses `m75_ep200` and the README should say so.
3. **LR check uses 2 warm-up epochs.** With the main recipe's 20, a 20-epoch check would be pure warm-up.
4. **Pretraining augmentation runs on the GPU.** RandomResizedCrop(0.2–1.0) + flip are implemented as one
   batched affine `grid_sample` (bilinear) over the uint8 dataset kept on the GPU, so a T4 isn't bottlenecked
   on Kaggle's 4 CPU cores. Fine-tuning keeps torchvision RandAugment in DataLoader workers.
5. **Overfit test** uses a fixed mask, real images and 500 steps, and asserts a >10× loss drop.
   Memorising per-pixel Gaussian noise plateaus at the mean, so noise is a poor test input.
6. **Checkpointing**: the full resumable state (`ckpt_latest.pt`) is overwritten every 25 epochs, and
   encoder-only weights are kept every 50 epochs (ep100 for the gate, ep200 for the ablation, ep400 main).
7. **Extra files** beyond the doc's layout: `src/mae/engine.py` (training loops, shared by scripts and tests),
   `scripts/run_block.sh` (the run matrix), `smoke_test.sh`, `merge_runs.py`, `model_stats.py`,
   `tests/test_splits.py`.

**Reconstruction grids:** the model predicts *normalised* pixels, so to draw a masked patch, its prediction
is rescaled with that patch's true mean and std. Brightness and colour cast in masked patches therefore come
from the original, and only the structure comes from the model.

## Layout

```
configs/    pretrain_m{50,75,90}.yaml, pretrain_m75_e200.yaml (optional), finetune.yaml, probe.yaml
splits/     committed label subsets
src/mae/    pos_embed.py, vit.py, mae.py, data.py, engine.py, utils.py
scripts/    make_splits, pretrain, finetune, probe, make_figures, merge_runs, model_stats, run_block.sh, smoke_test.sh
tests/      test_shapes, test_masking, test_loss, test_resume, test_splits
notebooks/  kaggle_pretrain.ipynb, kaggle_finetune.ipynb (thin wrappers)
results/    runs.csv, logs/, figures/, tables/
```

Reproducibility: seeds for Python, NumPy and torch come from the config; every `runs.csv` row records the
git commit. `cudnn.benchmark` is on, so GPU runs are not bit-exact, and seed variance is reported instead.
CPU resume is bit-exact (`tests/test_resume.py`).

## References

He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022 ·
Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021 ·
Touvron et al., *Training data-efficient image transformers* (DeiT), ICML 2021.
