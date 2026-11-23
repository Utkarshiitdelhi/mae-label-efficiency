#!/usr/bin/env bash
# Whole pipeline in miniature (minutes on CPU, ~1 min on a T4): pretrain x3 ratios ->
# fine-tune both arms with LR sweep + seed selection -> probes -> figures.
# Proves the plumbing before any GPU-hour is spent. Numbers are meaningless.
#   bash scripts/smoke_test.sh [DATA_ROOT]
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-./data}
S=./smoke
rm -rf "$S"
R=$S/results
PT="--data $DATA --results $R --set epochs=2 warmup_epochs=1 train_subset=2048 batch_size=256 recon_every=1 encoder_save_every=1"
FT="--data $DATA --results $R --max-steps 12 --set num_evals=3 test_subset=1000 num_workers=2"

for m in 75 50 90; do
  python scripts/pretrain.py --config configs/pretrain_m$m.yaml --out $S/ckpt/m$m $PT
done
# simulate a killed Kaggle session: resume m75 and extend it by one epoch
python scripts/pretrain.py --config configs/pretrain_m75.yaml --out $S/ckpt/m75 $PT epochs=3

E75=$S/ckpt/m75/encoder_m75_ep3.pt
for frac in 1 10; do
  python scripts/finetune.py --arm scratch --fraction $frac --seeds 0 --lrs 5e-4 2e-3 $FT
  python scripts/finetune.py --arm mae --fraction $frac --seeds 0 --lrs 5e-4 2e-3 \
      --encoder $E75 --encoder-name m75_ep400 $FT
done
python scripts/finetune.py --arm scratch --fraction 1 --seeds 1 2 --lr-from-sweep $FT
python scripts/finetune.py --arm mae --fraction 1 --seeds 1 2 --lr-from-sweep \
    --encoder $E75 --encoder-name m75_ep400 $FT
for m in 50 90; do
  python scripts/finetune.py --arm mae --fraction 1 --seeds 0 1 2 --lr-like m75_ep400 \
      --encoder $S/ckpt/m$m/encoder_m${m}_ep2.pt --encoder-name m${m}_ep200 $FT
done
python scripts/finetune.py --arm mae --fraction 1 --seeds 0 1 2 --lr-like m75_ep400 \
    --encoder $S/ckpt/m75/encoder_m75_ep2.pt --encoder-name m75_ep200 $FT

python scripts/probe.py --data $DATA --results $R --features $S/features --random \
    --encoder m75_ep400=$E75 --encoder m75_ep200=$S/ckpt/m75/encoder_m75_ep2.pt \
    --encoder m50_ep200=$S/ckpt/m50/encoder_m50_ep2.pt --encoder m90_ep200=$S/ckpt/m90/encoder_m90_ep2.pt \
    --set epochs=5 "subsets=[[1,0],[1,1],[1,2],[10,0]]" test_subset=1000

python scripts/make_figures.py --runs $R/runs.csv --out $R --tag " [SMOKE TEST]"
echo "SMOKE TEST OK -> $R"
