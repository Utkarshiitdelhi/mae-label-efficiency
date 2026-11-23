#!/usr/bin/env bash
# The design doc's run matrix, one named block per Kaggle session.
#
#   bash scripts/run_block.sh <block> [<block> ...]
#
# Blocks, in order (see README "Run plan"):
#   test          unit tests (fast + one-batch overfit) and the miniature smoke pipeline
#   timing        5-epoch MAE-75 timing run (Milestone 1; update the run matrix with it)
#   lrcheck       20-epoch pretrain at 3e-4 and 1e-3; keep the lower reconstruction loss
#   pretrain_m75  main 400-epoch pretrain (keeps encoders every 50 epochs, incl. ep100/ep200)
#   pretrain_abl  MAE-50 and MAE-90, 200 epochs each
#   pretrain_m75e200  OPTIONAL: MAE-75 on its own 200-epoch schedule (fully matched ablation)
#   ft_low        1% and 10% LR sweeps (both arms, seed 0) + 1% seeds 1,2 at the chosen LRs
#   ft_100        100% LR sweeps (both arms, seed 0)
#   ft_ablation   MAE-50 / MAE-75@200 / MAE-90 at 1%, 3 seeds, LR chosen for MAE-75 at 1%
#   probes        all linear probes (random init + every MAE encoder x 5 subsets x 3 LRs)
#   probe_ep100   Milestone-2 gate: epoch-100 MAE-75 probe vs random init
#   figures       figures + tables from runs.csv
#
# Environment (defaults work locally and on Kaggle):
#   DATA        CIFAR-10 root            (default ./data; downloaded if missing)
#   RESULTS     results dir              (default ./results)
#   CKPT        where pretrains write    (default ./checkpoints)
#   SEARCH      where to look for encoders / earlier runs.csv
#               (default "./checkpoints ./results /kaggle/input")
#   LR_PT       pretrain peak LR override chosen by lrcheck (default: config, 3e-4)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-./data}
RESULTS=${RESULTS:-./results}
CKPT=${CKPT:-./checkpoints}
SEARCH=${SEARCH:-"./checkpoints ./results /kaggle/input"}
LRSET=${LR_PT:+lr=$LR_PT}
mkdir -p "$RESULTS" "$CKPT"

enc() {   # path of encoder_<name>.pt anywhere under SEARCH
  local p
  p=$(find $SEARCH -name "encoder_$1.pt" -not -path "*/smoke/*" 2>/dev/null | head -1 || true)
  [ -n "$p" ] || { echo "missing encoder_$1.pt under: $SEARCH" >&2; exit 1; }
  echo "$p"
}

merge_previous() {   # bring in runs.csv from attached earlier sessions (needed for LR choice)
  local found
  # never merge smoke-test, timing, LR-check or gate runs into the real results
  found=$(find $SEARCH -name runs.csv -not -path "$RESULTS/*" -not -path "*/smoke/*" \
          -not -path "*/timing/*" -not -path "*/lrcheck/*" -not -path "*/gate/*" 2>/dev/null | tr '\n' ' ' || true)
  if [ -n "$found" ]; then
    python scripts/merge_runs.py --out "$RESULTS/runs.csv" $found "$RESULTS/runs.csv"
  fi
}

seed_resume() {   # copy an attached checkpoint so a stopped pretrain resumes
  local run=$1 latest
  [ -f "$CKPT/$run/ckpt_latest.pt" ] && return 0
  latest=$(find /kaggle/input -path "*$run/ckpt_latest.pt" 2>/dev/null | head -1 || true)
  if [ -n "$latest" ]; then
    mkdir -p "$CKPT/$run"
    cp "$(dirname "$latest")"/*.pt "$CKPT/$run/"
    [ -d "$(dirname "$latest")/recon" ] && cp -r "$(dirname "$latest")/recon" "$CKPT/$run/"
    echo "resuming $run from attached $latest"
  fi
}

pretrain() {   # pretrain <ratio> [extra --set args...]
  local cfg=$1; shift
  local run; run=$(python -c "import yaml;print(yaml.safe_load(open('configs/pretrain_$cfg.yaml'))['run_name'])")
  seed_resume "$run"
  python scripts/pretrain.py --config configs/pretrain_$cfg.yaml --data "$DATA" \
      --results "$RESULTS" --out "$CKPT/$run" --set $LRSET "$@"
  mkdir -p "$RESULTS/figures"
  [ -d "$CKPT/$run/recon" ] && cp "$(ls -v "$CKPT/$run"/recon/*.png | tail -1)" "$RESULTS/figures/recon_$run.png" || true
}

FT="--data $DATA --results $RESULTS"

for block in "$@"; do
  echo "================ $block ================"
  case $block in
    test)
      python -c "import sys; sys.path.insert(0, 'src'); from mae.data import load_cifar; load_cifar('$DATA', True); load_cifar('$DATA', False)"
      MAE_DATA="$DATA" python -m pytest -q tests
      bash scripts/smoke_test.sh "$DATA" ;;
    timing)
      python scripts/pretrain.py --config configs/pretrain_m75.yaml --data "$DATA" --results "$RESULTS/timing" \
          --out "$CKPT/timing" --no-resume --set epochs=5 warmup_epochs=1 recon_every=0 run_name=m75_timing
      python - "$RESULTS/timing/logs/pt_m75_timing.csv" <<'EOF'
import csv, sys
r = list(csv.DictReader(open(sys.argv[1])))
t = sorted(float(x["epoch_time_s"]) for x in r[1:])       # skip epoch 1 (warm-up / cudnn autotune)
s = t[len(t) // 2]
print(f"median epoch {s:.1f}s -> 400 epochs ~ {400 * s / 3600:.2f} h, 200 epochs ~ {200 * s / 3600:.2f} h; "
      f"peak GPU mem {r[-1]['gpu_mem_mb']} MB")
EOF
      ;;
    lrcheck)
      for lr in 3e-4 1e-3; do
        python scripts/pretrain.py --config configs/pretrain_m75.yaml --data "$DATA" --results "$RESULTS/lrcheck" \
            --out "$CKPT/lrcheck_$lr" --no-resume \
            --set epochs=20 warmup_epochs=2 recon_every=20 encoder_save_every=1000 lr=$lr run_name=m75_lr$lr
      done
      grep -h "pretrain" "$RESULTS/lrcheck/runs.csv" | cut -d, -f1,13 | sed 's/^/run,final_loss: /'
      echo "-> keep the LR with the lower final_loss; if 1e-3 wins, export LR_PT=1e-3 for all pretrains" ;;
    pretrain_m75)     pretrain m75 ;;
    pretrain_abl)     pretrain m50; pretrain m90 ;;
    pretrain_m75e200) pretrain m75_e200 ;;
    ft_low)
      merge_previous
      E=$(enc m75_ep400)
      for frac in 1 10; do
        python scripts/finetune.py --arm scratch --fraction $frac --seeds 0 $FT
        python scripts/finetune.py --arm mae --fraction $frac --seeds 0 --encoder "$E" --encoder-name m75_ep400 $FT
      done
      python scripts/finetune.py --arm scratch --fraction 1 --seeds 1 2 --lr-from-sweep $FT
      python scripts/finetune.py --arm mae --fraction 1 --seeds 1 2 --lr-from-sweep --encoder "$E" --encoder-name m75_ep400 $FT ;;
    ft_100)
      merge_previous
      E=$(enc m75_ep400)
      python scripts/finetune.py --arm scratch --fraction 100 --seeds 0 $FT ${LRS_100:+--lrs $LRS_100}
      python scripts/finetune.py --arm mae --fraction 100 --seeds 0 --encoder "$E" --encoder-name m75_ep400 $FT ${LRS_100:+--lrs $LRS_100} ;;
    ft_ablation)
      merge_previous
      M75=m75_ep200; find $SEARCH -name encoder_m75e200_ep200.pt 2>/dev/null | grep -q . && M75=m75e200_ep200
      for name in m50_ep200 $M75 m90_ep200; do
        python scripts/finetune.py --arm mae --fraction 1 --seeds 0 1 2 --lr-like m75_ep400 \
            --encoder "$(enc $name)" --encoder-name $name $FT
      done ;;
    probes)
      merge_previous
      ARGS="--random --encoder m75_ep400=$(enc m75_ep400) --encoder m75_ep200=$(enc m75_ep200)"
      ARGS="$ARGS --encoder m50_ep200=$(enc m50_ep200) --encoder m90_ep200=$(enc m90_ep200)"
      if find $SEARCH -name encoder_m75e200_ep200.pt 2>/dev/null | grep -q .; then
        ARGS="$ARGS --encoder m75e200_ep200=$(enc m75e200_ep200)"
      fi
      python scripts/probe.py --data "$DATA" --results "$RESULTS" --features ./features $ARGS ;;
    probe_ep100)
      python scripts/probe.py --data "$DATA" --results "$RESULTS/gate" --features ./features_gate \
          --random --encoder m75_ep100=$(enc m75_ep100) --set "subsets=[[100,0]]" ;;
    figures)
      merge_previous
      python scripts/make_figures.py --runs "$RESULTS/runs.csv" --out "$RESULTS" ;;
    *) echo "unknown block: $block" >&2; exit 1 ;;
  esac
done
