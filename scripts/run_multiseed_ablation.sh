#!/usr/bin/env bash
# Multi-seed ablation: run LP-SSM global / local-no-mod / local-mod
# with 3 additional seeds (1, 2, 3) to show ablation rows are seed-robust
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=/root/miniconda3/envs/mamba2/bin/python
LOG=logs/train/multiseed_ablation

# Pre-create all log directories
for SEED in 1 2 3; do
  mkdir -p "${LOG}/global_seed${SEED}"
  mkdir -p "${LOG}/local_nomod_seed${SEED}"
  mkdir -p "${LOG}/local_mod_seed${SEED}"
done

echo "=== Multi-seed ablation (seeds 1, 2, 3) ==="

for SEED in 1 2 3; do
  echo "[seed=$SEED] LP-SSM global (training_mode=global)..."
  $PYTHON -m src.cli.main train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode global \
    --seed $SEED \
    --log-dir ${LOG}/global_seed${SEED}/ \
    >> ${LOG}/global_seed${SEED}/stdout.log 2>&1
  echo "[seed=$SEED] global done."

  echo "[seed=$SEED] LP-SSM local no-mod..."
  $PYTHON -m src.cli.main train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --no-modulator \
    --warmup-epochs 20 --local-loss-weight 0.10 \
    --seed $SEED \
    --log-dir ${LOG}/local_nomod_seed${SEED}/ \
    >> ${LOG}/local_nomod_seed${SEED}/stdout.log 2>&1
  echo "[seed=$SEED] local_nomod done."

  echo "[seed=$SEED] LP-SSM local+mod (tuned)..."
  $PYTHON -m src.cli.main train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --mod-features ictal_ratio \
    --warmup-epochs 20 --local-loss-weight 0.10 \
    --seed $SEED \
    --log-dir ${LOG}/local_mod_seed${SEED}/ \
    >> ${LOG}/local_mod_seed${SEED}/stdout.log 2>&1
  echo "[seed=$SEED] local_mod done."
done

echo "=== Multi-seed ablation done ==="
