#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"

echo "============================================================"
echo "LP-SSM-EEG: Main Experiments"
echo "============================================================"

MODELS=(eegnet cnn_baseline transformer_baseline mamba_baseline lp_ssm_eeg)

for MODEL in "${MODELS[@]}"; do
    echo "[RUN] Training ${MODEL} on CHB-MIT (patient-wise CV)..."
    $PYTHON src/cli/main.py train \
        --model $MODEL \
        --dataset chbmit \
        --config configs/train/amp_single_4090.yaml \
        --log-dir logs/train/main_${MODEL}/ \
        2>&1 | tee logs/train/main_${MODEL}/stdout.log || \
        echo "[WARN] ${MODEL} failed. See logs."
done

echo "[RUN] LP-SSM-EEG on Sleep-EDF (transfer experiment)..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg \
    --dataset sleepedf \
    --config configs/train/amp_single_4090.yaml \
    --log-dir logs/train/main_lp_ssm_sleepedf/ \
    2>&1 | tee logs/train/main_lp_ssm_sleepedf/stdout.log || \
    echo "[WARN] Sleep-EDF experiment failed. See logs."

echo "============================================================"
echo "Main experiments done. Run 'make analysis' for figures."
echo "============================================================"
