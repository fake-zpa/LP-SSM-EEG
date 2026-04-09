#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"

echo "============================================================"
echo "LP-SSM-EEG: Smoke Test Pipeline"
echo "============================================================"

echo "[STEP 1] System check..."
$PYTHON src/cli/main.py system-info --log-dir logs/setup/

echo "[STEP 2] Download CHB-MIT chb01 (5 files max)..."
$PYTHON src/cli/main.py download \
    --dataset chbmit \
    --subjects chb01 \
    --max-files 5 \
    --log-dir logs/download/

echo "[STEP 3] Preprocess chb01 (smoke)..."
$PYTHON src/cli/main.py preprocess \
    --dataset chbmit \
    --subjects chb01 \
    --smoke \
    --log-dir logs/preprocess/

echo "[STEP 4] Train EEGNet smoke (3 epochs)..."
$PYTHON src/cli/main.py train \
    --model eegnet \
    --dataset chbmit \
    --subjects chb01 \
    --max-epochs 3 \
    --smoke \
    --log-dir logs/train/smoke_eegnet/

echo "[STEP 5] Train Mamba baseline smoke (3 epochs)..."
$PYTHON src/cli/main.py train \
    --model mamba_baseline \
    --dataset chbmit \
    --subjects chb01 \
    --max-epochs 3 \
    --smoke \
    --log-dir logs/train/smoke_mamba/

echo "[STEP 6] Train LP-SSM-EEG smoke (3 epochs, local mode)..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg \
    --dataset chbmit \
    --subjects chb01 \
    --max-epochs 3 \
    --smoke \
    --training-mode local \
    --log-dir logs/train/smoke_lp_ssm/

echo "[STEP 7] Train LP-SSM-EEG smoke (3 epochs, global mode — ablation)..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg \
    --dataset chbmit \
    --subjects chb01 \
    --max-epochs 3 \
    --smoke \
    --training-mode global \
    --log-dir logs/train/smoke_lp_ssm_global/

echo "============================================================"
echo "Smoke test complete. Check logs/train/ for results."
echo "============================================================"
