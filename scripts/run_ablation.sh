#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"

echo "============================================================"
echo "LP-SSM-EEG: Ablation Experiments"
echo "============================================================"

# Ablation 1: training mode (key mechanism claim)
echo "[ABLATION 1] Local vs Global training mode..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode global \
    --log-dir logs/train/ablation_lp_ssm_global/ || true

$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --log-dir logs/train/ablation_lp_ssm_local/ || true

# Ablation 2: modulator components
echo "[ABLATION 2] Modulator components..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --override model.local_modulator.freqband_consistency=false \
    --override model.local_modulator.cross_channel_coherence=false \
    --override model.local_modulator.event_confidence=false \
    --log-dir logs/train/ablation_no_modulator/ || true

$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --override model.local_modulator.freqband_consistency=true \
    --override model.local_modulator.cross_channel_coherence=false \
    --override model.local_modulator.event_confidence=false \
    --log-dir logs/train/ablation_freqband_only/ || true

# Ablation 3: local objectives
echo "[ABLATION 3] Local objective components..."
$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --override model.local_objectives.tf_reconstruction.enabled=true \
    --override model.local_objectives.temporal_consistency.enabled=false \
    --log-dir logs/train/ablation_tf_only/ || true

$PYTHON src/cli/main.py train \
    --model lp_ssm_eeg --dataset chbmit \
    --training-mode local \
    --override model.local_objectives.tf_reconstruction.enabled=false \
    --override model.local_objectives.temporal_consistency.enabled=true \
    --log-dir logs/train/ablation_consistency_only/ || true

# Ablation 4: context length
echo "[ABLATION 4] Context length comparison..."
for CTX in 4.0 8.0 16.0 30.0; do
    for MODEL in mamba_baseline lp_ssm_eeg; do
        $PYTHON src/cli/main.py train \
            --model $MODEL --dataset chbmit \
            --context-sec $CTX \
            --log-dir logs/train/ablation_ctx${CTX}_${MODEL}/ || true
    done
done

echo "============================================================"
echo "Ablation experiments done."
echo "============================================================"
