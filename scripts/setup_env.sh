#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "LP-SSM-EEG: Environment Setup"
echo "============================================================"

CONDA_ENV="mamba2"

if conda env list | grep -q "^${CONDA_ENV}"; then
    echo "[INFO] Conda environment '${CONDA_ENV}' already exists."
    echo "[INFO] Activating and installing dependencies..."
    conda run -n $CONDA_ENV pip install -r requirements.txt --quiet
else
    echo "[INFO] Creating conda environment '${CONDA_ENV}'..."
    conda create -n $CONDA_ENV python=3.10 -y
    conda run -n $CONDA_ENV pip install -r requirements.txt --quiet
fi

echo "[INFO] Running environment check..."
conda run -n $CONDA_ENV python src/cli/main.py system-info \
    --log-dir logs/setup/ \
    2>&1 | tee logs/setup/setup_$(date +%Y%m%d_%H%M%S).log || true

echo "============================================================"
echo "Setup complete. Activate with: conda activate ${CONDA_ENV}"
echo "============================================================"
