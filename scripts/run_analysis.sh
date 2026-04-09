#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"

echo "============================================================"
echo "LP-SSM-EEG: Analysis and Figure Generation"
echo "============================================================"

echo "[1] Summarizing all experiment results..."
$PYTHON src/analysis/summarize_results.py \
    --metrics-dir outputs/metrics/ \
    --output-dir outputs/tables/

echo "[2] Plotting learning curves..."
$PYTHON src/analysis/plot_learning_curves.py \
    --logs-dir logs/train/ \
    --output-dir outputs/figures/

echo "[3] Plotting ROC/PR curves..."
$PYTHON src/analysis/plot_roc_pr.py \
    --predictions-dir outputs/predictions/ \
    --output-dir outputs/figures/

echo "[4] Plotting confusion matrices..."
$PYTHON src/analysis/plot_confusion_matrix.py \
    --predictions-dir outputs/predictions/ \
    --output-dir outputs/figures/

echo "============================================================"
echo "Analysis complete. Figures in outputs/figures/"
echo "============================================================"
