#!/usr/bin/env bash
# Wait for both LOOCV processes to finish, then run t-SNE.
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="/root/miniconda3/envs/mamba2/bin/python"
LOG="logs/loocv_tuned/tsne_run.log"

echo "$(date '+%H:%M:%S') Waiting for LOOCV processes to finish..."
while pgrep -f "run_loocv.py" > /dev/null 2>&1; do
    lp_done=$(grep -c "LOOCV SUMMARY" logs/loocv_tuned/loocv_lp_ssm_tuned.log 2>/dev/null || echo 0)
    mb_done=$(grep -c "LOOCV SUMMARY" logs/loocv_tuned/loocv_mamba.log 2>/dev/null || echo 0)
    echo "$(date '+%H:%M:%S')  LP-SSM done=$lp_done  Mamba done=$mb_done"
    sleep 120
done

echo "$(date '+%H:%M:%S') LOOCV complete. Starting t-SNE..."
$PYTHON scripts/run_tsne.py 2>&1 | tee "$LOG"
echo "$(date '+%H:%M:%S') t-SNE complete."
