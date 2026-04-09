#!/bin/bash
# Auto-run preprocessing + zero-shot eval when SIENA download completes.
# Usage: nohup bash scripts/siena_auto_eval.sh > logs/siena_auto_eval.log 2>&1 &

cd /root/autodl-tmp/medical
TOTAL_EDF=41
LOG="logs/siena_auto_eval.log"

echo "[$(date)] Starting auto-eval watcher. Waiting for $TOTAL_EDF EDF files..." | tee -a "$LOG"

while true; do
    CURRENT=$(find data/raw/siena -name "*.edf" | wc -l)
    echo "[$(date)] EDF files: $CURRENT/$TOTAL_EDF" | tee -a "$LOG"
    
    if [ "$CURRENT" -ge "$TOTAL_EDF" ]; then
        echo "[$(date)] Download complete! Running preprocessing..." | tee -a "$LOG"
        break
    fi
    sleep 300   # check every 5 min
done

# Re-run preprocessing with all subjects (skip-download since aria2c handles it)
conda run -n mamba2 python -u scripts/download_siena.py --skip-download \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Preprocessing done. Running zero-shot eval..." | tee -a "$LOG"

conda run -n mamba2 python -u scripts/eval_siena_zeroshot.py \
    2>&1 | tee -a "$LOG"

echo "[$(date)] Done. Check outputs/metrics/siena_zeroshot.json" | tee -a "$LOG"

# Quick git commit of results
git add -A
git commit -m "SIENA full dataset: auto-eval results after download" 2>&1 | tee -a "$LOG"
