#!/usr/bin/env bash
# Hyperparameter search: lambda x warmup_epochs for LP-SSM-EEG
# Grid: lambda=[0.05,0.10,0.20] x warmup=[5,15,20] x seeds=[42,1,2]
# = 27 runs x ~9 min = ~4 hours
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="/root/miniconda3/envs/mamba2/bin/python"
LOG_DIR="logs/train/hparam_search"
mkdir -p "$LOG_DIR"

LAMBDAS=(0.05 0.10 0.20)
WARMUPS=(5 15 20)
SEEDS=(42 1 2)

for LMB in "${LAMBDAS[@]}"; do
  for WM in "${WARMUPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      TAG="lmb${LMB}_wm${WM}_seed${SEED}"
      echo "===== ${TAG} ====="
      $PYTHON src/cli/main.py train \
        --model lp_ssm_eeg \
        --dataset chbmit \
        --config configs/train/amp_single_4090.yaml \
        --training-mode local \
        --local-loss-weight $LMB \
        --warmup-epochs $WM \
        --seed $SEED \
        --log-dir "$LOG_DIR/${TAG}" \
        2>&1 | tee "$LOG_DIR/${TAG}.log"
    done
  done
done

echo "===== Hyperparameter search done. Collecting results... ====="
$PYTHON - << 'PYEOF'
import re, glob, statistics
from pathlib import Path
from collections import defaultdict

data = defaultdict(list)  # key=(lambda,warmup) -> [auroc]

for log in sorted(glob.glob("logs/train/hparam_search/*.log")):
    text = Path(log).read_text()
    m = re.search(r"Training complete.*?'best_val_metric': ([0-9.]+)", text)
    if not m: continue
    name = Path(log).stem  # lmb0.10_wm15_seed42
    pm = re.match(r"lmb([0-9.]+)_wm(\d+)_seed\d+", name)
    if not pm: continue
    lmb, wm = pm.group(1), int(pm.group(2))
    data[(lmb, wm)].append(float(m.group(1)))

print(f"\n{'λ':>6}  {'warmup':>6}  {'n':>3}  {'val mean':>9}  {'val std':>8}")
print("-" * 42)
for (lmb, wm) in sorted(data.keys()):
    vals = data[(lmb, wm)]
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {lmb:>5}  {wm:>6}  {len(vals):>3}  {mean:>9.4f}  {std:>8.4f}")
PYEOF
