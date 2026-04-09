#!/usr/bin/env bash
# Extra seeds experiment: Mamba + LP-SSM-EEG (warmup=10)
# Seeds: 42 1 2 3 10 20 100 200
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"
LOG_DIR="logs/train/extra_seeds"
mkdir -p "$LOG_DIR"

SEEDS=(42 1 2 3 10 20 100 200)

for SEED in "${SEEDS[@]}"; do
    echo "===== [Mamba] seed=$SEED ====="
    $PYTHON src/cli/main.py train \
        --model mamba_baseline \
        --dataset chbmit \
        --config configs/train/amp_single_4090.yaml \
        --seed $SEED \
        --log-dir "$LOG_DIR/mamba_seed${SEED}" \
        2>&1 | tee "$LOG_DIR/mamba_seed${SEED}.log"

    echo "===== [LP-SSM warmup=10] seed=$SEED ====="
    $PYTHON src/cli/main.py train \
        --model lp_ssm_eeg \
        --dataset chbmit \
        --config configs/train/amp_single_4090.yaml \
        --seed $SEED \
        --training-mode local \
        --log-dir "$LOG_DIR/lpssm_seed${SEED}" \
        2>&1 | tee "$LOG_DIR/lpssm_seed${SEED}.log"
done

echo "===== Extra seeds done. Collecting results... ====="
$PYTHON - << 'PYEOF'
import re, glob, json
from pathlib import Path

results = {"mamba": [], "lpssm": []}
for log in sorted(glob.glob("logs/train/extra_seeds/**/*.log", recursive=True)):
    # grab last AUROC line
    text = Path(log).read_text()
    aucs = re.findall(r"val_auroc[\"'\s:=]+([0-9.]+)", text)
    if not aucs: continue
    best = max(float(a) for a in aucs)
    key = "mamba" if "mamba_seed" in log else "lpssm"
    seed_m = re.search(r"seed(\d+)", log)
    seed = int(seed_m.group(1)) if seed_m else -1
    results[key].append({"seed": seed, "auroc": best})

for key, vals in results.items():
    if vals:
        aucs = [v["auroc"] for v in vals]
        import statistics
        print(f"{key}: n={len(aucs)}, mean={statistics.mean(aucs):.4f}, std={statistics.stdev(aucs):.4f}")
        for v in sorted(vals, key=lambda x: x["seed"]):
            print(f"  seed={v['seed']}: {v['auroc']:.4f}")
PYEOF
