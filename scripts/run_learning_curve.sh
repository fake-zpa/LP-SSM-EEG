#!/usr/bin/env bash
# Learning curve: n_train_patients = 1,2,3,4 x {mamba, lp_ssm} x seeds {42,1,2}
# Fixed test: chb04+chb10  Fixed val: chb05
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="conda run -n mamba2 python"
LOG_DIR="logs/train/learning_curve"
mkdir -p "$LOG_DIR"

# Patient sets (cumulative train subjects + fixed val=chb05 + fixed test=chb04,chb10)
# The manifest already encodes split assignments; subjects= just filters which rows to load.
# Including val/test subjects ensures those splits are non-empty.
declare -A TRAIN_SETS
TRAIN_SETS[1]="chb01,chb05,chb04,chb10"
TRAIN_SETS[2]="chb01,chb02,chb05,chb04,chb10"
TRAIN_SETS[3]="chb01,chb02,chb03,chb05,chb04,chb10"
TRAIN_SETS[4]="chb01,chb02,chb03,chb06,chb05,chb04,chb10"

SEEDS=(42 1 2)

for N in 1 2 3 4; do
    SUBJECTS="${TRAIN_SETS[$N]}"
    for SEED in "${SEEDS[@]}"; do

        echo "===== [Mamba] n_train=$N subjects=$SUBJECTS seed=$SEED ====="
        $PYTHON src/cli/main.py train \
            --model mamba_baseline \
            --dataset chbmit \
            --config configs/train/amp_single_4090.yaml \
            --subjects "$SUBJECTS" \
            --seed $SEED \
            --log-dir "$LOG_DIR/mamba_n${N}_seed${SEED}" \
            2>&1 | tee "$LOG_DIR/mamba_n${N}_seed${SEED}.log"

        echo "===== [LP-SSM warmup=10] n_train=$N subjects=$SUBJECTS seed=$SEED ====="
        $PYTHON src/cli/main.py train \
            --model lp_ssm_eeg \
            --dataset chbmit \
            --config configs/train/amp_single_4090.yaml \
            --subjects "$SUBJECTS" \
            --seed $SEED \
            --training-mode local \
            --log-dir "$LOG_DIR/lpssm_n${N}_seed${SEED}" \
            2>&1 | tee "$LOG_DIR/lpssm_n${N}_seed${SEED}.log"

    done
done

echo "===== Learning curve done. Collecting results... ====="
$PYTHON - << 'PYEOF'
import re, glob, statistics
from pathlib import Path
from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))  # data[model][n] -> [auroc]

for log in sorted(glob.glob("logs/train/learning_curve/**/*.log", recursive=True)):
    text = Path(log).read_text()
    aucs = re.findall(r"val_auroc[\"'\s:=]+([0-9.]+)", text)
    if not aucs:
        continue
    best = max(float(a) for a in aucs)
    name = Path(log).stem  # e.g. mamba_n2_seed42
    m = re.match(r"(mamba|lpssm)_n(\d+)_seed\d+", name)
    if not m:
        continue
    model, n = m.group(1), int(m.group(2))
    data[model][n].append(best)

print(f"{'N':>3} | {'Mamba mean±std':>18} | {'LP-SSM mean±std':>18} | {'Delta':>8}")
print("-" * 60)
for n in [1, 2, 3, 4]:
    m_vals = data["mamba"].get(n, [])
    l_vals = data["lpssm"].get(n, [])
    m_str = f"{statistics.mean(m_vals):.4f}±{statistics.stdev(m_vals):.4f}" if len(m_vals) > 1 else (f"{m_vals[0]:.4f}" if m_vals else "N/A")
    l_str = f"{statistics.mean(l_vals):.4f}±{statistics.stdev(l_vals):.4f}" if len(l_vals) > 1 else (f"{l_vals[0]:.4f}" if l_vals else "N/A")
    if m_vals and l_vals:
        delta = statistics.mean(l_vals) - statistics.mean(m_vals)
        d_str = f"{delta:+.4f}"
    else:
        d_str = "N/A"
    print(f"{n:>3} | {m_str:>18} | {l_str:>18} | {d_str:>8}")
PYEOF
