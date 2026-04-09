#!/usr/bin/env bash
# Confirm best config: lambda=0.10, warmup=20, 8 seeds
# Already have seeds 42,1,2 from hparam search — run remaining 5 seeds
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="/root/miniconda3/envs/mamba2/bin/python"
LOG_DIR="logs/train/best_config"
mkdir -p "$LOG_DIR"

SEEDS=(3 10 20 100 200)

for SEED in "${SEEDS[@]}"; do
  TAG="best_lmb0.10_wm20_seed${SEED}"
  echo "===== ${TAG} ====="
  $PYTHON src/cli/main.py train \
    --model lp_ssm_eeg \
    --dataset chbmit \
    --config configs/train/amp_single_4090.yaml \
    --training-mode local \
    --local-loss-weight 0.10 \
    --warmup-epochs 20 \
    --seed $SEED \
    --log-dir "$LOG_DIR/${TAG}" \
    2>&1 | tee "$LOG_DIR/${TAG}.log"
done

echo "===== 5 new seeds done. Now evaluating all 8 seeds (3 from hparam + 5 new) ====="

$PYTHON -u - << 'PYEOF'
import re, glob, json, statistics
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, ".")

from src.models import build_model
from src.utils.config import load_config
from src.eval.metrics_classification import compute_metrics
from src.data.dataset_chbmit import CHBMITDataset

CKPT_DIR = Path("outputs/checkpoints")
MANIFEST = "data/manifests/chbmit_manifest.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def eval_run_id(run_id):
    ckpt_path = CKPT_DIR / run_id / "best.pt"
    if not ckpt_path.exists(): return None
    model_cfg = load_config("configs/model/lp_ssm_eeg.yaml").get("architecture", {})
    model = build_model("lp_ssm_eeg", in_channels=22, n_classes=2,
                        **{k: model_cfg[k] for k in ("d_model","d_state","d_conv","expand","n_layers","dropout") if k in model_cfg},
                        training_mode="global")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state); model = model.to(DEVICE).eval()
    test_ds = CHBMITDataset(MANIFEST, split="test")
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    logits, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(DEVICE))
            logits.append((out["logits"] if isinstance(out,dict) else out).float().cpu())
            targets.append(y)
    m = compute_metrics(torch.cat(logits), torch.cat(targets))
    return round(m["val_auroc"], 4)

# Collect all 8 seeds: 3 from hparam_search, 5 from best_config
all_aurocs = []
seed_rows = []

# hparam runs (seeds 42, 1, 2)
for log in sorted(glob.glob("logs/train/hparam_search/lmb0.10_wm20_seed*.log")):
    text = Path(log).read_text()
    m = re.search(r"run_id=([a-z0-9_]+)", text)
    if not m: continue
    run_id = m.group(1)
    seed = re.search(r"seed(\d+)", Path(log).stem).group(1)
    print(f"  eval seed={seed} ({run_id}) ...", flush=True)
    a = eval_run_id(run_id)
    if a:
        all_aurocs.append(a)
        seed_rows.append((seed, a))

# new runs (seeds 3,10,20,100,200)
for log in sorted(glob.glob("logs/train/best_config/best_lmb0.10_wm20_seed*.log")):
    text = Path(log).read_text()
    m = re.search(r"run_id=([a-z0-9_]+)", text)
    if not m: continue
    run_id = m.group(1)
    seed = re.search(r"seed(\d+)", Path(log).stem).group(1)
    print(f"  eval seed={seed} ({run_id}) ...", flush=True)
    a = eval_run_id(run_id)
    if a:
        all_aurocs.append(a)
        seed_rows.append((seed, a))

print("\n══════════════════════════════════════════════════════")
print("FINAL RESULT: LP-SSM (λ=0.10, warmup=20) — 8 seeds")
print("══════════════════════════════════════════════════════")
for seed, a in sorted(seed_rows, key=lambda x: int(x[0])):
    print(f"  seed={seed:<5}  test_auroc={a:.4f}")
if len(all_aurocs) > 1:
    mean = statistics.mean(all_aurocs)
    std  = statistics.stdev(all_aurocs)
    print(f"\n  mean ± std = {mean:.4f} ± {std:.4f}  (n={len(all_aurocs)})")
    print(f"  Mamba baseline  = 0.8847 ± 0.0256  (n=8)")
    print(f"  Δ = {mean-0.8847:+.4f}")

out = {"model": "lp_ssm_eeg", "lambda": 0.10, "warmup": 20,
       "seeds": dict(seed_rows), "mean": statistics.mean(all_aurocs),
       "std": statistics.stdev(all_aurocs), "n": len(all_aurocs)}
Path("outputs/metrics/best_config_final.json").write_text(__import__("json").dumps(out, indent=2))
print("\nSaved → outputs/metrics/best_config_final.json")
PYEOF
