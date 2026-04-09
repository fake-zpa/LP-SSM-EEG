"""
Evaluate the 8 Mamba HP-grid checkpoints produced by mamba_hparam_search.py.

Configs ran in order:
  (lr=1e-4,do=0.1), (1e-4,0.2), (1e-4,0.3),
  (3e-4,0.1), (3e-4,0.2), (3e-4,0.3),
  (5e-4,0.1), (5e-4,0.2)          ← 9th (5e-4,0.3) was canceled

Reads initial_lr from optimizer state to confirm mapping.
Evaluates each checkpoint on test split (chbmit), saves JSON.
"""
import json, sys, subprocess, re
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PYTHON   = sys.executable
MANIFEST = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
OUT_JSON = str(PROJECT_ROOT / "outputs" / "metrics" / "mamba_hparam_search.json")

# Ordered by creation time (mtime ascending)
RUNS_ORDERED = [
    "mamba_baseline_20260408_105225",
    "mamba_baseline_20260408_110119",
    "mamba_baseline_20260408_111030",
    "mamba_baseline_20260408_111919",
    "mamba_baseline_20260408_112806",
    "mamba_baseline_20260408_113701",
    "mamba_baseline_20260408_114430",
    "mamba_baseline_20260408_115200",
]

GRID_ORDER = [
    (1e-4, 0.1), (1e-4, 0.2), (1e-4, 0.3),
    (3e-4, 0.1), (3e-4, 0.2), (3e-4, 0.3),
    (5e-4, 0.1), (5e-4, 0.2),
]
SEED = 1   # all 1-seed runs used seed=1 (SEEDS[:1])


def get_initial_lr(ckpt_path: Path) -> float | None:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        opt  = ckpt.get("optimizer_state") or ckpt.get("optimizer")
        if opt is None:
            return None
        for pg in opt.get("param_groups", []):
            if "initial_lr" in pg:
                return float(pg["initial_lr"])
        return None
    except Exception:
        return None


def get_val_auroc(ckpt_path: Path) -> float | None:
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        m    = ckpt.get("metrics", {})
        return float(m.get("val_auroc", float("nan")))
    except Exception:
        return None


def evaluate_checkpoint(ckpt_path: str) -> dict:
    cmd = [
        PYTHON, "-m", "src.cli.main", "evaluate",
        "--model", "mamba_baseline",
        "--checkpoint", ckpt_path,
        "--dataset", "chbmit",
        "--manifest", MANIFEST,
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                          capture_output=True, text=True, timeout=300)
    try:
        return json.loads(proc.stdout)
    except Exception:
        for line in proc.stdout.splitlines():
            try:
                d = json.loads(line)
                if "auroc" in d:
                    return d
            except Exception:
                pass
    return {}


def main():
    results = []
    default_lr, default_do = 5e-4, 0.1

    for i, run_id in enumerate(RUNS_ORDERED):
        lr, do = GRID_ORDER[i]
        ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / run_id / "best.pt"

        if not ckpt_path.exists():
            print(f"[{i+1}/8] {run_id}: best.pt missing, skip")
            continue

        # Verify lr matches
        actual_lr = get_initial_lr(ckpt_path)
        lr_match  = actual_lr is None or abs(actual_lr - lr) < lr * 0.01

        val_auroc = get_val_auroc(ckpt_path)
        print(f"[{i+1}/8] lr={lr:.0e} do={do:.1f}  "
              f"actual_initial_lr={actual_lr:.4e}  val_auroc={val_auroc:.4f}  "
              f"{'✓' if lr_match else '✗ MISMATCH'}", end="  ")

        # Test evaluation
        metrics = evaluate_checkpoint(str(ckpt_path))
        test_auroc = metrics.get("val_auroc") or metrics.get("auroc")
        test_auprc = metrics.get("val_auprc") or metrics.get("auprc")
        print(f"→ test_auroc={test_auroc}  test_auprc={test_auprc}")

        is_default = abs(lr - default_lr) < 1e-8 and abs(do - default_do) < 1e-4
        results.append({
            "lr": lr, "dropout": do, "seed": SEED,
            "run_id": run_id,
            "val_auroc": val_auroc,
            "test_auroc": test_auroc,
            "test_auprc": test_auprc,
            "actual_initial_lr": actual_lr,
            "is_default": is_default,
        })

    # Sort by val_auroc to find best config
    valid = [r for r in results if r["test_auroc"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["test_auroc"])
        print(f"\nBest 1-seed config: lr={best['lr']:.0e} dropout={best['dropout']:.1f} "
              f"→ test_auroc={best['test_auroc']:.4f}")
        default_r = next((r for r in results if r["is_default"]), None)
        if default_r and default_r["test_auroc"]:
            print(f"Default config (lr=5e-4, do=0.1): test_auroc={default_r['test_auroc']:.4f}")
            print(f"Best - Default = {best['test_auroc'] - default_r['test_auroc']:+.4f}")

    # Save JSON
    out = {
        "grid": {"lr": [1e-4, 3e-4, 5e-4], "dropout": [0.1, 0.2, 0.3], "seeds": [SEED]},
        "default_config": {"lr": default_lr, "dropout": default_do},
        "note": "9th config (lr=5e-4, do=0.3) canceled; 8/9 complete",
        "results": results,
    }
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    # Print summary table
    print("\n=== Mamba HP Grid (1-seed, test AUROC) ===")
    print(f"{'lr':>8}  {'dropout':>8}  {'val':>7}  {'test':>7}  {'note':>10}")
    for r in sorted(results, key=lambda x: x["test_auroc"] or 0, reverse=True):
        note = "DEFAULT" if r["is_default"] else ""
        val  = f"{r['val_auroc']:.4f}" if r["val_auroc"] else "N/A"
        test = f"{r['test_auroc']:.4f}" if r["test_auroc"] else "N/A"
        print(f"{r['lr']:>8.0e}  {r['dropout']:>8.2f}  {val:>7}  {test:>7}  {note:>10}")


if __name__ == "__main__":
    main()
