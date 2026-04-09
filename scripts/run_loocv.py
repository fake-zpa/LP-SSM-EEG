"""
7-fold Leave-One-Out Cross-Validation for LP-SSM-EEG vs Mamba.

Runs all 7 folds (chb01-06, chb10 as test patients) for both models.
Reports per-fold and aggregated AUROC + AUPRC.

Usage:
    conda run -n mamba2 python scripts/run_loocv.py [--model mamba_baseline|lp_ssm_eeg|both]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Use the current Python executable (already in mamba2 env when launched via conda run)
PYTHON = sys.executable

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SUBJECTS = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb10"]
FOLDS = [
    {"test": "chb01", "val": "chb02"},
    {"test": "chb02", "val": "chb03"},
    {"test": "chb03", "val": "chb04"},
    {"test": "chb04", "val": "chb05"},
    {"test": "chb05", "val": "chb06"},
    {"test": "chb06", "val": "chb10"},
    {"test": "chb10", "val": "chb05"},
]


def make_fold_manifest(orig_manifest_path: str, test_subj: str, val_subj: str) -> str:
    """Create a temporary manifest CSV with per-fold split assignments."""
    df = pd.read_csv(orig_manifest_path)
    col = "subject_id" if "subject_id" in df.columns else "subject"

    df["split"] = "train"
    df.loc[df[col] == test_subj, "split"] = "test"
    df.loc[df[col] == val_subj, "split"] = "val"
    # Exclude subjects with no ictal windows from training (chb07-09)
    no_ictal = df.groupby(col)["label"].max()
    no_ictal_subjects = no_ictal[no_ictal == 0].index.tolist()
    df = df[~df[col].isin(no_ictal_subjects)]

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def run_training(model: str, manifest_path: str, fold_idx: int, seed: int = 42,
                 max_epochs: int = 70,
                 local_loss_weight: float = 0.15,
                 warmup_epochs: int = 10,
                 min_early_stop_epochs: int = None) -> dict:
    """Train a model for one fold and return checkpoint path + metrics."""
    cmd = [
        PYTHON, "-m", "src.cli.main", "train",
        "--model", model,
        "--dataset", "chbmit",
        "--manifest", manifest_path,
        "--seed", str(seed),
        "--max-epochs", str(max_epochs),
    ]
    if model == "lp_ssm_eeg":
        cmd += ["--training-mode", "local", "--mod-features", "ictal_ratio",
                "--local-loss-weight", str(local_loss_weight),
                "--warmup-epochs", str(warmup_epochs)]
    # Apply min_early_stop_epochs to both models (Mamba also benefits from
    # a minimum training floor when validation ictal windows are scarce)
    _min_es = min_early_stop_epochs if min_early_stop_epochs is not None else warmup_epochs
    cmd += ["--min-early-stop-epochs", str(_min_es)]

    print(f"  [fold {fold_idx}] Training {model}...", flush=True)
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=1800
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: {(result.stderr or result.stdout)[-500:]}")
        return {}

    # Find the newest checkpoint created after t0
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    model_prefix = model.replace("_baseline", "")  # mamba_baseline -> mamba
    candidates = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.startswith(model.split("_")[0])
        and (p / "best.pt").exists()
        and (p / "best.pt").stat().st_mtime >= t0 - 5
    ]
    if not candidates:
        print(f"  WARNING: No new checkpoint found after training")
        return {}
    best_ckpt_dir = max(candidates, key=lambda p: (p / "best.pt").stat().st_mtime)
    run_id = best_ckpt_dir.name
    ckpt_path = best_ckpt_dir / "best.pt"
    print(f"  [fold {fold_idx}] {model} trained in {elapsed:.0f}s \u2192 {run_id}")
    return {"run_id": run_id, "ckpt_path": str(ckpt_path), "elapsed_s": elapsed}


def run_evaluation(model: str, ckpt_path: str, manifest_path: str) -> dict:
    """Evaluate a checkpoint on the fold's test split."""
    cmd = [
        PYTHON, "-m", "src.cli.main", "evaluate",
        "--model", model,
        "--checkpoint", ckpt_path,
        "--dataset", "chbmit",
        "--manifest", manifest_path,
    ]
    if model == "lp_ssm_eeg":
        cmd += ["--training-mode", "local", "--mod-features", "ictal_ratio"]

    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  EVAL ERROR: {result.stderr[-300:]}")
        return {}
    try:
        metrics = json.loads(result.stdout)
        return metrics
    except Exception:
        print(f"  EVAL parse error: {result.stdout[:200]}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["mamba_baseline", "lp_ssm_eeg", "both"])
    parser.add_argument("--max-epochs", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-loss-weight", type=float, default=0.15, dest="local_loss_weight")
    parser.add_argument("--warmup-epochs", type=int, default=10, dest="warmup_epochs")
    parser.add_argument("--min-early-stop-epochs", type=int, default=None, dest="min_early_stop_epochs",
                        help="Minimum epochs before early stopping; default = warmup-epochs")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "loocv_results.json"))
    args = parser.parse_args()

    models = ["mamba_baseline", "lp_ssm_eeg"] if args.model == "both" else [args.model]
    orig_manifest = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")

    all_results = {m: [] for m in models}

    for fold_idx, fold in enumerate(FOLDS):
        test_subj = fold["test"]
        val_subj = fold["val"]
        train_subjs = [s for s in SUBJECTS if s not in (test_subj, val_subj)]

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/7: test={test_subj}, val={val_subj}, train={train_subjs}")
        print(f"{'='*60}")

        fold_manifest = make_fold_manifest(orig_manifest, test_subj, val_subj)
        try:
            for model in models:
                train_info = run_training(
                    model, fold_manifest, fold_idx + 1,
                    seed=args.seed, max_epochs=args.max_epochs,
                    local_loss_weight=args.local_loss_weight,
                    warmup_epochs=args.warmup_epochs,
                    min_early_stop_epochs=args.min_early_stop_epochs,
                )
                if not train_info or not Path(train_info["ckpt_path"]).exists():
                    print(f"  Skipping evaluation — checkpoint not found")
                    continue

                metrics = run_evaluation(model, train_info["ckpt_path"], fold_manifest)
                fold_result = {
                    "fold": fold_idx + 1,
                    "test_subject": test_subj,
                    "val_subject": val_subj,
                    **train_info,
                    **metrics,
                }
                all_results[model].append(fold_result)
                print(f"  {model}: AUROC={metrics.get('val_auroc', 'N/A'):.4f}, AUPRC={metrics.get('val_auprc', 'N/A'):.4f}")
        finally:
            Path(fold_manifest).unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print("LOOCV SUMMARY")
    print(f"{'='*60}")

    summary = {}
    for model in models:
        results = all_results[model]
        if not results:
            continue
        aurocs = [r.get("val_auroc", 0) for r in results]
        auprcs = [r.get("val_auprc", 0) for r in results]
        summary[model] = {
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "auprc_mean": float(np.mean(auprcs)),
            "auprc_std": float(np.std(auprcs)),
            "n_folds": len(results),
            "per_fold": results,
        }
        print(f"\n{model}:")
        print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print(f"  AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
        for r in results:
            print(f"    Fold {r['fold']} (test={r['test_subject']}): AUROC={r.get('val_auroc',0):.4f}, AUPRC={r.get('val_auprc',0):.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
