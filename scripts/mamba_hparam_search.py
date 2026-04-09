"""
Mamba Baseline Hyperparameter Grid Search (Equal Tuning).

Mirrors the LP-SSM HP search effort (9 configs × 3 seeds = 27 runs).
Grid: lr ∈ {1e-4, 3e-4, 5e-4} × dropout ∈ {0.1, 0.2, 0.3}
Evaluates on test split (chb04+chb10) using the primary manifest.

Outputs:
  outputs/metrics/mamba_hparam_search.json
  Paper-ready table: best tuned Mamba vs default Mamba
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
OUT_JSON = str(PROJECT_ROOT / "outputs" / "metrics" / "mamba_hparam_search.json")

LR_VALUES      = [1e-4, 3e-4, 5e-4]
DROPOUT_VALUES = [0.1, 0.2, 0.3]
SEEDS          = [1, 2, 3]
MAX_EPOCHS     = 70
DEFAULT_MAMBA  = {"lr": 5e-4, "dropout": 0.1}   # matches mamba_baseline.yaml


def run_one(lr: float, dropout: float, seed: int, max_epochs: int = MAX_EPOCHS) -> dict:
    """Train Mamba with given lr/dropout/seed, return val+test metrics."""
    cmd = [
        PYTHON, "-m", "src.cli.main", "train",
        "--model", "mamba_baseline",
        "--dataset", "chbmit",
        "--manifest", MANIFEST,
        "--seed", str(seed),
        "--max-epochs", str(max_epochs),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--log-dir", "logs/mamba_hpsearch",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                          text=True, timeout=2400)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  TRAIN ERROR rc={proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}")
        return {"lr": lr, "dropout": dropout, "seed": seed, "val_auroc": None,
                "test_auroc": None, "checkpoint": None, "elapsed_s": round(elapsed, 1),
                "returncode": proc.returncode}

    # Discover newest checkpoint by mtime (same as run_loocv.py)
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.startswith("mamba")
        and (p / "best.pt").exists()
        and (p / "best.pt").stat().st_mtime >= t0 - 5
    ]
    ckpt_path = None
    if candidates:
        best_dir = max(candidates, key=lambda p: (p / "best.pt").stat().st_mtime)
        ckpt_path = str(best_dir / "best.pt")
    else:
        print("  WARNING: no new checkpoint found")

    # Evaluate on test split
    test_auroc = None
    test_auprc = None
    if ckpt_path and Path(ckpt_path).exists():
        eval_cmd = [
            PYTHON, "-m", "src.cli.main", "evaluate",
            "--model", "mamba_baseline",
            "--checkpoint", ckpt_path,
            "--dataset", "chbmit",
            "--manifest", MANIFEST,
        ]
        ep = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT),
                            capture_output=True, text=True, timeout=300)
        try:
            d = json.loads(ep.stdout)
            test_auroc = d.get("auroc")
            test_auprc = d.get("auprc")
        except Exception:
            for line in ep.stdout.splitlines():
                try:
                    d = json.loads(line)
                    if "auroc" in d:
                        test_auroc = d["auroc"]
                        test_auprc = d.get("auprc")
                        break
                except Exception:
                    pass

    # Parse best val AUROC from training stdout
    import re
    val_auroc = None
    for line in (proc.stdout + proc.stderr).splitlines():
        m = re.search(r"best[_\s]*(val[_\s]*)?auroc[:\s=]+([0-9.]+)", line, re.IGNORECASE)
        if m:
            try:
                val_auroc = float(m.group(2))
            except Exception:
                pass

    return {
        "lr": lr, "dropout": dropout, "seed": seed,
        "val_auroc": val_auroc, "test_auroc": test_auroc, "test_auprc": test_auprc,
        "checkpoint": ckpt_path,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
    }


def _key(lr, dropout, seed):
    return f"{lr:.6f}_{dropout:.2f}_{seed}"


def load_existing():
    """Load already-completed results from JSON."""
    p = Path(OUT_JSON)
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get("results", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke: 1 seed only, 3 epochs (sanity check)")
    args = parser.parse_args()

    configs = [(lr, do) for lr in LR_VALUES for do in DROPOUT_VALUES]
    seeds = SEEDS[:1] if args.smoke else SEEDS
    max_ep = 3 if args.smoke else MAX_EPOCHS

    # Resume: load existing results, skip completed combos
    existing = load_existing()
    done_keys = set()
    for r in existing:
        if r.get("test_auroc") is not None:
            done_keys.add(_key(r["lr"], r["dropout"], r["seed"]))
    results = list(existing)  # keep all prior results

    todo = []
    for lr, do in configs:
        for seed in seeds:
            if _key(lr, do, seed) not in done_keys:
                todo.append((lr, do, seed))

    total = len(configs) * len(seeds)
    print(f"Mamba HP grid search: {len(configs)} configs × {len(seeds)} seeds = {total} total")
    print(f"Already done: {len(done_keys)}, remaining: {len(todo)}")
    print(f"Grid: lr={LR_VALUES}, dropout={DROPOUT_VALUES}")
    print("=" * 60)

    for i, (lr, dropout, seed) in enumerate(todo, 1):
        tag = f"lr={lr:.0e} dropout={dropout} seed={seed}"
        print(f"\n[{i}/{len(todo)}] {tag}")
        r = run_one(lr, dropout, seed, max_epochs=max_ep)
        status = f"val={r['val_auroc']} test={r['test_auroc']}" if r["test_auroc"] else f"rc={r.get('returncode')}"
        print(f"  → {status} ({r.get('elapsed_s', 0):.0f}s)")
        results.append(r)

        # Save after each run (crash-safe)
        _save(results, seeds)

    _save(results, seeds)
    _print_summary(results, configs)


def _save(results, seeds):
    import numpy as np
    out = {
        "grid": {"lr": LR_VALUES, "dropout": DROPOUT_VALUES, "seeds": list(seeds)},
        "default_config": DEFAULT_MAMBA,
        "results": results,
    }
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [saved {len(results)} results → {OUT_JSON}]")


def _print_summary(results, configs):
    import numpy as np
    print("\n=== HP Grid Summary (test AUROC, mean ± std over seeds) ===")
    print(f"{'lr':>8} {'dropout':>8} {'n':>4} {'mean':>8} {'std':>8} {'note':>10}")
    for lr, dropout in configs:
        seed_rs = [r for r in results if abs(r["lr"] - lr) < 1e-10 and abs(r["dropout"] - dropout) < 1e-4]
        vals = [r["test_auroc"] for r in seed_rs if r.get("test_auroc") is not None]
        if vals:
            mn, sd = np.mean(vals), np.std(vals)
            note = "DEFAULT" if abs(lr - DEFAULT_MAMBA["lr"]) < 1e-8 and abs(dropout - DEFAULT_MAMBA["dropout"]) < 1e-4 else ""
            print(f"{lr:>8.0e} {dropout:>8.2f} {len(vals):>4} {mn:>8.4f} {sd:>8.4f} {note:>10}")


if __name__ == "__main__":
    main()
