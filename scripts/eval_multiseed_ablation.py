"""
Evaluate all multi-seed ablation checkpoints on test set (chb04+chb10).

Reads from logs/train/multiseed_ablation/ and outputs a summary table
suitable for insertion into the paper's ablation section.

Usage:
    python scripts/eval_multiseed_ablation.py [--nowait]
"""
import json, re, time, subprocess, sys, statistics
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = "/root/miniconda3/envs/mamba2/bin/python"
LOG_DIR = ROOT / "logs/train/multiseed_ablation"

SEEDS   = [1, 2, 3]
VARIANTS = ["global", "local_nomod", "local_mod"]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def find_run_id(seed: int, variant: str) -> str | None:
    log_dir = LOG_DIR / f"{variant}_seed{seed}"
    log = log_dir / "stdout.log"
    if not log.exists():
        return None
    text = log.read_text()
    m = re.search(r"run_id.*?'(lp_ssm_eeg_[^']+)'", text)
    return m.group(1) if m else None


def is_done(seed: int, variant: str) -> bool:
    log_dir = LOG_DIR / f"{variant}_seed{seed}"
    log = log_dir / "stdout.log"
    if not log.exists():
        return False
    return "Training complete" in log.read_text()


def wait_for_all(nowait=False):
    needed = len(SEEDS) * len(VARIANTS)
    while True:
        done = sum(1 for s in SEEDS for v in VARIANTS if is_done(s, v))
        print(f"  ablation runs: {done}/{needed}", flush=True)
        if done >= needed or nowait:
            break
        time.sleep(30)
    print()


def run_eval(run_id: str, variant: str) -> dict | None:
    ckpt = ROOT / "outputs/checkpoints" / run_id / "best.pt"
    if not ckpt.exists():
        print(f"  checkpoint not found: {ckpt}")
        return None

    extra = ""
    if variant == "global":
        extra = "--training-mode global"
    elif variant == "local_nomod":
        extra = "--training-mode local --no-modulator"
    else:  # local_mod
        extra = "--training-mode local --mod-features ictal_ratio"

    cmd = (f"{PYTHON} -m src.cli.main evaluate "
           f"--model lp_ssm_eeg "
           f"--checkpoint {ckpt} "
           f"--dataset chbmit "
           f"{extra}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT))
    # Try 1: parse the last {...} JSON block from stdout (pretty-printed)
    combined = proc.stdout + "\n" + proc.stderr
    for line in reversed(combined.splitlines()):
        line = line.strip()
        # inline compact JSON on INFO logger line
        if '"val_auroc"' in line:
            m = re.search(r'\{.*"val_auroc".*\}', line)
            if m:
                try:
                    d = json.loads(m.group(0))
                    if "val_auroc" in d:
                        return d
                except Exception:
                    pass
    # Try 2: reconstruct multi-line JSON block from stdout
    stdout = proc.stdout.strip()
    if stdout:
        # find last {...} block
        start = stdout.rfind('{')
        end   = stdout.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                d = json.loads(stdout[start:end+1])
                if "val_auroc" in d:
                    return d
            except Exception:
                pass
    return None


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    nowait = "--nowait" in sys.argv
    print("Waiting for multi-seed ablation to finish...")
    wait_for_all(nowait=nowait)

    results = {}  # variant → list of test_auroc
    for variant in VARIANTS:
        results[variant] = []
        for seed in SEEDS:
            if not is_done(seed, variant):
                print(f"  [{variant} seed={seed}] not done, skipping")
                continue
            run_id = find_run_id(seed, variant)
            if run_id is None:
                print(f"  [{variant} seed={seed}] run_id not found")
                continue
            print(f"  Evaluating [{variant} seed={seed}] {run_id} ...", flush=True)
            metrics = run_eval(run_id, variant)
            if metrics:
                ta = metrics.get("val_auroc")
                tp = metrics.get("val_auprc")
                print(f"    test_auroc={ta:.4f}  test_auprc={tp:.4f}")
                results[variant].append({"seed": seed, "run_id": run_id,
                                         "test_auroc": ta, "test_auprc": tp})
            else:
                print(f"    ERROR: eval failed")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = ROOT / "outputs/metrics/multiseed_ablation_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print("Multi-Seed Ablation — Test AUROC on chb04+chb10")
    print("=" * 55)
    labels = {
        "global":      "LP-SSM (global, no local obj.)",
        "local_nomod": "LP-SSM (local, no modulator)",
        "local_mod":   "LP-SSM (local + ictal-ratio mod.)",
    }
    for variant in VARIANTS:
        aurocs = [r["test_auroc"] for r in results[variant] if r.get("test_auroc")]
        if aurocs:
            mu = statistics.mean(aurocs)
            sd = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
            print(f"  {labels[variant]:<38}  {mu:.4f} ± {sd:.4f}  (n={len(aurocs)})")
        else:
            print(f"  {labels[variant]:<38}  no results")
    print()

    # print LaTeX row
    print("=== LaTeX rows for ablation table ===")
    for variant in VARIANTS:
        aurocs = [r["test_auroc"] for r in results[variant] if r.get("test_auroc")]
        if aurocs:
            mu = statistics.mean(aurocs)
            sd = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
            print(f"  {labels[variant]} & ${mu:.3f}\\pm{sd:.3f}$ \\\\")


if __name__ == "__main__":
    main()
