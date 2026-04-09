"""
Batch evaluation script for extra_seeds and learning_curve experiments.
Waits until all training runs complete, then evaluates each best.pt on
the test set (chb04+chb10) and prints a summary table.

Usage:
    python scripts/eval_new_experiments.py          # wait + eval all
    python scripts/eval_new_experiments.py --nowait # eval completed only
"""
import re, glob, json, time, subprocess, sys, statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PYTHON = "/root/miniconda3/envs/mamba2/bin/python"

EXTRA_SEEDS_LOGS = sorted(glob.glob(str(PROJECT_ROOT / "logs/train/extra_seeds/*.log")))
LC_LOGS = sorted(glob.glob(str(PROJECT_ROOT / "logs/train/learning_curve/*.log")))

# Expected total counts
EXPECTED_SEEDS = 16   # 8 seeds × 2 models
EXPECTED_LC    = 24   # 4 n_values × 2 models × 3 seeds


# ─────────────────────────────────────────────────────────────────────────────
def parse_completed(log_files):
    """Return list of dicts: {log_stem, run_id, model, val_auroc, ...}"""
    results = []
    for path in log_files:
        text = Path(path).read_text()
        m = re.search(r"Training complete.*?'run_id': '([^']+)'.*?'best_val_metric': ([0-9.]+)", text)
        if not m:
            continue
        run_id = m.group(1)
        val_auroc = float(m.group(2))
        model = "mamba_baseline" if "mamba_baseline" in run_id else "lp_ssm_eeg"
        results.append({
            "log": path,
            "log_stem": Path(path).stem,
            "run_id": run_id,
            "model": model,
            "val_auroc": val_auroc,
        })
    return results


def count_done(log_files):
    return sum(1 for p in log_files
               if Path(p).stem != "run" and
               re.search(r"Training complete", Path(p).read_text()))


def wait_for_all(nowait=False):
    print("Waiting for all training runs to complete...")
    while True:
        s_done = count_done(EXTRA_SEEDS_LOGS)
        lc_done = count_done(LC_LOGS)
        total = s_done + lc_done
        needed = EXPECTED_SEEDS + EXPECTED_LC
        print(f"  extra_seeds: {s_done}/{EXPECTED_SEEDS}  |  "
              f"learning_curve: {lc_done}/{EXPECTED_LC}  |  "
              f"total: {total}/{needed}", flush=True)
        if total >= needed or nowait:
            break
        time.sleep(60)
    print("Proceeding to evaluation.\n")


# ─────────────────────────────────────────────────────────────────────────────
def run_eval(run_id, model, extra_args=""):
    ckpt = CKPT_DIR / run_id / "best.pt"
    if not ckpt.exists():
        return None, f"checkpoint not found: {ckpt}"

    cmd = (
        f"{PYTHON} {PROJECT_ROOT}/src/cli/main.py evaluate "
        f"--model {model} "
        f"--checkpoint {ckpt} "
        f"--dataset chbmit "
        f"--training-mode {'local' if model == 'lp_ssm_eeg' else 'global'} "
        f"{extra_args}"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    # parse JSON output from evaluate command
    for line in reversed(proc.stdout.splitlines()):
        try:
            d = json.loads(line)
            if "val_auroc" in d or "auroc" in d:
                return d, None
        except Exception:
            pass
    return None, proc.stderr[-300:] if proc.stderr else "no JSON output"


# ─────────────────────────────────────────────────────────────────────────────
def label_from_stem(stem, group):
    """Return human-readable label for a log stem."""
    if group == "seeds":
        m = re.match(r"(mamba|lpssm)_seed(\d+)", stem)
        if m:
            model = "Mamba" if m.group(1) == "mamba" else "LP-SSM"
            return model, f"seed={m.group(2)}", None
    elif group == "lc":
        m = re.match(r"(mamba|lpssm)_n(\d+)_seed(\d+)", stem)
        if m:
            model = "Mamba" if m.group(1) == "mamba" else "LP-SSM"
            return model, f"n={m.group(2)}", f"seed={m.group(3)}"
    return stem, "", ""


# ─────────────────────────────────────────────────────────────────────────────
def main():
    nowait = "--nowait" in sys.argv
    wait_for_all(nowait=nowait)

    all_logs = [p for p in EXTRA_SEEDS_LOGS + LC_LOGS if Path(p).stem != "run"]
    completed = parse_completed(all_logs)
    print(f"Found {len(completed)} completed runs to evaluate.\n")

    eval_results = []
    for i, r in enumerate(completed, 1):
        stem = r["log_stem"]
        print(f"[{i:2}/{len(completed)}] Evaluating {stem} ({r['run_id']}) ...", flush=True)
        metrics, err = run_eval(r["run_id"], r["model"])
        if err:
            print(f"  ERROR: {err}")
            eval_results.append({**r, "test_auroc": None, "test_auprc": None, "error": err})
        else:
            ta = metrics.get("val_auroc", metrics.get("auroc"))
            tp = metrics.get("val_auprc", metrics.get("auprc"))
            print(f"  val_auroc={r['val_auroc']:.4f}  test_auroc={ta:.4f}  test_auprc={tp:.4f}")
            eval_results.append({**r, "test_auroc": round(ta,4) if ta else None, "test_auprc": round(tp,4) if tp else None, "error": None})

    # ── Save raw results ──────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "outputs" / "metrics" / "eval_new_experiments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nRaw results saved → {out_path}\n")

    # ── Summary table: Extra Seeds ────────────────────────────────────────────
    seed_res = [r for r in eval_results if "seed" in r["log_stem"] and "_n" not in r["log_stem"]]
    if seed_res:
        print("═" * 60)
        print("EXTRA SEEDS — Test AUROC on chb04+chb10")
        print("═" * 60)
        mamba_t = [r["test_auroc"] for r in seed_res if r["model"] == "mamba_baseline" and r["test_auroc"] is not None]
        lpssm_t = [r["test_auroc"] for r in seed_res if r["model"] == "lp_ssm_eeg" and r["test_auroc"] is not None]
        mamba_v = [r["val_auroc"] for r in seed_res if r["model"] == "mamba_baseline"]
        lpssm_v = [r["val_auroc"] for r in seed_res if r["model"] == "lp_ssm_eeg"]
        print(f"  Mamba  n={len(mamba_t)}  "
              f"val={statistics.mean(mamba_v):.4f}±{statistics.stdev(mamba_v):.4f}  "
              + (f"test={statistics.mean(mamba_t):.4f}±{statistics.stdev(mamba_t):.4f}" if len(mamba_t)>1 else ""))
        print(f"  LP-SSM n={len(lpssm_t)}  "
              f"val={statistics.mean(lpssm_v):.4f}±{statistics.stdev(lpssm_v):.4f}  "
              + (f"test={statistics.mean(lpssm_t):.4f}±{statistics.stdev(lpssm_t):.4f}" if len(lpssm_t)>1 else ""))
        if mamba_t and lpssm_t:
            delta = statistics.mean(lpssm_t) - statistics.mean(mamba_t)
            print(f"  Δ (LP-SSM − Mamba) = {delta:+.4f}")

    # ── Summary table: Learning Curve ─────────────────────────────────────────
    lc_res = [r for r in eval_results if "_n" in r["log_stem"]]
    if lc_res:
        print("\n" + "═" * 60)
        print("LEARNING CURVE — Test AUROC on chb04+chb10")
        print("═" * 60)
        print(f"  {'N':>2}  {'Model':<8}  {'mean±std (test)':>18}  {'Δ':>8}")
        print("  " + "-" * 44)
        for n in [1, 2, 3, 4]:
            m_vals = [r["test_auroc"] for r in lc_res
                      if r["model"] == "mamba_baseline" and f"_n{n}_" in r["log_stem"] and r["test_auroc"] is not None]
            l_vals = [r["test_auroc"] for r in lc_res
                      if r["model"] == "lp_ssm_eeg" and f"_n{n}_" in r["log_stem"] and r["test_auroc"] is not None]
            def fmt(vals):
                if len(vals) > 1:
                    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"
                elif len(vals) == 1:
                    return f"{vals[0]:.4f}"
                return "–"
            delta = (f"{statistics.mean(l_vals)-statistics.mean(m_vals):+.4f}"
                     if m_vals and l_vals else "–")
            print(f"  {n:>2}  {'Mamba':<8}  {fmt(m_vals):>18}")
            print(f"  {n:>2}  {'LP-SSM':<8}  {fmt(l_vals):>18}  {delta:>8}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
