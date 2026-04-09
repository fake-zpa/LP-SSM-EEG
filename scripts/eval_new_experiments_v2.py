"""
Batch evaluation for extra_seeds + learning_curve experiments.
Runs evaluation directly (no subprocess) to avoid buffering issues.
"""
import re, glob, json, sys, statistics
from pathlib import Path
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.utils.config import load_config
from src.eval.metrics_classification import compute_metrics

CKPT_DIR      = PROJECT_ROOT / "outputs" / "checkpoints"
MANIFEST_PATH = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

EXTRA_SEEDS_LOGS = sorted(glob.glob(str(PROJECT_ROOT / "logs/train/extra_seeds/*.log")))
LC_LOGS          = sorted(glob.glob(str(PROJECT_ROOT / "logs/train/learning_curve/*.log")))


def parse_completed(log_files):
    results = []
    for path in log_files:
        if Path(path).stem == "run":
            continue
        text = Path(path).read_text()
        m = re.search(r"Training complete.*?'run_id': '([^']+)'.*?'best_val_metric': ([0-9.]+)", text)
        if not m:
            continue
        run_id   = m.group(1)
        val_auroc = float(m.group(2))
        model    = "mamba_baseline" if "mamba_baseline" in run_id else "lp_ssm_eeg"
        results.append({"log": path, "log_stem": Path(path).stem,
                        "run_id": run_id, "model": model, "val_auroc": val_auroc})
    return results


def evaluate_checkpoint(run_id, model_name):
    ckpt_path = CKPT_DIR / run_id / "best.pt"
    if not ckpt_path.exists():
        return None, f"ckpt not found: {ckpt_path}"

    import pandas as _pd
    _mdf = _pd.read_csv(MANIFEST_PATH)
    in_channels = int(_mdf["n_channels"].iloc[0]) if "n_channels" in _mdf.columns else 22

    model_cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    model_arch = {}
    if model_cfg_path.exists():
        model_arch = load_config(str(model_cfg_path)).get("architecture", {})

    model_kwargs = dict(in_channels=in_channels, n_classes=2)
    for k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
        if k in model_arch:
            model_kwargs[k] = model_arch[k]
    if model_name == "lp_ssm_eeg":
        model_kwargs["training_mode"] = "global"

    model = build_model(model_name, **model_kwargs)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()

    from src.data.dataset_chbmit import CHBMITDataset
    test_ds = CHBMITDataset(MANIFEST_PATH, split="test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(DEVICE))
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.float().cpu())
            all_targets.append(y)

    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_targets))
    return metrics, None


def main():
    all_logs  = EXTRA_SEEDS_LOGS + LC_LOGS
    completed = parse_completed(all_logs)
    print(f"Found {len(completed)} completed runs.\n", flush=True)

    eval_results = []
    for i, r in enumerate(completed, 1):
        print(f"[{i:2}/{len(completed)}] {r['log_stem']}  val_auroc={r['val_auroc']:.4f} ...",
              end=" ", flush=True)
        metrics, err = evaluate_checkpoint(r["run_id"], r["model"])
        if err:
            print(f"ERROR: {err}", flush=True)
            eval_results.append({**r, "test_auroc": None, "test_auprc": None, "error": err})
        else:
            ta = round(metrics["val_auroc"], 4)
            tp = round(metrics["val_auprc"], 4)
            print(f"test_auroc={ta:.4f}  test_auprc={tp:.4f}", flush=True)
            eval_results.append({**r, "test_auroc": ta, "test_auprc": tp, "error": None})

    # Save
    out_path = PROJECT_ROOT / "outputs" / "metrics" / "eval_new_experiments.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved → {out_path}\n", flush=True)

    # ── Summary: Extra Seeds ──────────────────────────────────────────────────
    seed_res = [r for r in eval_results if "seed" in r["log_stem"] and "_n" not in r["log_stem"]]
    if seed_res:
        print("═" * 58)
        print("EXTRA SEEDS  —  Test AUROC on chb04+chb10")
        print("═" * 58)
        print(f"  {'Seed':<8} {'Mamba':>8} {'LP-SSM':>8}")
        seeds_done = sorted(set(int(re.search(r'\d+', r["log_stem"]).group())
                                for r in seed_res))
        for s in seeds_done:
            m = next((r["test_auroc"] for r in seed_res
                      if r["model"]=="mamba_baseline" and f"seed{s}"==r["log_stem"].split("_")[-1]), None)
            l = next((r["test_auroc"] for r in seed_res
                      if r["model"]=="lp_ssm_eeg" and f"seed{s}"==r["log_stem"].split("_")[-1]), None)
            print(f"  {s:<8} {str(m) if m else '–':>8} {str(l) if l else '–':>8}")
        mt = [r["test_auroc"] for r in seed_res if r["model"]=="mamba_baseline" and r["test_auroc"]]
        lt = [r["test_auroc"] for r in seed_res if r["model"]=="lp_ssm_eeg"     and r["test_auroc"]]
        if len(mt) > 1:
            print(f"\n  Mamba  n={len(mt)}  mean={statistics.mean(mt):.4f}  std={statistics.stdev(mt):.4f}")
        if len(lt) > 1:
            print(f"  LP-SSM n={len(lt)}  mean={statistics.mean(lt):.4f}  std={statistics.stdev(lt):.4f}")
        if mt and lt:
            print(f"  Δ = {statistics.mean(lt)-statistics.mean(mt):+.4f}")

    # ── Summary: Learning Curve ───────────────────────────────────────────────
    lc_res = [r for r in eval_results if "_n" in r["log_stem"] and r["test_auroc"] is not None]
    if lc_res:
        print("\n" + "═" * 58)
        print("LEARNING CURVE  —  Test AUROC on chb04+chb10")
        print("═" * 58)
        print(f"  {'N':>2}  {'Mamba mean±std':>18}  {'LP-SSM mean±std':>18}  {'Δ':>7}")
        print("  " + "-" * 50)
        for n in [1, 2, 3, 4]:
            mv = [r["test_auroc"] for r in lc_res if r["model"]=="mamba_baseline" and f"_n{n}_" in r["log_stem"]]
            lv = [r["test_auroc"] for r in lc_res if r["model"]=="lp_ssm_eeg"     and f"_n{n}_" in r["log_stem"]]
            def fmt(v):
                if len(v) > 1: return f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}"
                if len(v) == 1: return f"{v[0]:.4f}"
                return "–"
            delta = f"{statistics.mean(lv)-statistics.mean(mv):+.4f}" if mv and lv else "–"
            print(f"  {n:>2}  {fmt(mv):>18}  {fmt(lv):>18}  {delta:>7}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
