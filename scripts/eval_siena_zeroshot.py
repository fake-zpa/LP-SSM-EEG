"""
Zero-shot evaluation of LP-SSM-EEG and Mamba on SIENA EEG.

Uses CHB-MIT-trained models directly on SIENA test windows (no fine-tuning).
Reports per-patient and aggregate AUROC/AUPRC.

Usage:
    conda run -n mamba2 python scripts/eval_siena_zeroshot.py
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MANIFEST  = str(PROJECT_ROOT / "data" / "manifests" / "siena_manifest.csv")
OUT_JSON  = str(PROJECT_ROOT / "outputs" / "metrics" / "siena_zeroshot.json")

MODELS = {
    "Mamba":     ("mamba_baseline",
                  "outputs/checkpoints/mamba_baseline_20260330_153604/best.pt"),
    "LP-SSM-EEG":("lp_ssm_eeg",
                  "outputs/checkpoints/lp_ssm_eeg_20260402_154659/best.pt"),
}


class SIENADataset(Dataset):
    def __init__(self, manifest_path: str):
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == "test"].reset_index(drop=True)
        self.df = df

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        x     = np.load(str(PROJECT_ROOT / row["npy_path"]))  # [22, 1024]
        label = int(row["label"])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def load_model(model_name: str, ckpt_rel: str):
    from src.models import build_model
    from src.utils.config import load_config

    cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    model_arch = {}
    if cfg_path.exists():
        model_arch = load_config(str(cfg_path)).get("architecture", {})

    model_kwargs = dict(in_channels=22, n_classes=2)
    for k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
        if k in model_arch:
            model_kwargs[k] = model_arch[k]

    if model_name == "lp_ssm_eeg":
        model_kwargs.update(training_mode="global",
                            mod_use_ictal_ratio=True,
                            mod_use_band_powers=False,
                            mod_use_temporal_variance=False,
                            mod_use_event_uncertainty=False)

    model = build_model(model_name, **model_kwargs)
    ckpt  = torch.load(str(PROJECT_ROOT / ckpt_rel), map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def run_inference(model, ds: SIENADataset):
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out    = model(x.to(DEVICE))
            logits = out["logits"] if isinstance(out, dict) else out
            probs  = F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not Path(MANIFEST).exists():
        print(f"Manifest not found: {MANIFEST}")
        print("Run scripts/download_siena.py first.")
        return

    ds = SIENADataset(MANIFEST)
    df = ds.df
    print(f"SIENA test: {len(ds)} windows, "
          f"{df['label'].sum()} ictal ({df['label'].mean()*100:.1f}%), "
          f"{df['subject_id'].nunique()} subjects")

    if df['label'].sum() == 0:
        print("No ictal windows found - check preprocessing annotation parsing.")
        return

    results = {}
    for name, (model_key, ckpt) in MODELS.items():
        print(f"\n[{name}] Loading …")
        try:
            model = load_model(model_key, ckpt)
        except Exception as e:
            print(f"  Load error: {e}")
            continue

        print(f"  Running inference …")
        probs, labels = run_inference(model, ds)

        try:
            auroc = float(roc_auc_score(labels, probs))
            auprc = float(average_precision_score(labels, probs))
        except Exception as e:
            print(f"  Metric error: {e}")
            auroc = auprc = float("nan")

        # Per-patient breakdown
        per_patient = {}
        for subj in df["subject_id"].unique():
            mask = (df["subject_id"] == subj).values
            if mask.sum() < 5 or labels[mask].sum() == 0:
                continue
            try:
                pauroc = float(roc_auc_score(labels[mask], probs[mask]))
                pauprc = float(average_precision_score(labels[mask], probs[mask]))
                per_patient[subj] = {"auroc": pauroc, "auprc": pauprc,
                                     "n": int(mask.sum()),
                                     "n_ictal": int(labels[mask].sum())}
            except Exception:
                pass

        # Macro-average: equal weight per subject (unaffected by PN10 volume dominance)
        macro_auroc = float(np.mean([v["auroc"] for v in per_patient.values()])) if per_patient else float("nan")
        macro_auprc = float(np.mean([v["auprc"] for v in per_patient.values()])) if per_patient else float("nan")

        results[name] = {
            "auroc": auroc, "auprc": auprc,
            "macro_auroc": macro_auroc, "macro_auprc": macro_auprc,
            "n_windows": int(len(labels)),
            "n_ictal": int(labels.sum()),
            "per_patient": per_patient,
        }
        print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  macro_AUROC={macro_auroc:.3f}  macro_AUPRC={macro_auprc:.3f}")

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nSaved: {OUT_JSON}")

    print("\n=== SIENA Zero-Shot Results ===")
    print(f"{'Model':15s} {'AUROC':>8} {'AUPRC':>8} {'macro-AUROC':>12} {'macro-AUPRC':>12}")
    for n, r in results.items():
        print(f"{n:15s} {r['auroc']:>8.3f} {r['auprc']:>8.3f} {r.get('macro_auroc', float('nan')):>12.3f} {r.get('macro_auprc', float('nan')):>12.3f}")


if __name__ == "__main__":
    main()
