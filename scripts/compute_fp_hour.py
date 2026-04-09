"""
Compute False-Positive rate (FP/hour) for all primary-split models.

Usage:
    conda run -n mamba2 python scripts/compute_fp_hour.py

Outputs: outputs/metrics/fp_hour_results.json
"""

import json, sys, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
SFREQ = 256          # Hz
WINDOW_SEC = 4.0     # seconds per window
SAMPLES_PER_WIN = int(SFREQ * WINDOW_SEC)


def load_model(model_name: str, ckpt_path: str):
    from src.models import build_model
    import pandas as pd
    df = pd.read_csv(MANIFEST)
    in_ch = int(df["n_channels"].iloc[0]) if "n_channels" in df.columns else 23

    from src.utils.config import load_config
    cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    model_kwargs = dict(in_channels=in_ch, n_classes=2)
    if cfg_path.exists():
        cfg = load_config(str(cfg_path))
        arch = cfg.get("architecture", {})
        for k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
            if k in arch:
                model_kwargs[k] = arch[k]

    # LP-SSM needs training_mode; Mamba does not
    if model_name == "lp_ssm_eeg":
        model_kwargs["training_mode"] = "global"
        model_kwargs["mod_use_ictal_ratio"] = True

    model = build_model(model_name, **model_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def compute_fp_hour(model, model_name: str, threshold: float, device: str):
    """Return dict with FP/hour and supporting counts."""
    from src.data.dataset_chbmit import CHBMITDataset
    from torch.utils.data import DataLoader

    if model_name == "lp_ssm_eeg":
        model.training_mode = "global"

    model.eval().to(device)
    ds = CHBMITDataset(MANIFEST, split="test")
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=2, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            all_probs.append(probs.cpu())
            all_labels.append(y)

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    # Only interictal windows count for FP
    interictal_mask = labels == 0
    interictal_probs = probs[interictal_mask]
    n_interictal = interictal_mask.sum()
    n_ictal      = (~interictal_mask).sum()

    fps = (interictal_probs >= threshold).sum()
    total_interictal_sec = n_interictal * WINDOW_SEC
    total_interictal_hours = total_interictal_sec / 3600.0

    fp_per_hour = float(fps) / total_interictal_hours if total_interictal_hours > 0 else 0.0

    # Also compute at multiple thresholds for curve
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fp_curve = {}
    for thr in thresholds:
        fps_thr = (interictal_probs >= thr).sum()
        fp_curve[str(thr)] = round(float(fps_thr) / total_interictal_hours, 3)

    return {
        "threshold": threshold,
        "fp_per_hour": round(fp_per_hour, 3),
        "total_fps": int(fps),
        "total_interictal_windows": int(n_interictal),
        "total_ictal_windows": int(n_ictal),
        "total_interictal_hours": round(total_interictal_hours, 2),
        "fp_per_hour_curve": fp_curve,
    }


RUNS = [
    # (model_name, checkpoint_path, threshold, display_name)
    # Threshold = paper Table-1 val_threshold (sensitivity/specificity operating point)
    ("eegnet",
     "outputs/checkpoints/eegnet_20260330_143757/best.pt",
     0.5, "EEGNet"),
    ("cnn_baseline",
     "outputs/checkpoints/cnn_baseline_20260330_143919/best.pt",
     0.5, "CNN"),
    ("transformer_baseline",
     "outputs/checkpoints/transformer_baseline_20260330_132908/best.pt",
     0.5, "Transformer"),
    ("mamba_baseline",
     "outputs/checkpoints/mamba_baseline_20260330_153604/best.pt",
     0.5, "Mamba"),
    ("lp_ssm_eeg",
     "outputs/checkpoints/lp_ssm_eeg_20260330_155556/best.pt",
     0.9, "LP-SSM-EEG"),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    results = {}

    for model_name, ckpt_rel, threshold, label in RUNS:
        ckpt = str(PROJECT_ROOT / ckpt_rel)
        if not Path(ckpt).exists():
            print(f"  SKIP (checkpoint not found): {ckpt_rel}")
            continue
        print(f"\n[{label}] ckpt={ckpt_rel.split('/')[-2]}, threshold={threshold}")
        try:
            m = load_model(model_name, ckpt)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        r = compute_fp_hour(m, model_name, threshold, device)
        r["model"] = label
        r["ckpt"] = ckpt_rel
        results[label] = r
        print(f"  FP/hour={r['fp_per_hour']:.3f}  "
              f"(FPs={r['total_fps']}, interictal_hours={r['total_interictal_hours']:.1f}h)")
        print(f"  FP curve: {r['fp_per_hour_curve']}")

    out_path = PROJECT_ROOT / "outputs" / "metrics" / "fp_hour_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
