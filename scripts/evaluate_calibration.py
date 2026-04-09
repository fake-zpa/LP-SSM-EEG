"""
Calibration Evaluation: ECE, Brier Score, Reliability Diagram.

Computes calibration metrics for all models on the primary test split
(chb04+chb10). Produces:
  - outputs/metrics/calibration_results.json
  - paper/fig7_calibration.pdf  (reliability diagrams + ECE/Brier table)

Usage:
    conda run -n mamba2 python scripts/evaluate_calibration.py
"""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST     = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_BINS       = 10

CHECKPOINTS = {
    "Mamba":      "outputs/checkpoints/mamba_baseline_20260330_153604/best.pt",
    # April checkpoint has band_head + ictal_ratio-only modulator (matches paper's tuned config)
    "LP-SSM-EEG": "outputs/checkpoints/lp_ssm_eeg_20260402_154659/best.pt",
    "CNN":        "outputs/checkpoints/cnn_baseline_20260330_143919/best.pt",
    "EEGNet":     "outputs/checkpoints/eegnet_20260330_143757/best.pt",
    "Transformer":"outputs/checkpoints/transformer_baseline_20260330_132908/best.pt",
}


# ─── Calibration metrics ────────────────────────────────────────────────────

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10):
    """Compute ECE with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append((lo + hi) / 2)
            bin_sizes.append(0)
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        n    = mask.sum()
        ece += (n / len(probs)) * abs(acc - conf)
        bin_accs.append(float(acc))
        bin_confs.append(float(conf))
        bin_sizes.append(int(n))
    return float(ece), bin_accs, bin_confs, bin_sizes


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


# ─── Get predictions ────────────────────────────────────────────────────────

def load_dataset():
    from src.data.dataset_chbmit import CHBMITDataset
    return CHBMITDataset(MANIFEST, split="test", subjects=["chb04", "chb10"],
                         seed=42, max_neg_pos_ratio=None)


def get_probs(model_name: str, ckpt_path: str, ds) -> np.ndarray | None:
    """Run model forward and return [N, 2] softmax probabilities."""
    from src.models import build_model
    from src.utils.config import load_config

    model_cfg_path = PROJECT_ROOT / "configs" / "model" / f"{model_name}.yaml"
    model_arch = {}
    if model_cfg_path.exists():
        cfg = load_config(str(model_cfg_path))
        model_arch = cfg.get("architecture", {})

    import pandas as pd
    mdf = pd.read_csv(MANIFEST)
    in_channels = int(mdf["n_channels"].iloc[0]) if "n_channels" in mdf.columns else 22

    model_kwargs = dict(in_channels=in_channels, n_classes=2)
    for k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
        if k in model_arch:
            model_kwargs[k] = model_arch[k]

    if model_name == "eegnet":
        model_kwargs["window_samples"] = 1024
    elif model_name == "lp_ssm_eeg":
        model_kwargs["training_mode"] = "global"
        model_kwargs["mod_use_ictal_ratio"] = True
        model_kwargs["mod_use_band_powers"] = False
        model_kwargs["mod_use_temporal_variance"] = False
        model_kwargs["mod_use_event_uncertainty"] = False

    try:
        model = build_model(model_name, **model_kwargs)
        ckpt  = torch.load(str(PROJECT_ROOT / ckpt_path), map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"  [{model_name}] load error: {e}")
        return None

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            try:
                out = model(x)
                logits = out["logits"] if isinstance(out, dict) else out
                probs  = F.softmax(logits.float(), dim=-1).cpu().numpy()
                all_probs.append(probs)
            except Exception as e:
                print(f"  [{model_name}] forward error: {e}")
                break
    if not all_probs:
        return None
    return np.concatenate(all_probs, axis=0)  # [N, 2]


# ─── Plot ───────────────────────────────────────────────────────────────────

def plot_reliability_diagrams(model_results: dict, out_path: str):
    n_models = len(model_results)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # Color palette
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    table_rows = []

    for idx, (name, res) in enumerate(model_results.items()):
        if idx >= len(axes) - 1:
            break
        ax = axes[idx]
        ece_v, bin_accs, bin_confs, bin_sizes = res["ece"], res["bin_accs"], res["bin_confs"], res["bin_sizes"]
        brier_v = res["brier"]

        # Reliability diagram
        valid = [(bc, ba, bs) for bc, ba, bs in zip(bin_confs, bin_accs, bin_sizes)
                 if not np.isnan(ba) and bs > 0]
        if valid:
            bc_v, ba_v, bs_v = zip(*valid)
            ax.bar(bc_v, ba_v, width=0.09, alpha=0.7, color=colors[idx % len(colors)],
                   label="Model", align="center")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calib.")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        table_rows.append((name, ece_v, brier_v, res["auroc"]))

    # Summary table in last subplot
    ax_t = axes[n_models] if n_models < len(axes) else axes[-1]
    ax_t.axis("off")
    col_labels = ["Model", "ECE↓", "Brier↓", "AUROC↑"]
    tbl_data = [[r[0], f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.3f}"]
                for r in sorted(table_rows, key=lambda x: x[1])]
    tbl = ax_t.table(cellText=tbl_data, colLabels=col_labels,
                     cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    # title removed — provided by LaTeX caption

    # Hide unused axes
    for i in range(n_models + 1, len(axes)):
        axes[i].set_visible(False)

    # suptitle removed — provided by LaTeX caption
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    from sklearn.metrics import roc_auc_score

    print("Loading test dataset …")
    ds = load_dataset()
    labels = np.array([int(ds[i][1]) for i in range(len(ds))])
    print(f"  {len(ds)} windows  |  ictal={labels.sum()}  interictal={(labels==0).sum()}")

    results = {}
    for display_name, ckpt_rel in CHECKPOINTS.items():
        model_key = {
            "Mamba":       "mamba_baseline",
            "LP-SSM-EEG":  "lp_ssm_eeg",
            "CNN":         "cnn_baseline",
            "EEGNet":      "eegnet",
            "Transformer": "transformer_baseline",
        }[display_name]

        print(f"\n[{display_name}] Running …")
        probs = get_probs(model_key, ckpt_rel, ds)
        if probs is None:
            print(f"  Skipped (load/forward error)")
            continue

        ictal_probs = probs[:, 1]
        ece, bin_accs, bin_confs, bin_sizes = expected_calibration_error(ictal_probs, labels, N_BINS)
        brier = brier_score(ictal_probs, labels)
        try:
            auroc = float(roc_auc_score(labels, ictal_probs))
        except Exception:
            auroc = float("nan")

        results[display_name] = {
            "ece": ece, "brier": brier, "auroc": auroc,
            "bin_accs": bin_accs, "bin_confs": bin_confs, "bin_sizes": bin_sizes,
            "n_ictal": int(labels.sum()), "n_interictal": int((labels == 0).sum()),
        }
        print(f"  ECE={ece:.4f}  Brier={brier:.4f}  AUROC={auroc:.3f}")

    # Save JSON
    out_json = str(PROJECT_ROOT / "outputs" / "metrics" / "calibration_results.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\nSaved: {out_json}")

    # Plot
    fig_path = str(PROJECT_ROOT / "paper" / "fig7_calibration.pdf")
    docs_path = str(PROJECT_ROOT / "docs" / "figures" / "fig7_calibration.png")
    plot_reliability_diagrams(results, fig_path)
    plot_reliability_diagrams(results, docs_path.replace(".pdf", ".png"))

    # Print summary
    print("\n=== Calibration Summary ===")
    print(f"{'Model':15s} {'ECE':>8} {'Brier':>8} {'AUROC':>8}")
    for name, r in sorted(results.items(), key=lambda x: x[1]["ece"]):
        print(f"{name:15s} {r['ece']:>8.4f} {r['brier']:>8.4f} {r['auroc']:>8.3f}")


if __name__ == "__main__":
    main()
