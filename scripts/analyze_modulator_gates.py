"""
Modulator Gate Value Analysis — Figure F4 for LP-SSM-EEG paper.

Loads a trained LP-SSM-EEG checkpoint (ictal_ratio modulator),
hooks into each block's modulator, and collects gate values w ∈ [0.5, 2.0]
for all test windows. Plots ictal vs interictal w distribution.

Usage:
    python scripts/analyze_modulator_gates.py --checkpoint <path/to/best.pt>
    python scripts/analyze_modulator_gates.py --auto  # picks latest lp_ssm_eeg ckpt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_latest_lp_ssm_checkpoint():
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    candidates = sorted(
        ckpt_dir.glob("lp_ssm_eeg_*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            m = ckpt.get("metrics", {})
            val = m.get("val_auroc", 0)
            if 0.72 <= val <= 0.74:
                print(f"[auto] Selected {p.parent.name} (val_auroc={val:.4f})")
                return str(p)
        except Exception:
            continue
    print("[auto] Falling back to most recent checkpoint")
    return str(candidates[0]) if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--auto", action="store_true", help="Auto-select best ictal_ratio checkpoint")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "docs" / "figures"))
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if args.auto or ckpt_path is None:
        ckpt_path = find_latest_lp_ssm_checkpoint()
    if not ckpt_path:
        print("No checkpoint found. Pass --checkpoint <path>"); sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    from src.models.lp_ssm_eeg import LPSSMEEG
    from src.data.dataset_chbmit import CHBMITDataset

    manifest = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
    import pandas as pd
    mdf = pd.read_csv(manifest)
    in_channels = int(mdf["n_channels"].iloc[0]) if "n_channels" in mdf.columns else 23

    model = LPSSMEEG(
        in_channels=in_channels,
        n_classes=2,
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=6,
        training_mode="local",
        use_modulator=True,
        mod_use_band_powers=False,
        mod_use_ictal_ratio=True,
        mod_use_temporal_variance=False,
        mod_use_event_uncertainty=False,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    gate_values = []
    labels = []
    hooks = []

    def make_hook(block_idx):
        def hook(module, input, output):
            w = output.detach().cpu()
            gate_values.append((block_idx, w))
        return hook

    for i, block in enumerate(model.blocks):
        h = block.modulator.register_forward_hook(make_hook(i))
        hooks.append(h)

    test_ds = CHBMITDataset(manifest, split="test")
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    print(f"Test set: {len(test_ds)} windows")

    batch_labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            gate_values.clear()
            _ = model(x, return_local_losses=True)
            if gate_values:
                batch_labels.append(y.cpu().numpy())

    for h in hooks:
        h.remove()

    print(f"Collected {len(batch_labels)} batches, {sum(len(b) for b in batch_labels)} windows")

    all_labels = np.concatenate(batch_labels, axis=0) if batch_labels else None
    if all_labels is None:
        print("No gate values collected. Check that modulator is active."); sys.exit(1)

    gate_values_per_block = {}
    with torch.no_grad():
        all_ws = []
        for x, y in DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True):
            x = x.to(device)
            captured = []

            def make_capture(idx):
                def hook(module, inp, out):
                    captured.append(out.detach().cpu())
                return hook

            hs = [block.modulator.register_forward_hook(make_capture(i)) for i, block in enumerate(model.blocks)]
            model(x, return_local_losses=True)
            for h2 in hs:
                h2.remove()

            if captured:
                stacked = torch.stack(captured, dim=0).mean(dim=0)
                all_ws.append(stacked.numpy())

    all_ws_arr = np.concatenate(all_ws, axis=0)
    all_labels_arr = np.concatenate(
        [y.cpu().numpy() for _, y in DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)],
        axis=0
    )

    ictal_mask = all_labels_arr == 1
    interictal_mask = ~ictal_mask

    print(f"Ictal windows: {ictal_mask.sum()}, Interictal windows: {interictal_mask.sum()}")
    print(f"Mean gate (ictal):      {all_ws_arr[ictal_mask].mean():.4f} ± {all_ws_arr[ictal_mask].std():.4f}")
    print(f"Mean gate (interictal): {all_ws_arr[interictal_mask].mean():.4f} ± {all_ws_arr[interictal_mask].std():.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LP-SSM-EEG: Modulator Gate Value Distribution\n(ictal_ratio, test set = chb04 + chb10)", fontsize=12)

    ax1.hist(all_ws_arr[interictal_mask], bins=50, alpha=0.6, color="#4878CF",
             label=f"Interictal (n={interictal_mask.sum():,})", density=True)
    ax1.hist(all_ws_arr[ictal_mask], bins=30, alpha=0.7, color="#E24A33",
             label=f"Ictal (n={ictal_mask.sum():,})", density=True)
    ax1.set_xlabel("Modulator gate weight w")
    ax1.set_ylabel("Density")
    ax1.set_title("Gate Value Distributions (Density)")
    ax1.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="w=1 (identity)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0.4, 2.1)

    mu_i  = all_ws_arr[ictal_mask].mean()
    mu_ii = all_ws_arr[interictal_mask].mean()
    std_i  = all_ws_arr[ictal_mask].std()
    std_ii = all_ws_arr[interictal_mask].std()

    ax2.bar(["Interictal", "Ictal"], [mu_ii, mu_i], yerr=[std_ii, std_i],
            color=["#4878CF", "#E24A33"], alpha=0.8, capsize=6, width=0.4)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Mean gate weight w")
    ax2.set_title("Mean Gate Weight (mean ± std)")
    ax2.set_ylim(0.4, 2.1)
    for i, (v, e) in enumerate(zip([mu_ii, mu_i], [std_ii, std_i])):
        ax2.text(i, v + e + 0.03, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig4_modulator_gates.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out_path}")

    stats_path = out_dir / "fig4_modulator_gate_stats.txt"
    with open(str(stats_path), "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test set: chb04 + chb10\n")
        f.write(f"Ictal windows:      {ictal_mask.sum()}\n")
        f.write(f"Interictal windows: {interictal_mask.sum()}\n\n")
        f.write(f"Mean gate (ictal):      {mu_i:.4f} ± {std_i:.4f}\n")
        f.write(f"Mean gate (interictal): {mu_ii:.4f} ± {std_ii:.4f}\n")
        f.write(f"Δ (ictal - interictal): {mu_i - mu_ii:+.4f}\n")
        f.write(f"\nInterpretation:\n")
        f.write(f"  w > 1.0: modulator amplifies local loss (training focuses more on these windows)\n")
        f.write(f"  w < 1.0: modulator suppresses local loss\n")
        f.write(f"  Ictal windows should have higher w if modulator correctly identifies seizures.\n")
    print(f"Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
