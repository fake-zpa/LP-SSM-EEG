"""
Per-band Analysis: Learned Spectral Weight Distribution in LP-SSM-EEG.

Extracts:
  1. Band power distributions (delta/theta/alpha/beta) for ictal vs interictal windows
  2. Ictal-ratio modulator output distribution (ictal vs interictal)
  3. Per-band denoising head MSE (which bands the model tracks)

Outputs:
  - outputs/metrics/band_analysis.json
  - paper/fig6_band_analysis.pdf
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Use a checkpoint that has band_head (trained after architecture upgrade)
LPSSM_CKPT = "outputs/checkpoints/lp_ssm_eeg_20260402_154659/best.pt"
MANIFEST   = "data/manifests/chbmit_manifest.csv"
TEST_SUBS  = ["chb04", "chb10"]
SFREQ      = 256.0
BANDS      = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0),
              ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0)]
BAND_NAMES = [b[0] for b in BANDS]
BAND_COLORS = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]


# ─── Band-power helpers ────────────────────────────────────────────────────
def compute_bandpowers(raw_x: np.ndarray) -> np.ndarray:
    """raw_x: [C, T] → [n_bands]  (log band power, mean over channels)"""
    T = raw_x.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / SFREQ)
    X = np.fft.rfft(raw_x.mean(axis=0))
    psd = np.abs(X) ** 2 / T
    bps = []
    for _, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        bps.append(float(np.log(psd[mask].mean() + 1e-9)))
    return np.array(bps)


def compute_ictal_ratio(raw_x: np.ndarray) -> float:
    """(beta+theta) / (alpha+delta) ratio"""
    T = raw_x.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / SFREQ)
    X = np.fft.rfft(raw_x.mean(axis=0))
    psd = np.abs(X) ** 2 / T
    def bp(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return psd[mask].mean() + 1e-10
    delta = bp(0.5, 4.0);  theta = bp(4.0, 8.0)
    alpha = bp(8.0, 13.0); beta  = bp(13.0, 30.0)
    return float(np.log((beta + theta) / (alpha + delta)))


# ─── Load test dataset ─────────────────────────────────────────────────────
def load_test_windows():
    import pandas as pd
    from src.data.dataset_chbmit import CHBMITDataset

    ds = CHBMITDataset(manifest_path=MANIFEST, split="test",
                       subjects=TEST_SUBS, seed=42,
                       max_neg_pos_ratio=None)
    return ds


# ─── Extract modulator outputs via hook ───────────────────────────────────
def extract_modulator_outputs(model, ds, max_batches=900):
    """Run LP-SSM on batches; capture per-block modulator weights via forward hook."""
    from torch.utils.data import DataLoader
    from src.models.local_modulator_v2 import EEGLocalModulatorV2

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # Collect per-sample mean modulator weight across blocks
    mod_outputs = []
    labels_all  = []
    # block_ws[i] accumulates one tensor per forward call
    captured_blocks = []   # list of per-block capture lists

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, EEGLocalModulatorV2):
            bucket = []
            captured_blocks.append(bucket)
            def _hook(m, inp, out, _b=bucket):
                # out: scalar or [B] or [B,1]
                w = out.detach().cpu()
                if w.dim() == 0:
                    w = w.unsqueeze(0)
                _b.append(w.squeeze(-1) if w.dim() > 1 else w)
            hooks.append(module.register_forward_hook(_hook))

    model.eval()
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1]
            for b in captured_blocks:
                b.clear()
            try:
                _ = model(x, return_local_losses=True)
            except Exception as e:
                print(f"  forward error: {e}")
                break
            if captured_blocks and any(len(b) > 0 for b in captured_blocks):
                # stack: each bucket has one [B] tensor per call
                ws = [b[0] for b in captured_blocks if b]  # list of [B]
                w_mean = torch.stack(ws, dim=0).mean(dim=0)   # [B]
                mod_outputs.append(w_mean.numpy())
                labels_all.append(y.numpy())
            n_batches += 1
            if n_batches >= max_batches:
                break

    for h in hooks:
        h.remove()

    if not mod_outputs:
        print("  No modulator outputs captured - hooks may not have fired")
        return None, None

    mod_arr = np.concatenate(mod_outputs, axis=0)  # [N]
    lbl_arr = np.concatenate(labels_all, axis=0)    # [N]
    return mod_arr, lbl_arr


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading test windows …")
    ds = load_test_windows()
    print(f"  {len(ds)} windows, test subjects: {TEST_SUBS}")

    # 1. Compute band powers & ictal ratio for every window
    band_powers_ictal     = [[] for _ in BANDS]
    band_powers_interictal= [[] for _ in BANDS]
    ratios_ictal          = []
    ratios_interictal     = []

    for i in range(len(ds)):
        item = ds[i]
        x_np = item[0].numpy() if isinstance(item[0], torch.Tensor) else item[0]  # [C, T]
        label = int(item[1])
        bps = compute_bandpowers(x_np)
        rat = compute_ictal_ratio(x_np)
        for b in range(len(BANDS)):
            if label == 1:
                band_powers_ictal[b].append(bps[b])
            else:
                band_powers_interictal[b].append(bps[b])
        (ratios_ictal if label == 1 else ratios_interictal).append(rat)

    n_ictal     = len(ratios_ictal)
    n_interictal= len(ratios_interictal)
    print(f"  ictal={n_ictal}, interictal={n_interictal}")

    # 2. Load LP-SSM model for modulator hook
    print("Loading LP-SSM model …")
    mod_arr = lbl_arr = None
    try:
        from src.models.lp_ssm_eeg import LPSSMEEG
        ckpt  = torch.load(LPSSM_CKPT, map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        cfg_d = ckpt.get("config", {}) or {}
        # Build model with checkpoint architecture (d_model=256, n_layers=6)
        base_kw = {"d_model": 256, "n_layers": 6, "in_channels": 22,
                   "mod_use_band_powers": False, "mod_use_temporal_variance": False,
                   "mod_use_event_uncertainty": False, "mod_use_ictal_ratio": True}
        allowed = LPSSMEEG.__init__.__code__.co_varnames
        init_kw = {k: v for k, v in {**base_kw, **cfg_d}.items() if k in allowed}
        model   = LPSSMEEG(**init_kw)
        missing_r = model.load_state_dict(state, strict=False)
        if missing_r.missing_keys:
            print(f"  Note: {len(missing_r.missing_keys)} missing keys (expected for arch diff)")
        model.to(DEVICE)
        mod_arr, lbl_arr = extract_modulator_outputs(model, ds)
        if mod_arr is not None:
            print(f"  Captured modulator outputs: {mod_arr.shape}")
    except Exception as e:
        print(f"  Warning: could not load model ({e}), skipping modulator analysis")

    # 3. Compute per-band MSE from band head predictions
    band_head_mse_ictal     = None
    band_head_mse_interictal= None
    if mod_arr is not None:
        # Compute from BandSelectiveReconstructionHead predictions
        try:
            from src.models.denoising_head import BandSelectiveReconstructionHead
            loader2 = torch.utils.data.DataLoader(ds, batch_size=64,
                                                  shuffle=False, num_workers=0)
            # Extract band-head MSE via per-block hook on local_head outputs
            from src.models.lp_ssm_eeg import LPSSMBlock
            mses_ictal     = [[] for _ in BANDS]
            mses_interictal= [[] for _ in BANDS]
            # Capture (x_block, h_block) tuples per block via forward hooks
            block_io = {}
            def make_block_hook(bname):
                def _bh(m, inp, out):
                    # inp[0] = x (block input), out = h (block output)
                    block_io[bname] = (
                        inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu(),
                        out[0].detach().cpu() if isinstance(out, tuple) else out.detach().cpu()
                    )
                return _bh
            bk_hooks = []
            first_block_name = None
            for bname, bmod in model.named_modules():
                if isinstance(bmod, LPSSMBlock):
                    if first_block_name is None:
                        first_block_name = bname
                    bk_hooks.append(bmod.register_forward_hook(make_block_hook(bname)))
                    break  # only first block for MSE analysis
            model.eval()
            with torch.no_grad():
                for batch in loader2:
                    xb, yb = batch[0].to(DEVICE), batch[1].numpy()
                    try:
                        fwd = model(xb)
                    except Exception as fe:
                        print(f"  fwd err: {fe}")
                        break
                    if first_block_name not in block_io:
                        continue
                    x_blk, h_blk = block_io.pop(first_block_name)
                    # get band_head from first block
                    first_block_mod = dict(model.named_modules())[first_block_name] if first_block_name else None
                    if first_block_mod is None or not hasattr(first_block_mod, 'local_head'):
                        break
                    bh = first_block_mod.local_head.band_head
                    if bh is None:
                        break
                    h_pool = h_blk.mean(dim=1).to(DEVICE)  # [B, d_model]
                    with torch.no_grad():
                        pred_b = bh.head(h_pool).cpu()      # [B, 4]
                    # Compute targets from raw EEG (x_blk is block input ≈ projected EEG for block 0)
                    # Use original xb (raw EEG) for band power targets
                    xb_cpu = xb.cpu()
                    bp_targets = []
                    for _, lo, hi in BANDS:
                        freqs = torch.fft.rfftfreq(xb_cpu.shape[-1], d=1.0/SFREQ)
                        X = torch.fft.rfft(xb_cpu.mean(dim=1))  # [B, freq]
                        psd = X.abs()**2 / xb_cpu.shape[-1]
                        mask = (freqs >= lo) & (freqs < hi)
                        bp_t = torch.log(psd[:, mask].mean(dim=-1) + 1e-9)  # [B]
                        bp_targets.append(bp_t)
                    target = torch.stack(bp_targets, dim=-1)  # [B, 4]
                    # Normalise before MSE
                    target_n = F.layer_norm(target, target.shape[-1:])
                    pred_n   = F.layer_norm(pred_b, pred_b.shape[-1:])
                    per_band_mse = ((pred_n - target_n) ** 2).mean(0).numpy()  # [4]
                    for idx, lbl in enumerate(yb):
                        for b in range(len(BANDS)):
                            (mses_ictal[b] if lbl == 1 else mses_interictal[b]).append(
                                float(per_band_mse[b]))
            for h in bk_hooks:
                h.remove()
            if any(mses_ictal[0]):
                band_head_mse_ictal     = [float(np.mean(m)) for m in mses_ictal]
                band_head_mse_interictal= [float(np.mean(m)) for m in mses_interictal]
                print(f"  Band-head MSE (ictal):     {band_head_mse_ictal}")
                print(f"  Band-head MSE (interictal):{band_head_mse_interictal}")
        except Exception as e:
            print(f"  Warning: band-head MSE extraction failed ({e})")

    # 4. Save JSON
    results = {
        "band_names": BAND_NAMES,
        "n_ictal": n_ictal,
        "n_interictal": n_interictal,
        "band_power_mean_ictal":      [float(np.mean(bp)) for bp in band_powers_ictal],
        "band_power_mean_interictal": [float(np.mean(bp)) for bp in band_powers_interictal],
        "band_power_std_ictal":       [float(np.std(bp)) for bp in band_powers_ictal],
        "band_power_std_interictal":  [float(np.std(bp)) for bp in band_powers_interictal],
        "ictal_ratio_mean_ictal":     float(np.mean(ratios_ictal)) if ratios_ictal else None,
        "ictal_ratio_mean_interictal":float(np.mean(ratios_interictal)) if ratios_interictal else None,
        "ictal_ratio_std_ictal":      float(np.std(ratios_ictal)) if ratios_ictal else None,
        "ictal_ratio_std_interictal": float(np.std(ratios_interictal)) if ratios_interictal else None,
        "modulator_mean_ictal":       float(np.mean(mod_arr[lbl_arr == 1])) if mod_arr is not None else None,
        "modulator_mean_interictal":  float(np.mean(mod_arr[lbl_arr == 0])) if mod_arr is not None else None,
        "band_head_mse_ictal":        band_head_mse_ictal,
        "band_head_mse_interictal":   band_head_mse_interictal,
    }
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/band_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: outputs/metrics/band_analysis.json")

    # 5. Generate figure
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13,
                          'axes.labelsize': 12, 'xtick.labelsize': 11,
                          'ytick.labelsize': 11, 'legend.fontsize': 10})
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.40)

    # Panel A: per-band power comparison (bar chart)
    ax_a = fig.add_subplot(gs[0, :2])
    x_pos = np.arange(len(BANDS))
    width = 0.35
    mu_i  = results["band_power_mean_ictal"]
    mu_n  = results["band_power_mean_interictal"]
    se_i  = [s / np.sqrt(n_ictal + 1)  for s in results["band_power_std_ictal"]]
    se_n  = [s / np.sqrt(n_interictal + 1) for s in results["band_power_std_interictal"]]
    bars_i = ax_a.bar(x_pos - width/2, mu_i, width, yerr=se_i,
                      label="Ictal", color="#d62728", alpha=0.8, capsize=4)
    bars_n = ax_a.bar(x_pos + width/2, mu_n, width, yerr=se_n,
                      label="Interictal", color="#1f77b4", alpha=0.8, capsize=4)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([f"{b[0].capitalize()}\n({b[1]}-{b[2]} Hz)" for b in BANDS])
    ax_a.set_ylabel("Log Band Power (dB)")
    ax_a.set_title("(A) Per-Band Power: Ictal vs. Interictal", fontsize=14, fontweight='bold')
    ax_a.legend()
    ax_a.grid(axis="y", alpha=0.3)

    # Panel B: ictal ratio distribution
    ax_b = fig.add_subplot(gs[0, 2])
    if ratios_ictal and ratios_interictal:
        rat_i_clamp = np.clip(ratios_ictal,     np.percentile(ratios_ictal, 2),
                              np.percentile(ratios_ictal, 98))
        rat_n_clamp = np.clip(ratios_interictal, np.percentile(ratios_interictal, 2),
                              np.percentile(ratios_interictal, 98))
        ax_b.hist(rat_n_clamp, bins=40, density=True, alpha=0.6, color="#1f77b4", label="Interictal")
        ax_b.hist(rat_i_clamp, bins=40, density=True, alpha=0.7, color="#d62728", label="Ictal")
        ax_b.axvline(np.mean(ratios_ictal),     color="#d62728", ls="--", lw=1.5)
        ax_b.axvline(np.mean(ratios_interictal), color="#1f77b4", ls="--", lw=1.5)
    ax_b.set_xlabel("log[(β+θ)/(α+δ)]")
    ax_b.set_ylabel("Density")
    ax_b.set_title("(B) Ictal Ratio Distribution", fontsize=14, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(alpha=0.3)

    # Panel C: modulator weight distribution
    ax_c = fig.add_subplot(gs[1, :2])
    if mod_arr is not None:
        mod_i = mod_arr[lbl_arr == 1].flatten()
        mod_n = mod_arr[lbl_arr == 0].flatten()
        ax_c.hist(mod_n, bins=50, density=True, alpha=0.6, color="#1f77b4", label="Interictal")
        ax_c.hist(mod_i, bins=50, density=True, alpha=0.7, color="#d62728", label="Ictal")
        ax_c.axvline(np.mean(mod_i), color="#d62728", ls="--", lw=1.5,
                     label=f"Ictal mean={np.mean(mod_i):.3f}")
        ax_c.axvline(np.mean(mod_n), color="#1f77b4", ls="--", lw=1.5,
                     label=f"Interictal mean={np.mean(mod_n):.3f}")
    else:
        ax_c.text(0.5, 0.5, "Model not available", ha="center", va="center",
                  transform=ax_c.transAxes)
    ax_c.set_xlabel("Modulator Weight")
    ax_c.set_ylabel("Density")
    ax_c.set_title("(C) Modulator Weight", fontsize=14, fontweight='bold')
    ax_c.legend(fontsize=10)
    ax_c.grid(alpha=0.3)

    # Panel D: band-head MSE comparison
    ax_d = fig.add_subplot(gs[1, 2])
    if band_head_mse_ictal and band_head_mse_interictal:
        x_pos2 = np.arange(len(BANDS))
        ax_d.bar(x_pos2 - width/2, band_head_mse_ictal,     width,
                 label="Ictal",      color="#d62728", alpha=0.8)
        ax_d.bar(x_pos2 + width/2,  band_head_mse_interictal, width,
                 label="Interictal", color="#1f77b4", alpha=0.8)
        ax_d.set_xticks(x_pos2)
        ax_d.set_xticklabels([b[0].capitalize() for b in BANDS])
        ax_d.set_ylabel("MSE (norm. pred)")
        ax_d.set_title("(D) Band-Head Recon MSE", fontsize=14, fontweight='bold')
        ax_d.legend(fontsize=10)
        ax_d.grid(axis="y", alpha=0.3)
    else:
        ax_d.text(0.5, 0.5, "Band-head data\nunavailable", ha="center", va="center",
                  transform=ax_d.transAxes)
        ax_d.set_title("(D) Band-Head Recon MSE", fontsize=14, fontweight='bold')

    # suptitle removed — provided by LaTeX caption

    os.makedirs("paper", exist_ok=True)
    fig.savefig("paper/fig6_band_analysis.pdf", bbox_inches="tight")
    fig.savefig("docs/figures/fig6_band_analysis.png", bbox_inches="tight", dpi=150)
    print("Saved: paper/fig6_band_analysis.pdf")
    plt.close(fig)

    # Print summary
    print("\n=== Band Analysis Summary ===")
    for b_idx, (bname, lo, hi) in enumerate(BANDS):
        d_i = results["band_power_mean_ictal"][b_idx]
        d_n = results["band_power_mean_interictal"][b_idx]
        print(f"  {bname:6s} ({lo:4.1f}-{hi:4.1f} Hz): ictal={d_i:.3f}  interictal={d_n:.3f}  diff={d_i-d_n:+.3f}")
    if results["ictal_ratio_mean_ictal"]:
        print(f"  Ictal ratio: ictal={results['ictal_ratio_mean_ictal']:.3f} "
              f"interictal={results['ictal_ratio_mean_interictal']:.3f}")
    if results["modulator_mean_ictal"]:
        print(f"  Modulator weight: ictal={results['modulator_mean_ictal']:.4f} "
              f"interictal={results['modulator_mean_interictal']:.4f}")


if __name__ == "__main__":
    main()
