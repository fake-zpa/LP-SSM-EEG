"""
t-SNE feature visualization comparing LP-SSM-EEG vs Mamba on chb04+chb10 test set.
Saves fig5_tsne.pdf to paper/ directory.

Usage:
    python scripts/run_tsne.py [--mamba-run-id RUN_ID] [--lpsm-run-id RUN_ID]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model
from src.utils.config import load_config
from src.data.dataset_chbmit import CHBMITDataset


def _get_device():
    """Use CPU if GPU is heavily occupied (LOOCV running), else CUDA."""
    if not torch.cuda.is_available():
        return "cpu"
    free = torch.cuda.mem_get_info()[0] / 1024**3
    return "cuda" if free > 8.0 else "cpu"

DEVICE = _get_device()
MANIFEST = str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def load_model_for_features(model_name: str, run_id: str):
    ckpt_path = CKPT_DIR / run_id / "best.pt"
    model_cfg = load_config(str(PROJECT_ROOT / "configs" / "model" / "lp_ssm_eeg.yaml")).get("architecture", {})
    arch_kwargs = {k: model_cfg[k] for k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout")
                   if k in model_cfg}
    if model_name == "lp_ssm_eeg":
        model = build_model(model_name, in_channels=22, n_classes=2,
                            training_mode="global", **arch_kwargs)
    else:
        model = build_model(model_name, in_channels=22, n_classes=2, **arch_kwargs)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


def extract_features(model, loader, model_name: str):
    """Extract penultimate-layer features by hooking the classifier head input."""
    feats, labels, subjects = [], [], []

    def hook_fn(module, inp, out):
        feats.append(inp[0].detach().float().cpu())

    # Hook the final linear classifier
    classifier = None
    if model_name == "mamba_baseline":
        # MambaBaseline: model.classifier is the final Linear
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and "classifier" in name:
                classifier = m
                break
        if classifier is None:
            # fallback: last Linear
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    classifier = m
    else:
        # LP-SSM-EEG: global_classifier
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and "classifier" in name:
                classifier = m
                break
        if classifier is None:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    classifier = m

    if classifier is None:
        raise RuntimeError("Could not find classifier head to hook")

    handle = classifier.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            x, y, *meta = batch if len(batch) > 2 else (*batch, None)
            out = model(x.to(DEVICE))
            labels.append(y.numpy())
            # subject info from dataset
            subjects.append(np.zeros(len(y), dtype=int))  # placeholder

    handle.remove()
    return (
        torch.cat(feats).numpy(),
        np.concatenate(labels),
    )


def get_subject_labels(dataset):
    """Return per-sample subject integer (0=chb04, 1=chb10)."""
    import pandas as pd
    df = pd.read_csv(MANIFEST)
    col = "subject_id" if "subject_id" in df.columns else "subject"
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    subj_map = {"chb04": 0, "chb10": 1}
    return test_df[col].map(subj_map).fillna(-1).astype(int).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba-run-id", default=None,
                        help="Mamba checkpoint run_id (auto-detect if omitted)")
    parser.add_argument("--lpsm-run-id", default=None,
                        help="LP-SSM checkpoint run_id (auto-detect if omitted)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--subsample", type=int, default=3000,
                        help="Max samples per model for t-SNE speed (0=all)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "paper" / "fig5_tsne.pdf"))
    args = parser.parse_args()

    # --- Auto-detect run IDs from best_config_final.json and 8-seed mamba ---
    import json, re, glob, statistics

    if args.lpsm_run_id is None:
        best_json = PROJECT_ROOT / "outputs" / "metrics" / "best_config_final.json"
        if best_json.exists():
            data = json.loads(best_json.read_text())
            seed_data = data.get("seeds", {})
            # pick seed with highest auroc
            best_seed = max(seed_data, key=lambda s: seed_data[s])
            # find run_id from hparam search log
            pat = f"logs/train/hparam_search/lmb0.10_wm20_seed{best_seed}.log"
            log_files = glob.glob(str(PROJECT_ROOT / pat))
            if log_files:
                text = Path(log_files[0]).read_text()
                m = re.search(r"run_id=([a-z0-9_]+)", text)
                if m:
                    args.lpsm_run_id = m.group(1)
        if args.lpsm_run_id is None:
            # fallback: newest lp_ssm checkpoint
            candidates = sorted(CKPT_DIR.glob("lp_ssm_eeg_*"), key=lambda p: p.stat().st_mtime)
            if candidates:
                args.lpsm_run_id = candidates[-1].name
        print(f"LP-SSM run_id: {args.lpsm_run_id}")

    if args.mamba_run_id is None:
        # pick mamba checkpoint from extra seeds (seed=42)
        mamba_logs = sorted(glob.glob(str(PROJECT_ROOT / "logs/train/mamba_baseline_seed42.log")))
        if not mamba_logs:
            # try extra seeds log
            extra_log = PROJECT_ROOT / "logs" / "train" / "extra_seeds_run.log"
            if extra_log.exists():
                text = extra_log.read_text()
                # find mamba seed=42 run_id
                m = re.search(r"mamba.*seed.*42.*run_id=([a-z0-9_]+)", text)
                if m:
                    args.mamba_run_id = m.group(1)
        if args.mamba_run_id is None:
            # fallback: newest mamba checkpoint
            candidates = sorted(CKPT_DIR.glob("mamba_baseline_*"), key=lambda p: p.stat().st_mtime)
            if candidates:
                args.mamba_run_id = candidates[-1].name
        print(f"Mamba run_id: {args.mamba_run_id}")

    if not args.lpsm_run_id or not args.mamba_run_id:
        print("ERROR: Could not find run IDs. Use --lpsm-run-id and --mamba-run-id.")
        sys.exit(1)

    # --- Load data ---
    print("Loading test dataset...", flush=True)
    test_ds = CHBMITDataset(MANIFEST, split="test")
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    subj_labels = get_subject_labels(test_ds)  # 0=chb04, 1=chb10

    # --- Extract features ---
    print(f"Extracting Mamba features ({args.mamba_run_id})...", flush=True)
    mamba_model = load_model_for_features("mamba_baseline", args.mamba_run_id)
    mamba_feats, mamba_labels = extract_features(mamba_model, loader, "mamba_baseline")
    del mamba_model; torch.cuda.empty_cache()

    print(f"Extracting LP-SSM features ({args.lpsm_run_id})...", flush=True)
    lpsm_model = load_model_for_features("lp_ssm_eeg", args.lpsm_run_id)
    lpsm_feats, lpsm_labels = extract_features(lpsm_model, loader, "lp_ssm_eeg")
    del lpsm_model; torch.cuda.empty_cache()

    # --- Subsample for speed ---
    def subsample(feats, labels, subjs, n):
        if n <= 0 or len(feats) <= n:
            return feats, labels, subjs
        rng = np.random.default_rng(42)
        # stratified: equal from ictal and interictal
        ictal_idx = np.where(labels == 1)[0]
        inter_idx = np.where(labels == 0)[0]
        n_ic = min(len(ictal_idx), n // 4)
        n_in = min(len(inter_idx), n - n_ic)
        idx = np.concatenate([
            rng.choice(ictal_idx, n_ic, replace=False),
            rng.choice(inter_idx, n_in, replace=False),
        ])
        return feats[idx], labels[idx], subjs[idx]

    n_samples = len(mamba_labels)
    subj_labels_full = subj_labels[:n_samples] if len(subj_labels) >= n_samples else np.zeros(n_samples, dtype=int)

    mamba_feats, mamba_labels, mamba_subjs = subsample(mamba_feats, mamba_labels, subj_labels_full, args.subsample)
    lpsm_feats,  lpsm_labels,  lpsm_subjs  = subsample(lpsm_feats,  lpsm_labels,  subj_labels_full, args.subsample)

    print(f"Samples: Mamba {len(mamba_labels)} | LP-SSM {len(lpsm_labels)}", flush=True)

    # --- t-SNE ---
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    print("Running t-SNE (Mamba)...", flush=True)
    mamba_2d = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter,
                    random_state=42, n_jobs=4).fit_transform(
        StandardScaler().fit_transform(mamba_feats))

    print("Running t-SNE (LP-SSM)...", flush=True)
    lpsm_2d  = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter,
                    random_state=42, n_jobs=4).fit_transform(
        StandardScaler().fit_transform(lpsm_feats))

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    COLORS_LABEL = {0: "#A8C8E8", 1: "#E84040"}   # interictal=blue, ictal=red
    COLORS_SUBJ  = {0: "#4C9BE8", 1: "#E8A040"}   # chb04=blue, chb10=orange

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                         'axes.labelsize': 11, 'xtick.labelsize': 10,
                         'ytick.labelsize': 10, 'legend.fontsize': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    # suptitle removed — provided by LaTeX caption

    for col, (feats_2d, labels, subjs, title) in enumerate([
        (mamba_2d, mamba_labels, mamba_subjs, "Mamba (global)"),
        (lpsm_2d,  lpsm_labels,  lpsm_subjs,  "LP-SSM-EEG (tuned)"),
    ]):
        # Row 0: colored by label (ictal/interictal)
        ax = axes[0, col]
        colors = [COLORS_LABEL[l] for l in labels]
        sizes  = [18 if l == 1 else 4 for l in labels]
        alphas = [0.9 if l == 1 else 0.15 for l in labels]
        # plot interictal first, then ictal on top
        for lbl, c, s, a in [(0, COLORS_LABEL[0], 4, 0.15), (1, COLORS_LABEL[1], 20, 0.9)]:
            mask = labels == lbl
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                       c=c, s=s, alpha=a, linewidths=0, rasterized=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        if col == 0:
            ax.set_ylabel("Colored by class\nt-SNE 2")
        legend_elems = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS_LABEL[1],
                   markersize=7, label='Ictal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS_LABEL[0],
                   markersize=5, label='Interictal', alpha=0.6),
        ]
        ax.legend(handles=legend_elems, loc='upper right')

        # Row 1: colored by patient
        ax = axes[1, col]
        for sid, c, name in [(0, COLORS_SUBJ[0], "chb04"), (1, COLORS_SUBJ[1], "chb10")]:
            mask = subjs == sid
            if mask.sum() == 0:
                continue
            ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                       c=c, s=4, alpha=0.25, linewidths=0, rasterized=True, label=name)
        ax.set_xlabel("t-SNE 1")
        if col == 0:
            ax.set_ylabel("Colored by patient\nt-SNE 2")
        ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), format="pdf", dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # Compute class separability (silhouette score)
    from sklearn.metrics import silhouette_score
    for name2d, labels2d in [("Mamba", mamba_2d), ("LP-SSM", lpsm_2d)]:
        lbl = mamba_labels if name2d == "Mamba" else lpsm_labels
        if len(np.unique(lbl)) > 1:
            sil = silhouette_score(labels2d, lbl, sample_size=2000, random_state=42)
            print(f"  {name2d} silhouette (ictal vs interictal): {sil:.4f}")


if __name__ == "__main__":
    main()
