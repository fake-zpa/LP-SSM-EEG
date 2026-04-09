"""Plot ROC and PR curves from saved prediction files."""
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    found = False
    for probs_file in Path(args.predictions_dir).rglob("*_probs.npy"):
        run_id = probs_file.parent.name
        targets_file = probs_file.parent / probs_file.name.replace("probs", "targets")
        if not targets_file.exists():
            continue
        probs = np.load(probs_file)
        targets = np.load(targets_file)
        pos_probs = probs[:, 1] if probs.ndim == 2 else probs

        fpr, tpr, _ = roc_curve(targets, pos_probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f"{run_id[:25]} ({roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(targets, pos_probs)
        pr_auc = auc(rec, prec)
        axes[1].plot(rec, prec, label=f"{run_id[:25]} ({pr_auc:.3f})")
        found = True

    if not found:
        print("No prediction files found.")
        return

    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curves")
    axes[0].legend(fontsize=7)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curves")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    out = Path(args.output_dir) / "roc_pr_curves.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
