"""Plot confusion matrices from saved prediction files."""
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default="outputs/predictions")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for preds_file in Path(args.predictions_dir).rglob("*_preds.npy"):
        run_id = preds_file.parent.name
        targets_file = preds_file.parent / preds_file.name.replace("preds", "targets")
        if not targets_file.exists():
            continue

        preds = np.load(preds_file)
        targets = np.load(targets_file)
        cm = confusion_matrix(targets, preds)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax)
        ax.set_title(f"Confusion Matrix — {run_id[:30]}")
        out = Path(args.output_dir) / f"cm_{run_id}.pdf"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
