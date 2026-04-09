"""Plot training/validation learning curves from epoch_metrics.jsonl files."""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_metrics(log_dir: str):
    curves = {}
    for f in Path(log_dir).rglob("epoch_metrics.jsonl"):
        run_id = f.parent.parent.name
        epochs, losses, aurocs = [], [], []
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                epochs.append(d.get("epoch", 0))
                losses.append(d.get("train_loss", 0))
                aurocs.append(d.get("val_auroc", 0))
        curves[run_id] = {"epochs": epochs, "loss": losses, "auroc": aurocs}
    return curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs/train")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    curves = load_metrics(args.logs_dir)
    if not curves:
        print("No epoch metrics found.")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for run_id, c in curves.items():
        axes[0].plot(c["epochs"], c["loss"], label=run_id[:30])
        axes[1].plot(c["epochs"], c["auroc"], label=run_id[:30])

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    out = Path(args.output_dir) / "learning_curves.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
