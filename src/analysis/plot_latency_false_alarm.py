"""Plot detection latency and false alarm rate (CHB-MIT event-level metrics)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    data = {}
    for f in Path(args.metrics_dir).rglob("test_metrics.json"):
        run_id = f.parent.name
        with open(f) as fh:
            m = json.load(fh)
        fa = m.get("test_false_alarms_per_hour")
        sens = m.get("test_event_sensitivity")
        if fa is not None and sens is not None:
            data[run_id] = (fa, sens)

    if not data:
        print("No event-level metrics found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for run_id, (fa, sens) in data.items():
        ax.scatter(fa, sens, label=run_id[:25], s=80)

    ax.set_xlabel("False Alarms per Hour")
    ax.set_ylabel("Event Sensitivity")
    ax.set_title("Detection Performance (Sensitivity vs FA Rate)")
    ax.legend(fontsize=8)
    out = Path(args.output_dir) / "latency_false_alarm.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
