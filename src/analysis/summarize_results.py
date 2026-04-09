"""Aggregate and tabulate all experiment results from metrics directories."""
import json
import argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd


def collect_metrics(metrics_dir: str) -> List[Dict]:
    rows = []
    for summary_file in Path(metrics_dir).rglob("training_summary.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            run_id = data.get("run_id", summary_file.parent.name)
            best = data.get("best_metrics", {})
            row = {
                "run_id": run_id,
                "best_val_metric": data.get("best_val_metric"),
                "total_epochs": data.get("total_epochs"),
                "training_time_min": data.get("training_time_min"),
            }
            for k, v in best.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Could not read {summary_file}: {e}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    rows = collect_metrics(args.metrics_dir)
    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / "all_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.to_string())


if __name__ == "__main__":
    main()
