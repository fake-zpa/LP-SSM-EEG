"""
Process LOOCV results from docs/loocv_results.json and update
docs/RESULTS_SUMMARY.md and docs/PAPER_DRAFT.md with the summary table.

Usage:
    conda run -n mamba2 python scripts/process_loocv_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "loocv_results.json"


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        print(f"Results not found: {RESULTS_PATH}")
        sys.exit(1)
    with open(str(RESULTS_PATH)) as f:
        return json.load(f)


def format_summary_table(data: dict) -> str:
    """Generate markdown table of per-fold and aggregated results."""
    models = list(data.keys())
    fold_subjects = None

    lines = []
    lines.append("**Table: 7-Fold Leave-One-Out Cross-Validation (all CHB-MIT patients with seizures)**\n")
    lines.append("| Test Patient | " + " | ".join(f"{m} AUROC | {m} AUPRC" for m in models) + " |")
    lines.append("|" + "--|" * (1 + 2 * len(models)))

    # Collect per-fold data aligned by subject
    fold_data_by_subject = {}
    for model in models:
        for fold in data[model].get("per_fold", []):
            subj = fold["test_subject"]
            if subj not in fold_data_by_subject:
                fold_data_by_subject[subj] = {}
            fold_data_by_subject[subj][model] = {
                "auroc": fold.get("val_auroc", float("nan")),
                "auprc": fold.get("val_auprc", float("nan")),
            }

    for subj in sorted(fold_data_by_subject.keys()):
        row = f"| {subj} |"
        for model in models:
            d = fold_data_by_subject[subj].get(model, {})
            auroc = d.get("auroc", float("nan"))
            auprc = d.get("auprc", float("nan"))
            row += f" {auroc:.4f} | {auprc:.4f} |"
        lines.append(row)

    lines.append("|---|" * (1 + 2 * len(models)))
    for model in models:
        n_folds = data[model].get("n_folds", 0)
        auroc_mean = data[model].get("auroc_mean", float("nan"))
        auroc_std = data[model].get("auroc_std", float("nan"))
        auprc_mean = data[model].get("auprc_mean", float("nan"))
        auprc_std = data[model].get("auprc_std", float("nan"))
        lines.append(f"| **{model} (n={n_folds}) mean±std** | {auroc_mean:.4f}±{auroc_std:.4f} | {auprc_mean:.4f}±{auprc_std:.4f} |")

    lines.append("")
    lines.append("*Note: LOOCV uses warmup=10 (default in amp_single_4090.yaml). v2 adds min_epochs=warmup_epochs guard to prevent early stopping before warmup completes.*")
    lines.append("*Evaluation key 'val_auroc' = test-split AUROC (naming artifact from evaluate CLI).*")

    return "\n".join(lines)


def compute_delta(data: dict) -> dict:
    """Compute per-fold delta between LP-SSM and Mamba."""
    if "mamba_baseline" not in data or "lp_ssm_eeg" not in data:
        return {}

    mamba_folds = {f["test_subject"]: f for f in data["mamba_baseline"].get("per_fold", [])}
    lp_folds = {f["test_subject"]: f for f in data["lp_ssm_eeg"].get("per_fold", [])}
    common = set(mamba_folds.keys()) & set(lp_folds.keys())

    deltas_auroc = []
    deltas_auprc = []
    for subj in common:
        da = lp_folds[subj].get("val_auroc", 0) - mamba_folds[subj].get("val_auroc", 0)
        dp = lp_folds[subj].get("val_auprc", 0) - mamba_folds[subj].get("val_auprc", 0)
        deltas_auroc.append(da)
        deltas_auprc.append(dp)

    return {
        "mean_delta_auroc": float(np.mean(deltas_auroc)) if deltas_auroc else float("nan"),
        "std_delta_auroc": float(np.std(deltas_auroc)) if deltas_auroc else float("nan"),
        "mean_delta_auprc": float(np.mean(deltas_auprc)) if deltas_auprc else float("nan"),
        "std_delta_auprc": float(np.std(deltas_auprc)) if deltas_auprc else float("nan"),
        "n_matched": len(common),
        "lp_wins_auroc": sum(d > 0 for d in deltas_auroc),
        "lp_wins_auprc": sum(d > 0 for d in deltas_auprc),
    }


def print_summary(data: dict):
    print("\n" + "=" * 70)
    print("LOOCV RESULTS SUMMARY")
    print("=" * 70)

    for model in data:
        print(f"\n{model} ({data[model].get('n_folds', 0)} folds):")
        print(f"  AUROC: {data[model]['auroc_mean']:.4f} ± {data[model]['auroc_std']:.4f}")
        print(f"  AUPRC: {data[model]['auprc_mean']:.4f} ± {data[model]['auprc_std']:.4f}")
        for fold in data[model].get("per_fold", []):
            print(f"    {fold['test_subject']}: AUROC={fold.get('val_auroc',0):.4f}  AUPRC={fold.get('val_auprc',0):.4f}")

    deltas = compute_delta(data)
    if deltas:
        print(f"\nLP-SSM vs Mamba (matched folds, n={deltas['n_matched']}):")
        print(f"  ΔAUROC: {deltas['mean_delta_auroc']:+.4f} ± {deltas['std_delta_auroc']:.4f}")
        print(f"  ΔAUPRC: {deltas['mean_delta_auprc']:+.4f} ± {deltas['std_delta_auprc']:.4f}")
        print(f"  LP-SSM wins AUROC: {deltas['lp_wins_auroc']}/{deltas['n_matched']} folds")
        print(f"  LP-SSM wins AUPRC: {deltas['lp_wins_auprc']}/{deltas['n_matched']} folds")

    print()
    print(format_summary_table(data))


def build_paper_table(data: dict) -> str:
    """Build a compact per-fold + summary table for §4.8."""
    mamba = data.get("mamba_baseline", {})
    lp = data.get("lp_ssm_eeg", {})
    mamba_folds = {f["test_subject"]: f for f in mamba.get("per_fold", [])}
    lp_folds = {f["test_subject"]: f for f in lp.get("per_fold", [])}
    subjects = sorted(set(mamba_folds) | set(lp_folds))

    lines = []
    lines.append("| Test Patient | Ictal N | Mamba AUROC | Mamba AUPRC | LP-SSM AUROC | LP-SSM AUPRC | ΔAUROC |")
    lines.append("|-------------|---------|-------------|-------------|-------------|-------------|--------|")

    ictal_counts = {"chb01": 35, "chb02": 88, "chb03": 206, "chb04": 191, "chb05": 281, "chb06": 31, "chb10": 60}
    for subj in subjects:
        m = mamba_folds.get(subj, {})
        l = lp_folds.get(subj, {})
        ma, mp = m.get("val_auroc", float("nan")), m.get("val_auprc", float("nan"))
        la, lp_ = l.get("val_auroc", float("nan")), l.get("val_auprc", float("nan"))
        delta = la - ma if not (np.isnan(la) or np.isnan(ma)) else float("nan")
        n = ictal_counts.get(subj, "?")
        winner = "**" if delta > 0 else ""
        lines.append(f"| {subj} | {n} | {ma:.4f} | {mp:.4f} | {winner}{la:.4f}{winner} | {lp_:.4f} | {delta:+.4f} |")

    deltas = compute_delta(data)
    n_folds = deltas.get("n_matched", 0)
    lines.append(f"| **7-fold mean±std** | — | {mamba.get('auroc_mean', 0):.4f}±{mamba.get('auroc_std', 0):.4f} | {mamba.get('auprc_mean', 0):.4f}±{mamba.get('auprc_std', 0):.4f} | {lp.get('auroc_mean', 0):.4f}±{lp.get('auroc_std', 0):.4f} | {lp.get('auprc_mean', 0):.4f}±{lp.get('auprc_std', 0):.4f} | {deltas.get('mean_delta_auroc', 0):+.4f} |")
    return "\n".join(lines)


def auto_update_paper(data: dict, version: str = "v2"):
    """Update §4.8 LOOCV table in PAPER_DRAFT.md — handles both first-time insert and v2 update."""
    draft_path = PROJECT_ROOT / "docs" / "PAPER_DRAFT.md"
    with open(str(draft_path)) as f:
        content = f.read()

    deltas = compute_delta(data)
    mamba = data.get("mamba_baseline", {})
    lp = data.get("lp_ssm_eeg", {})
    n_folds = deltas.get("n_matched", 0)
    lp_wins_a = deltas.get("lp_wins_auroc", 0)
    lp_wins_p = deltas.get("lp_wins_auprc", 0)

    table = build_paper_table(data)
    discussion = (
        f"\n\nOver {n_folds} matched patient folds, LP-SSM ictal-ratio achieves "
        f"AUROC {lp.get('auroc_mean',0):.4f}±{lp.get('auroc_std',0):.4f} vs Mamba "
        f"{mamba.get('auroc_mean',0):.4f}±{mamba.get('auroc_std',0):.4f} "
        f"(ΔAUROC={deltas.get('mean_delta_auroc',0):+.4f}). "
        f"LP-SSM wins AUROC on {lp_wins_a}/{n_folds} folds and AUPRC on {lp_wins_p}/{n_folds} folds. "
        f"Mean per-patient AUPRC: LP-SSM {lp.get('auprc_mean',0):.4f}±{lp.get('auprc_std',0):.4f} vs "
        f"Mamba {mamba.get('auprc_mean',0):.4f}±{mamba.get('auprc_std',0):.4f}."
    )

    if "[TABLE PENDING" in content:
        # First-time insert
        old_block = "**[TABLE PENDING — results in `docs/loocv_results.json` when LOOCV completes]**"
        new_block = table + discussion
        content = content.replace(old_block, new_block)
        print(f"PAPER_DRAFT.md §4.8 updated with LOOCV results ({version})")
    else:
        # v2 update: replace the existing table (| Test Patient | ... header line through 7-fold mean row)
        import re
        # Find and replace the main LOOCV results table + summary line
        old_table_pattern = re.compile(
            r"\| Test Patient \| Ictal N \|.*?\| \*\*7-fold mean±std\*\*.*?\|[^\n]*\n",
            re.DOTALL
        )
        new_block = table + "\n"
        content, n_subs = old_table_pattern.subn(new_block, content, count=1)
        # Also replace the 'Over N matched patient folds...' summary line
        old_summary_pattern = re.compile(
            r"Over \d+ matched patient folds, LP-SSM ictal-ratio achieves.*?Mamba [0-9.]+±[0-9.]+\.\n",
            re.DOTALL
        )
        new_summary = discussion.lstrip("\n") + "\n"
        content, n_subs2 = old_summary_pattern.subn(new_summary, content, count=1)
        # Update the 'running' note to 'complete'
        content = content.replace(
            "*Note: LOOCV v2 (with min_epochs=10 guard) is running. Results will supersede this section.*",
            f"*Note: LOOCV {version} (with min_epochs=warmup_epochs=10 guard) complete. warmup=10 confirmed used (amp_single_4090.yaml).*"
        )
        print(f"PAPER_DRAFT.md §4.8 updated with LOOCV {version} results (table replaced, n_subs={n_subs})")

    with open(str(draft_path), "w") as f:
        f.write(content)


def auto_update_results_summary(data: dict):
    """Add LOOCV section to RESULTS_SUMMARY.md."""
    rs_path = PROJECT_ROOT / "docs" / "RESULTS_SUMMARY.md"
    with open(str(rs_path)) as f:
        content = f.read()

    if "### LOOCV: 7-Fold Leave-One-Out" in content:
        print("RESULTS_SUMMARY.md: LOOCV section already present")
        return

    deltas = compute_delta(data)
    mamba = data.get("mamba_baseline", {})
    lp = data.get("lp_ssm_eeg", {})
    table = build_paper_table(data)

    section = f"""

### LOOCV: 7-Fold Leave-One-Out Cross-Validation — 2026-03-31

Both models trained with seed=42, warmup=10, max_epochs=70. Each fold: 1 test patient, 1 val, 5 train.

{table}

**Key findings:**
- LP-SSM ΔAUROC (7-fold mean): {deltas.get('mean_delta_auroc', 0):+.4f}
- LP-SSM wins AUROC: {deltas.get('lp_wins_auroc', 0)}/{deltas.get('n_matched', 0)} folds
- LP-SSM wins AUPRC: {deltas.get('lp_wins_auprc', 0)}/{deltas.get('n_matched', 0)} folds
- LP-SSM mean AUPRC: {lp.get('auprc_mean', 0):.4f}±{lp.get('auprc_std', 0):.4f} vs Mamba {mamba.get('auprc_mean', 0):.4f}±{mamba.get('auprc_std', 0):.4f}

"""
    # Insert before the CHB-MIT archived section
    insert_before = "### CHB-MIT — 4-Subject Results (archived, test=chb03)"
    content = content.replace(insert_before, section + insert_before)
    with open(str(rs_path), "w") as f:
        f.write(content)
    print(f"RESULTS_SUMMARY.md updated with LOOCV section")


def main():
    data = load_results()
    print_summary(data)

    # Save markdown table to docs
    table = format_summary_table(data)
    out_path = PROJECT_ROOT / "docs" / "loocv_table.md"
    with open(str(out_path), "w") as f:
        f.write(f"# 7-Fold LOOCV Results\n\n")
        f.write(f"Generated from: `docs/loocv_results.json`\n\n")
        f.write(table + "\n")
    print(f"\nMarkdown table saved: {out_path}")

    # Auto-update paper documents
    auto_update_paper(data)
    auto_update_results_summary(data)
    print("\nAll documents updated. Run 'git add -A && git commit' to save.")


if __name__ == "__main__":
    main()
