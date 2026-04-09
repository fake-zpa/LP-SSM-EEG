"""
Regenerate fig4_per_patient.pdf — per-patient AUROC/AUPRC for primary
split + all 7 LOOCV folds (Mamba vs LP-SSM-EEG).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                     'axes.labelsize': 11, 'xtick.labelsize': 10,
                     'ytick.labelsize': 10, 'legend.fontsize': 10})

OUT = Path("paper")
OUT.mkdir(exist_ok=True)

# ── Primary split per-patient data (from Table 2 in paper) ──────────
primary_subjects = ["chb04", "chb10"]
primary_mamba_auc  = [0.944, 0.820]
primary_mamba_ap   = [0.283, 0.087]
primary_lpsm_auc   = [0.921, 0.845]
primary_lpsm_ap    = [0.317, 0.203]

# ── LOOCV per-fold data ─────────────────────────────────────────────
d = json.load(open("outputs/metrics/loocv_tuned_final.json"))
mamba_folds = d["mamba_baseline"]["mamba_baseline"]["per_fold"]
lpsm_folds  = d["lp_ssm_tuned"]["lp_ssm_eeg"]["per_fold"]

loocv_subjects = [f["test_subject"] for f in mamba_folds]
loocv_m_auc = [f["val_auroc"] for f in mamba_folds]
loocv_m_ap  = [f["val_auprc"] for f in mamba_folds]
loocv_l_auc = [f["val_auroc"] for f in lpsm_folds]
loocv_l_ap  = [f["val_auprc"] for f in lpsm_folds]

# ── Combine: primary + LOOCV ────────────────────────────────────────
all_subjects = primary_subjects + [""] + loocv_subjects  # gap separator
m_auc = primary_mamba_auc + [np.nan] + loocv_m_auc
m_ap  = primary_mamba_ap  + [np.nan] + loocv_m_ap
l_auc = primary_lpsm_auc  + [np.nan] + loocv_l_auc
l_ap  = primary_lpsm_ap   + [np.nan] + loocv_l_ap

x = np.arange(len(all_subjects))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ── Left: AUROC ─────────────────────────────────────────────────────
ax1.bar(x - width/2, m_auc, width, label="Mamba", color="#4878CF",
        edgecolor="white", linewidth=0.5)
ax1.bar(x + width/2, l_auc, width, label="LP-SSM-EEG", color="#D65F5F",
        edgecolor="white", linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(all_subjects, rotation=30, ha="right")
ax1.set_ylabel("AUROC")
ax1.set_title("Per-Patient AUROC", fontweight="bold")
ax1.set_ylim(0.3, 1.05)
ax1.legend(loc="lower left")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.25)

# Divider between primary and LOOCV
div_x = 2
ax1.axvline(div_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.5)
ax1.text(0.5, 1.02, "Primary", fontsize=11, ha="center",
         transform=ax1.get_xaxis_transform(), color="#555")
ax1.text(5.5, 1.02, "LOOCV folds", fontsize=11, ha="center",
         transform=ax1.get_xaxis_transform(), color="#555")

# ── Right: AUPRC ────────────────────────────────────────────────────
ax2.bar(x - width/2, m_ap, width, label="Mamba", color="#4878CF",
        edgecolor="white", linewidth=0.5)
ax2.bar(x + width/2, l_ap, width, label="LP-SSM-EEG", color="#D65F5F",
        edgecolor="white", linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(all_subjects, rotation=30, ha="right")
ax2.set_ylabel("AUPRC")
ax2.set_title("Per-Patient AUPRC", fontweight="bold")
ax2.set_ylim(0, 0.45)
ax2.legend(loc="upper right")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.25)

ax2.axvline(div_x, color="gray", linestyle=":", linewidth=1.2, alpha=0.5)
ax2.text(0.5, 1.02, "Primary", fontsize=11, ha="center",
         transform=ax2.get_xaxis_transform(), color="#555")
ax2.text(5.5, 1.02, "LOOCV folds", fontsize=11, ha="center",
         transform=ax2.get_xaxis_transform(), color="#555")

plt.tight_layout()
out = OUT / "fig4_per_patient.pdf"
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
