"""
Generate two supplemental figures:
  fig8a_loocv_bars.pdf   -- per-fold LOOCV AUROC bar chart (Mamba vs LP-SSM)
  fig8b_learning_curve.pdf -- N training subjects vs test AUROC learning curve
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ─── FIGURE 1: per-fold LOOCV AUROC ────────────────────────────────────────
# Use TUNED LOOCV (lambda=0.10, warmup=20) to match paper Table (tab:loocv)
d = json.load(open("outputs/metrics/loocv_tuned_final.json"))
mamba_folds = d["mamba_baseline"]["mamba_baseline"]["per_fold"]
lpsm_folds  = d["lp_ssm_tuned"]["lp_ssm_eeg"]["per_fold"]

subjects  = [f["test_subject"] for f in mamba_folds]
m_aurocs  = [f["val_auroc"]    for f in mamba_folds]
l_aurocs  = [f["val_auroc"]    for f in lpsm_folds]

x     = np.arange(len(subjects))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.2))
bars_m = ax.bar(x - width/2, m_aurocs, width, label="Mamba",
                color="#4878CF", edgecolor="white", linewidth=0.5)
bars_l = ax.bar(x + width/2, l_aurocs, width, label="LP-SSM-EEG",
                color="#D65F5F", edgecolor="white", linewidth=0.5)

m_mean = np.mean(m_aurocs)
l_mean = np.mean(l_aurocs)
ax.axhline(m_mean, color="#4878CF", linestyle="--", linewidth=1.2,
           alpha=0.7, label=f"Mamba mean ({m_mean:.3f})")
ax.axhline(l_mean, color="#D65F5F", linestyle="--", linewidth=1.2,
           alpha=0.7, label=f"LP-SSM mean ({l_mean:.3f})")

ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

# Annotate winner per fold
for xi, (ma, la) in enumerate(zip(m_aurocs, l_aurocs)):
    top = max(ma, la) + 0.035
    if la > ma:
        ax.text(xi + width/2, top, "★", ha="center", va="bottom",
                fontsize=10, color="#D65F5F")

ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=10)
ax.set_xlabel("Test subject (LOOCV fold)", fontsize=11)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_ylim(0.35, 1.05)
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out1 = OUT / "fig8a_loocv_bars.pdf"
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.savefig(str(out1).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ─── FIGURE 2: learning curve (N=1..4, from Tab.4 in paper, 3 seeds) ───────
# Table 4 data: N train | Mamba mean±std | LP-SSM mean±std
N_vals  = [1,     2,     3,     4]
m_mean  = [0.436, 0.760, 0.856, 0.896]
m_std   = [0.033, 0.027, 0.068, 0.017]
l_mean  = [0.360, 0.704, 0.871, 0.890]
l_std   = [0.065, 0.030, 0.032, 0.036]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(N_vals, m_mean, yerr=m_std, fmt="o-", color="#4878CF",
            linewidth=2, markersize=7, capsize=4, label="Mamba")
ax.errorbar(N_vals, l_mean, yerr=l_std, fmt="s-", color="#D65F5F",
            linewidth=2, markersize=7, capsize=4, label="LP-SSM-EEG")

# Mark crossover at N=3
ax.axvline(2.5, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
ax.text(2.55, 0.38, "N<3:\nLP-SSM\n< Mamba", fontsize=7.5, color="#D65F5F",
        style="italic", va="center", alpha=0.8)
ax.text(3.2, 0.38, "N≥3:\nLP-SSM\n≥ Mamba", fontsize=7.5, color="#D65F5F",
        style="italic", va="center", alpha=0.8)

ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

ax.set_xticks(N_vals)
ax.set_xticklabels([f"N={n}" for n in N_vals], fontsize=10)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_xlabel("Number of training subjects", fontsize=11)
ax.set_ylim(0.25, 1.0)
# title removed — provided by LaTeX caption
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Shade regions
ax.fill_betweenx([0.25, 1.0], 0.7, 2.5, color="#D65F5F", alpha=0.04)
ax.fill_betweenx([0.25, 1.0], 2.5, 4.3, color="#4878CF", alpha=0.04)

plt.tight_layout()
out2 = OUT / "fig8b_learning_curve.pdf"
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.savefig(str(out2).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print("\nDone. Both figures ready for paper inclusion.")
