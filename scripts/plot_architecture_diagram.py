"""Generate a clean LP-SSM-EEG architecture diagram (simplified, for main text)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, label, fc, ec="#333333", ls="-", lw=1.2, fs=9, tc="#222222"):
    """Draw a rounded rectangle with centred label."""
    p = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.3)

def harrow(ax, x1, x2, y, color="#333333", lw=1.0, ls="-"):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls))

def varrow(ax, x, y1, y2, color="#333333", lw=1.0, ls="-"):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls))

# ── figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 4.0))
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.04, 1.06)
ax.axis("off")

# colours
C_MAIN  = "#E8F0FE"   # light blue – inference path
C_TRAIN = "#FFF0F0"   # light red  – train-only
EC_TR   = "#CC4444"    # red edge

# ── row 1 (y≈0.30): inference pipeline ──────────────────────────────
bw, bh = 0.14, 0.18
gap = 0.025
y_main = 0.30

xs = {}  # store x-centres for arrows
# 1. Input
x = 0.02;  rbox(ax, x, y_main, bw, bh, "EEG window\n(4 s × 23 ch)", C_MAIN); xs["in"] = x + bw
# 2. Projection
x = 0.19;  rbox(ax, x, y_main, bw, bh, "Input\nprojection", C_MAIN); xs["proj_l"] = x; xs["proj_r"] = x + bw
# 3. Mamba backbone (wider)
bw2 = 0.24
x = 0.37;  rbox(ax, x, y_main, bw2, bh, "×L  Mamba SSM\nblocks (residual)", C_MAIN, fs=9); xs["bb_l"] = x; xs["bb_r"] = x + bw2; xs["bb_cx"] = x + bw2 / 2
# 4. Global classifier
x = 0.65;  rbox(ax, x, y_main, bw, bh, "LayerNorm\n+ classifier", C_MAIN); xs["cls_l"] = x; xs["cls_r"] = x + bw
# 5. Output
x = 0.83;  rbox(ax, x, y_main, bw, bh, "Seizure\nprobability", "#F0F8E8", fs=9); xs["out_l"] = x

# horizontal arrows
ym = y_main + bh / 2
harrow(ax, xs["in"] + gap/2, xs["proj_l"] - gap/2, ym)
harrow(ax, xs["proj_r"] + gap/2, xs["bb_l"] - gap/2, ym)
harrow(ax, xs["bb_r"] + gap/2, xs["cls_l"] - gap/2, ym)
harrow(ax, xs["cls_r"] + gap/2, xs["out_l"] - gap/2, ym)

# ── row 2 (y≈0.68): train-only components ──────────────────────────
y_train = 0.70
tw = 0.56
tx = 0.22

# Dashed container
container = mpatches.FancyBboxPatch(
    (tx - 0.02, y_train - 0.02), tw + 0.04, 0.28,
    boxstyle="round,pad=0.01,rounding_size=0.02",
    facecolor=C_TRAIN, edgecolor=EC_TR, linewidth=1.3, linestyle="--",
)
ax.add_patch(container)

# Banner text
ax.text(tx + tw / 2, y_train + 0.28,
        "TRAIN-ONLY  (discarded at inference)",
        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=EC_TR)

# Sub-boxes inside train container
sbw, sbh = 0.24, 0.14
rbox(ax, tx, y_train, sbw, sbh,
     "Spectral modulator\n(band powers, ictal ratio → w)",
     "#FFFFFF", ec=EC_TR, ls="--", fs=8, tc="#663333")

rbox(ax, tx + sbw + 0.06, y_train, sbw, sbh,
     "Local denoising head\n(TF recon + band targets)",
     "#FFFFFF", ec=EC_TR, ls="--", fs=8, tc="#663333")

# ── vertical arrows: backbone ↔ train-only ─────────────────────────
# backbone top → modulator bottom
varrow(ax, xs["bb_cx"] - 0.04, y_main + bh + 0.02, y_train - 0.04, color=EC_TR, ls="--")
# backbone top → local head bottom
varrow(ax, xs["bb_cx"] + 0.12, y_main + bh + 0.02, y_train - 0.04, color=EC_TR, ls="--")

# loss labels
ax.text(xs["bb_cx"] - 0.07, (y_main + bh + y_train) / 2 + 0.01,
        r"$h_\ell$", fontsize=9, color=EC_TR, ha="center")
ax.text(xs["bb_cx"] + 0.16, (y_main + bh + y_train) / 2 + 0.01,
        r"$h_\ell$", fontsize=9, color=EC_TR, ha="center")

# loss annotation near local head
ax.text(tx + sbw + 0.06 + sbw + 0.025, y_train + sbh / 2,
        r"$\mathcal{L}_{\mathrm{local}}$",
        fontsize=11, color=EC_TR, ha="left", va="center")

# ── bottom note ─────────────────────────────────────────────────────
ax.text(0.50, -0.02,
        "Solid blue = inference path (2.64 M params, 1.31 ms/window);  "
        "Dashed red = train-only auxiliary heads.",
        ha="center", va="top", fontsize=8, color="#555555")

plt.tight_layout()
out_pdf = OUT / "fig0_architecture.pdf"
out_png = OUT / "fig0_architecture.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
