"""Schematic Figures 1 (study pipeline) and 3 (architecture and objectives)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
C = {"data": "#E8F1FA", "model": "#EAF5EA", "sc": "#FDF0E3", "eval": "#F3ECF7", "edge": "#4A4A4A",
     "blue": "#2471A3", "red": "#B03A2E", "grey": "#7F8C8D", "head": "#F9F9F9"}


def box(ax, x, y, w, h, text, fc, fs=8.5, bold_first=False, ec=None, lw=1.0, ls="-", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.8",
                                fc=fc, ec=ec or C["edge"], lw=lw, ls=ls))
    tx = x + w / 2 if ha == "center" else x + 1.2
    if bold_first:
        first, _, rest = text.partition("\n")
        ax.text(tx, y + h - 1.6, first, ha=ha, va="top", fontsize=fs + 0.5, fontweight="bold")
        ax.text(tx, y + h - 4.4, rest, ha=ha, va="top", fontsize=fs, linespacing=1.35)
    else:
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, linespacing=1.3)


def arrow(ax, p, q, color=None, ls="-", lw=1.2, rad=0.0):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color or C["edge"], lw=lw, ls=ls,
                                                    shrinkA=0, shrinkB=0, mutation_scale=11,
                                                    connectionstyle=f"arc3,rad={rad}"))


def figure1():
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 121); ax.set_ylim(0, 54); ax.axis("off")
    for x, t in [(1, "1  Data and partitioning"), (31, "2  Models and training"), (65, "3  Out-of-fold evaluation"), (95, "4  Additional B4 analyses")]:
        ax.text(x, 52.5, t, fontsize=10.5, fontweight="bold", color="#222", va="top")
    box(ax, 1, 36, 26, 13, "ISIC 2018 Task 3 training set\n10,015 dermoscopic images\n7,470 distinct lesions, 7 classes", C["data"], bold_first=True, fs=7.8)
    box(ax, 1, 13, 26, 19, "Lesion-level partitioning\nStratifiedGroupKFold (k = 5)\ngroup = lesion_id,\nstratified by diagnosis;\nno lesion in more than one fold", C["data"], bold_first=True, fs=8.3)
    box(ax, 1, 2, 26, 8, "SC-EfficientNet pilot (fold 0, B0)\npre-specified pass criteria", C["sc"], bold_first=True, fs=8)
    arrow(ax, (14, 36), (14, 32))
    box(ax, 31, 38, 30, 11, "Two scales\nB0: 224 × 224, CE, no sampler\nB4: 384 × 384, CE, weighted sampler", C["model"], bold_first=True, fs=8.3)
    box(ax, 31, 15, 30, 20, "Five model variants\n•  Standard baseline (1×1 head conv)\n•  Headless control\n•  Attention-only (B0 only)\n•  KL-only ablation (mask via KL only)\n•  SC-EfficientNet (supervised mask + priors)", C["model"], bold_first=True, fs=7.7, ha="left")
    box(ax, 31, 2, 30, 10, "Training protocol\nAdam, lr 10⁻⁴, 20 epochs, FP32\nseeds 42, 43, 44; final-epoch model", C["model"], bold_first=True, fs=8.1)
    arrow(ax, (27, 22), (31, 25)); arrow(ax, (27, 6), (31, 7)); arrow(ax, (46, 38), (46, 35)); arrow(ax, (46, 15), (46, 12))
    box(ax, 65, 37, 26, 12, "Pooled OOF predictions\nper configuration and seed\n10,015 held-out images", C["eval"], bold_first=True, fs=8.3)
    box(ax, 65, 16, 26, 18, "Metrics and paired tests\naccuracy, balanced accuracy,\nmacro-F1 (mean ± SD over seeds);\nlesion-level bootstrap 95% CI;\nexact McNemar test", C["eval"], bold_first=True, fs=7.7)
    box(ax, 65, 2, 26, 11, "Mask diagnostics\nraw mask mean and fraction\nof positions < 0.5", C["eval"], bold_first=True, fs=8.1)
    arrow(ax, (61, 25), (65, 41)); arrow(ax, (78, 37), (78, 34)); arrow(ax, (78, 16), (78, 13))
    items = [("Official ISIC 2018 test set", "1,512 images; 3 seeds"), ("Synthetic artifacts", "hair, ruler, occlusion; 3 seeds"),
             ("TTA and XAI", "5 views; attention vs. Grad-CAM"), ("ROC / PR, class-wise", "SC-EfficientNet-B4, seed 42"), ("Complexity and latency", "params, FLOPs, GPU time")]
    ys = [41, 31.5, 22, 12.5, 3]
    ax.plot([93, 93], [ys[-1] + 3.5, ys[0] + 3.5], color=C["edge"], lw=1.2)
    ax.plot([91, 93], [43, 43], color=C["edge"], lw=1.2)
    for (t, sub), y in zip(items, ys):
        box(ax, 95, y, 25, 7.5, f"{t}\n{sub}", C["eval"], bold_first=True, fs=7.9)
        arrow(ax, (93, y + 3.5), (95, y + 3.5))
    fig.savefig(FIG / "Figure1_study_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG / "Figure1_study_pipeline.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def figure3():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 120); ax.set_ylim(0, 66); ax.axis("off")
    ax.text(1, 65, "(a) SC-EfficientNet (Supervised-Consistency EfficientNet)", fontsize=11, fontweight="bold", va="top")
    box(ax, 1, 40, 12, 10, "Input image\n224² (B0)\n384² (B4)", C["data"], fs=8)
    box(ax, 16, 40, 17, 10, "EfficientNet-B0/B4\nfinal stage\n(no 1×1 head conv)", C["model"], fs=8)
    box(ax, 36, 40, 10, 10, "$F$\n$C \\times H \\times W$", C["head"], fs=9)
    arrow(ax, (13, 45), (16, 45)); arrow(ax, (33, 45), (36, 45))
    ax.text(55, 60.5, "original path", fontsize=8, color=C["grey"], style="italic")
    box(ax, 55, 52, 8, 7, "GAP", C["head"], fs=8.5)
    box(ax, 67, 52, 15, 7, "Linear head\n$(W_{org}, b_{org})$", C["head"], fs=8)
    box(ax, 86, 52, 11, 7, "$P_{org}$", "#FFFFFF", fs=10, ec=C["grey"])
    arrow(ax, (46, 48), (55, 55.5)); arrow(ax, (63, 55.5), (67, 55.5)); arrow(ax, (82, 55.5), (86, 55.5))
    box(ax, 55, 40, 12, 9, "$F' = F \\odot M$", C["sc"], fs=9)
    box(ax, 70, 40.5, 8, 8, "GAP", C["head"], fs=8.5)
    box(ax, 81, 40.5, 16, 8, "Linear head\n$(W_{mask}, b_{mask})$", C["head"], fs=8)
    box(ax, 101, 40.5, 12, 8, "$P_{masked}$", "#FFFFFF", fs=10, ec=C["blue"], lw=1.6)
    arrow(ax, (46, 44.5), (55, 44.5)); arrow(ax, (67, 44.5), (70, 44.5)); arrow(ax, (78, 44.5), (81, 44.5)); arrow(ax, (97, 44.5), (101, 44.5))
    ax.text(107, 38.8, "inference output", fontsize=8, color=C["blue"], ha="center", va="top")
    ax.text(55, 37.6, "masked path", fontsize=8, color=C["grey"], style="italic", va="top")
    box(ax, 50, 23, 26, 10, "Spatial attention (98 parameters)\nchannel-wise mean and max\n→ 7×7 conv → sigmoid", C["sc"], fs=7.8)
    box(ax, 80, 23, 12, 10, "$M$\n$1 \\times H \\times W$", C["sc"], fs=9)
    arrow(ax, (41, 40), (54, 33)); arrow(ax, (76, 28), (80, 28)); arrow(ax, (86, 33), (64, 40))
    ax.annotate("", xy=(93, 52), xytext=(105, 48.5), arrowprops=dict(arrowstyle="<|-|>", color=C["red"], lw=1.1, ls="--", mutation_scale=10))
    ax.text(101, 51.2, "KL consistency", fontsize=8, color=C["red"])
    ax.text(1, 20, "(b) Training objectives", fontsize=11, fontweight="bold", va="top")
    box(ax, 1, 2.6, 50, 14.6, "", "#FFFFFF", ec=C["red"], lw=1.4)
    ax.text(3, 15.6, "KL-only ablation", fontsize=9.5, fontweight="bold", color=C["red"], va="top")
    ax.text(3, 11.6, r"$\mathcal{L}_{KL\text{-}only} = \mathcal{L}_{cls}(P_{org}, y) + \lambda\,\mathrm{KL}(\mathrm{sg}(P_{org}) \,\|\, P_{masked})$", fontsize=9.5, va="top")
    ax.text(3, 7.0, "M is trained only through the KL term and collapses\nto a nearly constant map (mean ≈ 0.98–0.99).", fontsize=7.8, color=C["red"], va="top", linespacing=1.3)
    box(ax, 54, 2.6, 65, 14.6, "", "#FFFFFF", ec=C["blue"], lw=1.4)
    ax.text(56, 15.6, "SC-EfficientNet (proposed)", fontsize=9.5, fontweight="bold", color=C["blue"], va="top")
    ax.text(56, 11.6, r"$\mathcal{L}_{SC} = \mathcal{L}_{cls}(P_{org}, y) + \mathcal{L}_{cls}(P_{masked}, y) + \lambda\,\mathrm{KL}(\mathrm{sg}(P_{org}) \,\|\, P_{masked})$", fontsize=9.5, va="top")
    ax.text(62, 8.0, r"$+\ \beta_{area}\,(\bar{m} - \rho)^2 + \beta_{bin}\,\mathrm{mean}(M \odot (1 - M))$", fontsize=9.5, va="top")
    ax.text(56, 5.2, r"$\lambda = 0.10,\ \rho = 0.5,\ \beta_{area} = 1,\ \beta_{bin} = 0.1$;  selective mask (≈ 50% of positions < 0.5)", fontsize=7.6, color=C["blue"], va="top")
    ax.text(1, 1.0, "Controls: standard baseline = full EfficientNet (with 1×1 head conv) → GAP → linear;  headless = final-stage features → GAP → linear.  sg = stop-gradient.",
            fontsize=7.6, color=C["grey"], va="top")
    fig.savefig(FIG / "Figure3_architecture_objectives.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(FIG / "Figure3_architecture_objectives.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    figure1(); figure3(); print("done")
