"""Revision-v2 figures (4, 11, 12, 13) from exported results only."""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})


def hist(pattern):
    out = []
    for f in sorted(glob.glob(str(RES / pattern / "training_history.csv"))):
        h = pd.read_csv(f)
        if "mask_mean" in h and h.mask_mean.max() > 0:
            out.append(h.groupby("epoch")[["mask_mean", "mask_frac_below_05"]].mean())
    return out


def fig4():
    groups = [("B0 KL-only ablation", "sc_b0_ce_lambda0.1_seed4*", "#c0392b", "-"),
              ("B0 attention-only", "sc_b0_ce_lambda0_attention_only_seed4*", "#7f8c8d", "-"),
              ("B0 SC-EfficientNet", "scv2_b0_*_seed4*", "#2471a3", "-"),
              ("B4 KL-only ablation", "sc_b4_ce_lambda0.1_sampler_seed4*", "#c0392b", "--"),
              ("B4 SC-EfficientNet", "scv2_b4_*_seed4*", "#2471a3", "--")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, pat, col, ls in groups:
        hs = hist(pat)
        if not hs:
            continue
        for k, key in enumerate(["mask_mean", "mask_frac_below_05"]):
            M = np.stack([h[key].values for h in hs])
            e = hs[0].index.values
            ax[k].plot(e, M.mean(0), color=col, ls=ls, lw=2, label=f"{name} (n={len(hs)})")
            ax[k].fill_between(e, M.min(0), M.max(0), color=col, alpha=0.15)
    ax[0].set_ylabel("Mean raw mask value"); ax[1].set_ylabel("Fraction of mask positions < 0.5")
    for a, t in zip(ax, ["(a) Mean mask value", "(b) Spatial selectivity"]):
        a.set_xlabel("Epoch"); a.set_title(t); a.set_ylim(-0.02, 1.02); a.grid(alpha=0.3)
    ax[1].legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "Figure4_mask_trajectories.png", dpi=300); plt.close(fig)


def fig11():
    df = pd.read_csv(RES / "statistics" / "artifact_multiseed_per_run.csv")
    names = ["B4 baseline", "B4 SC-v1", "B4 SC-v2"]
    cols = {"B4 baseline": "#7f8c8d", "B4 SC-v1": "#c0392b", "B4 SC-v2": "#2471a3"}
    arts = ["hair", "ruler", "occlusion"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(3); w = 0.26
    for k, (m, lab) in enumerate([("drop_accuracy", "Accuracy drop (pp)"), ("drop_macro_f1", "Macro-F1 drop (pp)")]):
        for i, name in enumerate(names):
            g = df[df.config == name].groupby("artifact")[m]
            mu, sd = g.mean().loc[arts], g.std().loc[arts]
            ax[k].bar(x + (i - 1) * w, mu.values, w, yerr=sd.values, capsize=3, label={"B4 baseline": "Baseline", "B4 SC-v1": "KL-only ablation", "B4 SC-v2": "SC-EfficientNet"}[name], color=cols[name])
            for j, a in enumerate(arts):
                pts = df[(df.config == name) & (df.artifact == a)][m].values
                ax[k].scatter(np.full(len(pts), x[j] + (i - 1) * w), pts, s=8, color="black", zorder=3)
        ax[k].set_xticks(x); ax[k].set_xticklabels(["Hair", "Ruler", "Occlusion"]); ax[k].set_ylabel(lab); ax[k].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "Figure11_artifact_robustness.png", dpi=300); plt.close(fig)


def fig12(xai_v1_zip_dir, xai_v2_dir):
    sel = [("MEL", "fold2", "0029744"), ("NV", "fold0", "0024812"), ("BCC", "fold0", "0029337"),
           ("AKIEC", "fold0", "0028854"), ("BKL", "fold0", "0025810"), ("DF", "fold1", "0027727"), ("VASC", "fold0", "0026467")]
    cols = ["Original", "KL-only attention", "SC-EfficientNet attention", "SC-EfficientNet overlay", "SC-EfficientNet Grad-CAM"]
    fig, ax = plt.subplots(len(sel), 5, figsize=(10, 14.5))
    for r, (c, fold, iid) in enumerate(sel):
        p1 = Image.open(Path(xai_v1_zip_dir) / fold / f"{c}_ISIC_{iid}_pred-{c}.png").convert("RGB")
        p2 = Image.open(Path(xai_v2_dir) / fold / f"{c}_ISIC_{iid}_pred-{c}.png").convert("RGB")
        W, H = p1.size; pw = W // 5
        crop = lambda im, k: im.crop((k * pw, H - pw, (k + 1) * pw, H))
        panels = [crop(p2, 0), crop(p1, 1), crop(p2, 1), crop(p2, 2), crop(p2, 4)]
        for k, im in enumerate(panels):
            ax[r, k].imshow(im); ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
            if r == 0: ax[r, k].set_title(cols[k], fontsize=9)
        ax[r, 0].set_ylabel(f"{c}\n(true = pred)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "Figure12_XAI_v1_vs_v2.png", dpi=200); plt.close(fig)


def fig13():
    a = pd.read_csv(RES / "xai" / "xai_class_summary.csv").set_index("true_class")
    b = pd.read_csv(RES / "b4_scv2_xai" / "xai_class_summary.csv").set_index("true_class")
    cls = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    x = np.arange(len(cls)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(x - w / 2, a.loc[cls, "mean_attention_gradcam_pearson"], w, label="KL-only ablation", color="#c0392b")
    ax.bar(x + w / 2, b.loc[cls, "mean_attention_gradcam_pearson"], w, label="SC-EfficientNet", color="#2471a3")
    ax.set_xticks(x); ax.set_xticklabels(cls); ax.set_ylabel("Mean Pearson r (attention vs Grad-CAM)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "Figure13_attention_gradcam_correlation.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    import sys, tempfile, zipfile
    fig4(); fig11(); fig13()
    with tempfile.TemporaryDirectory() as t:
        zipfile.ZipFile(RES / "xai" / "sc_b4_384_xai.zip").extractall(t)
        fig12(Path(t) / "sc_b4_384_xai", RES / "b4_scv2_xai")
    print("done")
