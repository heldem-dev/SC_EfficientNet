"""Figures 5-10 (detailed analyses) for SC-EfficientNet-B4, seed 42 (revised objective)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
B4 = ROOT / "results/scv2_b4_ce_lambda0.1_area0.5_sampler_seed42_20261004_134635"
TTA = ROOT / "results/b4_scv2_tta_20261004_161603"
OUT = ROOT / "figures" / "sc_seed42"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
OUT.mkdir(parents=True, exist_ok=True)

# Figure 4: training dynamics.
h = pd.read_csv(B4 / "training_history.csv")
h["weighted_consistency"] = 0.10 * h["loss_cons"]
m = h.groupby("epoch")[["loss", "loss_cls", "weighted_consistency"]].mean()
lo = h.groupby("epoch")["loss"].min()
hi = h.groupby("epoch")["loss"].max()
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(m.index, m.loss, label="Total loss (mean)", linewidth=2)
ax.plot(m.index, m.loss_cls, label="Classification loss (mean)", linewidth=2)
ax.plot(m.index, m.weighted_consistency, label=r"Weighted consistency loss ($\lambda L_{cons}$)", linewidth=2)
ax.fill_between(m.index, lo, hi, alpha=0.12)
ax.set(xlabel="Epoch", ylabel="Loss", title="SC-EfficientNet-B4 training dynamics across five folds")
ax.set_xticks(range(1, int(h.epoch.max()) + 1))
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure4_training_dynamics.png", dpi=300)
plt.close(fig)

# Figure 5: confusion matrices.
cm = pd.read_csv(B4 / "confusion_matrix_oof.csv", index_col=0).values
cm_tta = pd.read_csv(TTA / "confusion_matrix_oof_tta.csv", index_col=0).values
fig = plt.figure(figsize=(11, 5))
for ax, mat, title in [
    (fig.add_subplot(121), cm, "OOF confusion matrix without TTA"),
    (fig.add_subplot(122), cm_tta, "OOF confusion matrix with 5-view TTA"),
]:
    im = ax.imshow(mat)
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(7), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "Figure5_confusion_matrices.png", dpi=300)
plt.close(fig)

# OOF probabilities for ROC/PR.
p = pd.read_csv(TTA / "oof_predictions_tta.csv")
y = p.y_true.to_numpy()
yb = label_binarize(y, classes=np.arange(len(CLASSES)))
P = p[[f"prob_{c}" for c in CLASSES]].to_numpy()

# Figure 6: ROC/AUC.
auc_values = {}
fig, ax = plt.subplots(figsize=(9, 7))
for i, c in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(yb[:, i], P[:, i])
    a = auc(fpr, tpr)
    auc_values[c] = float(a)
    ax.plot(fpr, tpr, linewidth=1.8, label=f"{c} (AUC={a:.3f})")
fpr, tpr, _ = roc_curve(yb.ravel(), P.ravel())
micro_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, "--", linewidth=2, label=f"micro-average (AUC={micro_auc:.3f})")
ax.plot([0, 1], [0, 1], ":", linewidth=1.5, label="Random")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves from five-fold out-of-fold predictions")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure6_ROC_AUC.png", dpi=300)
plt.close(fig)

# Figure 7: PR/AP.
ap_values = {}
fig, ax = plt.subplots(figsize=(9, 7))
for i, c in enumerate(CLASSES):
    precision, recall, _ = precision_recall_curve(yb[:, i], P[:, i])
    ap = average_precision_score(yb[:, i], P[:, i])
    ap_values[c] = float(ap)
    ax.plot(recall, precision, linewidth=1.8, label=f"{c} (AP={ap:.3f})")
micro_ap = average_precision_score(yb.ravel(), P.ravel())
precision, recall, _ = precision_recall_curve(yb.ravel(), P.ravel())
ax.plot(recall, precision, "--", linewidth=2, label=f"micro-average (AP={micro_ap:.3f})")
ax.set(xlabel="Recall", ylabel="Precision", title="One-vs-rest precision-recall curves from five-fold out-of-fold predictions")
ax.legend(loc="lower left", fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure7_PR_AP.png", dpi=300)
plt.close(fig)
(OUT / "roc_pr_metrics.json").write_text(
    json.dumps({"class_auc": auc_values, "micro_auc": micro_auc, "class_ap": ap_values, "micro_ap": micro_ap}, indent=2),
    encoding="utf-8",
)

# Figure 8: TTA comparison, read directly from exported summaries.
before = pd.read_json(B4 / "summary.json", typ="series")
after = pd.read_json(TTA / "summary_tta.json", typ="series")
names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1"]
before_values = [100 * float(before[k]) for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
after_values = [100 * float(after[k]) for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]]
x = np.arange(len(names))
w = 0.36
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x - w / 2, before_values, w, label="Without TTA")
ax.bar(x + w / 2, after_values, w, label="5-view TTA")
ax.set_xticks(x, names, rotation=20, ha="right")
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Effect of five-view test-time augmentation")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure8_TTA_effect.png", dpi=300)
plt.close(fig)

# Figure 9: fold stability.
fm = pd.read_csv(TTA / "fold_metrics_tta.csv")
fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(fm.fold, fm.accuracy * 100, "o-", label="Accuracy")
ax.plot(fm.fold, fm.macro_f1 * 100, "o-", label="Macro-F1")
ax.set_xticks(fm.fold)
ax.set(xlabel="Fold", ylabel="Score (%)", title="Fold-wise performance for five-view TTA")
ax.legend()
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure9_fold_stability.png", dpi=300)
plt.close(fig)

# Figure 10: derive class-wise metrics from the exported B4 OOF report.
report = pd.read_csv(B4 / "classification_report_oof.csv")
report = report[report.label.isin(CLASSES)].copy()
report = report.set_index("label").loc[CLASSES]
cls = pd.DataFrame({
    "Class": CLASSES,
    "Precision": report["precision"].to_numpy(),
    "Recall": report["recall"].to_numpy(),
    "F1": report["f1-score"].to_numpy(),
})
x = np.arange(len(CLASSES))
w = 0.24
fig, ax = plt.subplots(figsize=(10, 5.8))
for j, k in enumerate(["Precision", "Recall", "F1"]):
    ax.bar(x + (j - 1) * w, cls[k] * 100, w, label=k)
ax.set_xticks(x, CLASSES)
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Class-wise performance of SC-EfficientNet-B4 at 384 × 384")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure10_classwise_performance.png", dpi=300)
plt.close(fig)

# Figure 11: exported artifact metrics.
art = pd.read_csv(ROOT / "results" / "artifact" / "global_oof_metrics.csv")
art["artifact"] = art["artifact"].str.title()
x = np.arange(len(art))
w = 0.25
fig, ax = plt.subplots(figsize=(9, 5.5))
for j, k in enumerate(["accuracy", "macro_f1", "macro_recall"]):
    ax.bar(x + (j - 1) * w, art[k] * 100, w, label={"accuracy": "Accuracy", "macro_f1": "Macro-F1", "macro_recall": "Macro Recall"}[k])
ax.set_xticks(x, art["artifact"])
ax.set_ylim(0, 100)
ax.set_ylabel("Score (%)")
ax.set_title("Performance under controlled synthetic perturbations")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure11_artifact_robustness.png", dpi=300)
plt.close(fig)

print("Generated Figures 4-11 in", OUT)
