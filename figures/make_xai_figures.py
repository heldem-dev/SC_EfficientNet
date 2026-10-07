"""Regenerate Figures 12-13 from the archived XAI outputs."""

from pathlib import Path
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "xai" / "sc_b4_384_xai.zip"
SUMMARY = ROOT / "results" / "xai" / "xai_class_summary.csv"
OUT = ROOT / "figures"
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
OUT.mkdir(exist_ok=True)

# One correctly classified validation image per true class (true class == predicted class).
SELECTED = {
    "MEL": "fold2/MEL_ISIC_0029744_pred-MEL.png",
    "NV": "fold0/NV_ISIC_0024812_pred-NV.png",
    "BCC": "fold0/BCC_ISIC_0029337_pred-BCC.png",
    "AKIEC": "fold0/AKIEC_ISIC_0028854_pred-AKIEC.png",
    "BKL": "fold0/BKL_ISIC_0025810_pred-BKL.png",
    "DF": "fold1/DF_ISIC_0027727_pred-DF.png",
    "VASC": "fold0/VASC_ISIC_0026467_pred-VASC.png",
}

with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(ARCHIVE, "r") as zf:
        zf.extractall(tmp)
    xai_root = Path(tmp) / "sc_b4_384_xai"
    for c, rel in SELECTED.items():
        assert rel.split("/")[1].startswith(c + "_") and rel.endswith(f"pred-{c}.png"), rel

    imgs = [(c, Image.open(xai_root / SELECTED[c]).convert("RGB")) for c in CLASSES]
    w, h = imgs[0][1].size
    label_w = 230
    canvas = Image.new("RGB", (label_w + w, len(imgs) * h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except OSError:
        font = None
    for i, (c, im) in enumerate(imgs):
        canvas.paste(im, (label_w, i * h))
        draw.text((15, i * h + h // 2 - 36), f"True: {c}", fill="black", font=font)
        draw.text((15, i * h + h // 2 + 6), f"Pred: {c}", fill="black", font=font)
    canvas.save(OUT / "Figure12_XAI_representative_7classes.png", dpi=(300, 300))

x = pd.read_csv(SUMMARY)
x["order"] = x["true_class"].map({c: i for i, c in enumerate(CLASSES)})
x = x.sort_values("order")
fig, ax1 = plt.subplots(figsize=(10, 5.5))
idx = np.arange(len(x))
width = 0.36
ax1.bar(idx - width / 2, x["mean_attention_gradcam_pearson"], width, label="Attention–Grad-CAM Pearson r")
ax1.set_ylabel("Pearson correlation")
ax1.set_ylim(-0.02, 0.18)
ax1.set_xticks(idx, x["true_class"])
ax1.grid(axis="y", alpha=0.25)
ax2 = ax1.twinx()
ax2.bar(idx + width / 2, x["mean_predicted_probability"], width, label="Mean predicted probability", alpha=0.65)
ax2.set_ylabel("Mean predicted probability")
ax2.set_ylim(0, 1.05)
ax1.set_title("Native spatial attention versus Grad-CAM: quantitative XAI analysis")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "Figure13_XAI_quantitative_summary.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Created Figure 12 and Figure 13 in", OUT)
