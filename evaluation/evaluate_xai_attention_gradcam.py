"""
XAI comparison: native Spatial Attention vs Grad-CAM
SC-EfficientNet-B4, 384x384, same 5-fold group split.

For each selected validation image:
- Original image
- Native spatial attention map
- Grad-CAM from the last EfficientNet feature map
- Attention-overlay
- Grad-CAM-overlay

The explanation target is the predicted class of the masked inference path.
No retraining is performed.

Outputs:
results/<output_name>/
  xai_summary.csv
  fold*/<class>_<image_id>_xai.png
  config.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from models.sc_model_unified import SCEfficientNet

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


def find_image(root, image_id, dx):
    candidates = [
        root / str(dx).upper() / f"{image_id}.jpg",
        root / str(dx).upper() / f"{image_id}.jpeg",
        root / str(dx).upper() / f"{image_id}.png",
        root / f"{image_id}.jpg",
        root / f"{image_id}.jpeg",
        root / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(root.rglob(f"{image_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Image not found: {image_id}")


def transform(img, size):
    return Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])(img)


def denormalize(t):
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    x = (t * std + mean).clamp(0, 1)
    return x.detach().cpu().permute(1, 2, 0).numpy()


def normalize_map(x):
    x = x - x.min()
    mx = x.max()
    if mx > 1e-8:
        x = x / mx
    return x


def heatmap_rgb(hm):
    # Matplotlib-free simple perceptual-ish red/yellow/blue map.
    # Values are normalized and converted to RGB.
    h = np.clip(hm, 0, 1)
    r = np.clip(2.0 * h, 0, 1)
    g = np.clip(2.0 * (1.0 - np.abs(h - 0.5) * 2.0), 0, 1)
    b = np.clip(2.0 * (1.0 - h), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(base, hm, alpha=0.45):
    color = heatmap_rgb(hm)
    return np.clip((1 - alpha) * base + alpha * color, 0, 1)


def save_panel(images, titles, path):
    from PIL import ImageDraw, ImageFont

    h, w, _ = images[0].shape
    margin = 70
    canvas = Image.new("RGB", (w * len(images), h + margin), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (img, title) in enumerate(zip(images, titles)):
        im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        canvas.paste(im, (i * w, margin))
        draw.text((i * w + 8, 10), title, fill="black")

    canvas.save(path, quality=95)


class FeatureMapCapture:
    """
    Captures the final 4-D feature tensor produced inside the timm
    EfficientNet backbone, without assuming a specific module indexing API.
    """

    def __init__(self, backbone):
        self.candidates = []
        self.handles = []

        for name, module in backbone.named_modules():
            # The root module output is handled by child hooks. We register
            # hooks broadly and retain only tensor outputs with BCHW shape.
            self.handles.append(
                module.register_forward_hook(self._hook)
            )

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output) and output.ndim == 4:
            output.retain_grad()
            self.candidates.append(output)

    def clear(self):
        self.candidates.clear()

    def get_final(self):
        if not self.candidates:
            raise RuntimeError(
                "Could not capture a 4-D feature map from the EfficientNet backbone."
            )
        return self.candidates[-1]

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class GradCAM:
    """
    Grad-CAM computed from a captured final 4-D backbone feature map.
    """

    def __init__(self, backbone, output_size):
        self.output_size = output_size
        self.capture = FeatureMapCapture(backbone)

    def clear(self):
        self.capture.clear()

    def generate(self):
        activations = self.capture.get_final()
        gradients = activations.grad

        if gradients is None:
            raise RuntimeError(
                "Gradients for the captured feature map are unavailable."
            )

        # B,C,H,W
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()
        return normalize_map(cam)

    def remove(self):
        self.capture.remove()

def get_attention_and_logits(model, x):
    feat = model.backbone(x)[-1]
    mask = model.attention(feat)

    pooled_org = model.pool(feat).flatten(1)
    pooled_mask = model.pool(feat * mask).flatten(1)

    logits_org = model.classifier_org(pooled_org)
    logits_mask = model.classifier_masked(pooled_mask)
    return feat, mask, logits_org, logits_mask


def load_model(checkpoint, device):
    model = SCEfficientNet(
        num_classes=7,
        pretrained=False,
        backbone="b4",
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def stable_rank(image_id):
    return int(hashlib.sha256(str(image_id).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--image_root", default="skin_cancer_data")
    ap.add_argument("--splits", default="splits/stratified_group_5fold.csv")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--images_per_class", type=int, default=2)
    ap.add_argument("--output_name", default="sc_b4_384_xai")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    checkpoint_dir = resolve(args.checkpoint_dir)
    image_root = resolve(args.image_root)
    split_path = resolve(args.splits)

    output_dir = root / "results" / args.output_name
    suffix = 1
    while output_dir.exists():
        output_dir = root / "results" / f"{args.output_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)

    df = pd.read_csv(split_path)

    # Verify group isolation exactly as in the other revision experiments.
    for fold in range(5):
        train_lesions = set(df.loc[df.fold != fold, "lesion_id"])
        val_lesions = set(df.loc[df.fold == fold, "lesion_id"])
        overlap = train_lesions & val_lesions
        if overlap:
            raise RuntimeError(f"Lesion leakage in fold {fold}: {len(overlap)}")

    print("=" * 72)
    print("SC-EfficientNet-B4 XAI: Spatial Attention vs Grad-CAM")
    print("=" * 72)
    print("Device:", device)
    print("Image size:", args.img_size)
    print("Images per class:", args.images_per_class)
    print("Checkpoint:", checkpoint_dir)
    print("Output:", output_dir)

    transform_fn = lambda img: transform(img, args.img_size)
    summary = []

    for fold in range(5):
        val_df = df[df.fold == fold].copy()

        # Deterministic class-balanced sample from this validation fold.
        selected = []
        for cls in CLASSES:
            sub = val_df[val_df.dx.str.upper() == cls].copy()
            sub["_rank"] = sub.image_id.map(stable_rank)
            sub = sub.sort_values("_rank").head(args.images_per_class)
            selected.append(sub)
        selected = pd.concat(selected, ignore_index=True)

        fold_out = output_dir / f"fold{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        checkpoint = checkpoint_dir / f"fold{fold}.pt"
        model = load_model(checkpoint, device)

        # Capture the final 4-D feature map produced by the timm
        # EfficientNet backbone. This avoids relying on backbone[-1],
        # which is not valid for the features_only model wrapper.
        gradcam = GradCAM(model.backbone, args.img_size)

        print(f"\nFOLD {fold}: {len(selected)} images")

        for _, row in selected.iterrows():
            image_id = str(row.image_id)
            true_cls = str(row.dx).upper()
            image_path = find_image(image_root, image_id, true_cls)

            original = Image.open(image_path).convert("RGB")
            base = np.asarray(
                original.resize((args.img_size, args.img_size))
            ).astype(np.float32) / 255.0

            x = transform_fn(original).unsqueeze(0).to(device)

            model.zero_grad(set_to_none=True)
            gradcam.clear()
            feat, mask, logits_org, logits_mask = get_attention_and_logits(model, x)

            probs = torch.softmax(logits_mask, dim=1)
            pred_id = int(probs.argmax(dim=1).item())
            pred_cls = CLASSES[pred_id]
            pred_score = float(probs[0, pred_id].item())

            # Native spatial attention map.
            attention = mask[0, 0].detach().cpu().numpy()
            attention = normalize_map(attention)
            attention_img = np.array(
                Image.fromarray((attention * 255).astype(np.uint8)).resize(
                    (args.img_size, args.img_size)
                )
            ) / 255.0

            # Grad-CAM for the predicted masked-path class.
            score = logits_mask[0, pred_id]
            score.backward()

            gradcam_map = gradcam.generate()

            attention_overlay = overlay(base, attention_img)
            gradcam_overlay = overlay(base, gradcam_map)

            out_name = f"{true_cls}_{image_id}_pred-{pred_cls}.png"
            save_panel(
                [
                    base,
                    heatmap_rgb(attention_img),
                    attention_overlay,
                    heatmap_rgb(gradcam_map),
                    gradcam_overlay,
                ],
                [
                    f"Original\nTrue: {true_cls}",
                    "Spatial Attention",
                    "Attention Overlay",
                    "Grad-CAM",
                    f"Grad-CAM Overlay\nPred: {pred_cls} ({pred_score:.2f})",
                ],
                fold_out / out_name,
            )

            # Quantitative similarity between the two normalized maps.
            a = attention_img.flatten()
            g = gradcam_map.flatten()
            corr = float(np.corrcoef(a, g)[0, 1]) if np.std(a) > 1e-8 and np.std(g) > 1e-8 else 0.0

            summary.append({
                "fold": fold,
                "image_id": image_id,
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "predicted_probability": pred_score,
                "attention_mean": float(attention_img.mean()),
                "attention_max": float(attention_img.max()),
                "gradcam_mean": float(gradcam_map.mean()),
                "gradcam_max": float(gradcam_map.max()),
                "attention_gradcam_pearson": corr,
                "image_path": str(image_path),
            })

        gradcam.remove()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / "xai_summary.csv", index=False)

    class_summary = (
        summary_df.groupby("true_class")
        .agg(
            n=("image_id", "count"),
            mean_attention_gradcam_pearson=("attention_gradcam_pearson", "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
        )
        .reset_index()
    )
    class_summary.to_csv(output_dir / "xai_class_summary.csv", index=False)

    config = {
        "protocol": "native_spatial_attention_vs_gradcam",
        "model": "SC-EfficientNet-B4",
        "backbone": "efficientnet_b4",
        "img_size": args.img_size,
        "folds": 5,
        "images_per_class_per_fold": args.images_per_class,
        "target_class": "predicted class from masked inference path",
        "gradcam_layer": "final 4-D feature map captured dynamically from model.backbone",
        "attention_source": "model.attention(feat)[0,0]",
        "split_file": str(split_path),
        "checkpoint_dir": str(checkpoint_dir),
        "note": (
            "Pearson correlation is descriptive only; attention maps and "
            "Grad-CAM are different explanation mechanisms and are not "
            "treated as interchangeable."
        ),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("XAI COMPLETED")
    print("=" * 72)
    print("Images:", len(summary_df))
    print("Results:", output_dir)
    print("\nClass summary:")
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
