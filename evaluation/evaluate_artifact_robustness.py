"""
Synthetic artifact robustness evaluation for SC-EfficientNet-B4 (384x384).

Protocol:
- Same stratified_group_5fold.csv split used in the revision experiments.
- No retraining.
- Same fold-specific checkpoints as the B4/384 experiment.
- Clean validation images plus three deterministic synthetic artifact conditions:
  1) hair-like thin strokes
  2) ruler/marking-like linear artifacts
  3) circular occlusion/specular-like artifact
- Reports Accuracy, Macro Precision, Macro Recall, Macro-F1 and class-wise metrics.
- Artifact generation is deterministic per image_id, so results are reproducible.

Important:
This is a synthetic artifact robustness experiment. It does not claim to
measure robustness to naturally occurring artifacts in the ISIC/HAM10000 data.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from models.sc_model_unified import SCEfficientNet  # noqa: F401,E402
from models.factory import build_from_checkpoint


CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def stable_seed(image_id, artifact):
    key = f"{image_id}|{artifact}|42".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def make_transform(img_size):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def apply_hair_artifact(img, rng):
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    n = int(rng.integers(3, 7))
    for _ in range(n):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(int(0.35 * w), int(0.85 * w)))
        angle = float(rng.uniform(-35, 35))

        # Slightly curved polyline to resemble thin hair.
        points = []
        for t in np.linspace(0, 1, 18):
            x = x0 + length * t * np.cos(np.deg2rad(angle))
            y = y0 + length * t * np.sin(np.deg2rad(angle))
            y += 8.0 * np.sin(t * np.pi * 2 + rng.uniform(0, 2*np.pi))
            points.append((int(x), int(y)))

        width = int(rng.integers(1, 3))
        alpha = int(rng.integers(110, 190))
        draw.line(points, fill=(25, 25, 25, alpha), width=width)

    return img


def apply_ruler_artifact(img, rng):
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # A ruler/marking-like artifact: one strong edge line plus short ticks.
    horizontal = bool(rng.integers(0, 2))
    thickness = int(rng.integers(5, 10))
    alpha = int(rng.integers(110, 175))

    if horizontal:
        y = int(rng.integers(int(0.08*h), int(0.92*h)))
        draw.line([(0, y), (w, y)], fill=(245, 245, 245, alpha), width=thickness)
        for x in range(0, w, max(20, w // 12)):
            tick = int(rng.integers(int(0.015*h), int(0.05*h)))
            draw.line(
                [(x, y), (x, y + tick)],
                fill=(40, 40, 40, min(210, alpha + 25)),
                width=max(1, thickness // 2),
            )
    else:
        x = int(rng.integers(int(0.08*w), int(0.92*w)))
        draw.line([(x, 0), (x, h)], fill=(245, 245, 245, alpha), width=thickness)
        for y in range(0, h, max(20, h // 12)):
            tick = int(rng.integers(int(0.015*w), int(0.05*w)))
            draw.line(
                [(x, y), (x + tick, y)],
                fill=(40, 40, 40, min(210, alpha + 25)),
                width=max(1, thickness // 2),
            )

    return img


def apply_occlusion_artifact(img, rng):
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Moderate circular/elliptical obstruction, placed away from the extreme
    # borders so that it can interfere with lesion/background information.
    cx = int(rng.integers(int(0.2*w), int(0.8*w)))
    cy = int(rng.integers(int(0.2*h), int(0.8*h)))
    rx = int(rng.integers(int(0.07*w), int(0.13*w)))
    ry = int(rng.integers(int(0.07*h), int(0.13*h)))

    fill = (250, 250, 250, int(rng.integers(115, 180)))
    outline = (255, 255, 255, int(rng.integers(150, 220)))
    draw.ellipse(
        [cx-rx, cy-ry, cx+rx, cy+ry],
        fill=fill,
        outline=outline,
        width=max(2, int(min(w, h) * 0.008)),
    )
    return img


def apply_artifact(img, image_id, artifact):
    if artifact == "clean":
        return img
    rng = np.random.default_rng(stable_seed(image_id, artifact))
    if artifact == "hair":
        return apply_hair_artifact(img, rng)
    if artifact == "ruler":
        return apply_ruler_artifact(img, rng)
    if artifact == "occlusion":
        return apply_occlusion_artifact(img, rng)
    raise ValueError(f"Unknown artifact: {artifact}")


class ArtifactDataset(Dataset):
    def __init__(self, df, image_root, transform, artifact):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.artifact = artifact

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        label = CLASSES.index(str(row["dx"]).upper())
        path = self.image_root / str(row["dx"]).upper() / f"{image_id}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        img = apply_artifact(img, image_id, self.artifact)
        img = self.transform(img)

        return img, label, image_id


def load_model(checkpoint_path, device):
    # Baseline or SC, B0 or B4, as recorded in the checkpoint config.
    model, model_type, backbone = build_from_checkpoint(
        checkpoint_path, device, num_classes=len(CLASSES))
    print(f"Loaded {model_type} model with backbone {backbone}")
    return model


@torch.no_grad()
def evaluate_condition(model, loader, device):
    y_true, y_pred, rows = [], [], []

    for images, labels, ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu().numpy()

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(pred.tolist())

        for image_id, yt, yp, p in zip(
            ids, labels.numpy(), pred, probs.cpu().numpy()
        ):
            row = {
                "image_id": image_id,
                "y_true": int(yt),
                "y_pred": int(yp),
            }
            row.update({
                f"prob_{c}": float(p[i])
                for i, c in enumerate(CLASSES)
            })
            rows.append(row)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )
    return metrics, report, cm, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint_dir",
        required=True,
        help="B4/384 checkpoint directory containing fold0.pt ... fold4.pt",
    )
    ap.add_argument(
        "--image_root",
        default="skin_cancer_data",
    )
    ap.add_argument(
        "--splits",
        default="splits/stratified_group_5fold.csv",
    )
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_name", default="sc_b4_384_artifact")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = root / args.checkpoint_dir

    image_root = Path(args.image_root)
    if not image_root.is_absolute():
        image_root = root / args.image_root

    split_path = Path(args.splits)
    if not split_path.is_absolute():
        split_path = root / args.splits

    output_root = root / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / args.output_name
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{args.output_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)

    df = pd.read_csv(split_path)
    required = {"image_id", "dx", "lesion_id", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")

    # Verify all five folds have no lesion overlap between train and validation.
    for fold in range(5):
        train_lesions = set(df.loc[df.fold != fold, "lesion_id"])
        val_lesions = set(df.loc[df.fold == fold, "lesion_id"])
        overlap = train_lesions & val_lesions
        if overlap:
            raise RuntimeError(
                f"Lesion leakage detected in fold {fold}: {len(overlap)} overlaps"
            )

    artifacts = ["clean", "hair", "ruler", "occlusion"]
    transform = make_transform(args.img_size)

    print("=" * 72)
    print("Synthetic Artifact Robustness (model type read from checkpoint)")
    print("=" * 72)
    print("Device:", device)
    print("Image size:", args.img_size)
    print("Checkpoint:", checkpoint_dir)
    print("Data:", image_root)
    print("Artifacts:", ", ".join(artifacts))
    print("Output:", output_dir)

    all_fold_metrics = []
    all_reports = []
    all_rows = []

    for fold in range(5):
        print("\n" + "=" * 72)
        print(f"FOLD {fold}")
        print("=" * 72)

        train_df = df[df.fold != fold].copy()
        val_df = df[df.fold == fold].copy().reset_index(drop=True)

        checkpoint = checkpoint_dir / f"fold{fold}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        print(f"Validation images: {len(val_df)}")
        print(f"Validation lesions: {val_df['lesion_id'].nunique()}")
        print(f"Checkpoint: {checkpoint}")

        model = load_model(checkpoint, device)

        fold_result = {"fold": fold}

        for artifact in artifacts:
            print(f"  Evaluating condition: {artifact}")

            ds = ArtifactDataset(
                val_df,
                image_root=image_root,
                transform=transform,
                artifact=artifact,
            )
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

            metrics, report, cm, rows = evaluate_condition(
                model, loader, device
            )

            fold_result[f"{artifact}_accuracy"] = metrics["accuracy"]
            fold_result[f"{artifact}_macro_precision"] = metrics["macro_precision"]
            fold_result[f"{artifact}_macro_recall"] = metrics["macro_recall"]
            fold_result[f"{artifact}_macro_f1"] = metrics["macro_f1"]

            report_df = pd.DataFrame(report).T
            report_df.to_csv(
                output_dir / f"classification_report_fold{fold}_{artifact}.csv"
            )

            pd.DataFrame(
                cm, index=CLASSES, columns=CLASSES
            ).to_csv(
                output_dir / f"confusion_matrix_fold{fold}_{artifact}.csv"
            )

            for row in rows:
                row["fold"] = fold
                row["artifact"] = artifact
                all_rows.append(row)

            print(
                f"    Accuracy={metrics['accuracy']:.4f} | "
                f"Macro-F1={metrics['macro_f1']:.4f}"
            )

        all_fold_metrics.append(fold_result)

    # Global OOF metrics for each condition.
    oof_df = pd.DataFrame(all_rows)
    global_rows = []

    for artifact in artifacts:
        sub = oof_df[oof_df.artifact == artifact].copy()
        y_true = sub.y_true.to_numpy()
        y_pred = sub.y_pred.to_numpy()

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "artifact": artifact,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        global_rows.append(metrics)

        pd.DataFrame(report).T.to_csv(
            output_dir / f"classification_report_oof_{artifact}.csv"
        )

        cm = confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASSES)))
        )
        pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
            output_dir / f"confusion_matrix_oof_{artifact}.csv"
        )

    global_metrics = pd.DataFrame(global_rows)

    clean = global_metrics[
        global_metrics.artifact == "clean"
    ].iloc[0]

    degradation_rows = []
    for _, row in global_metrics.iterrows():
        if row.artifact == "clean":
            continue
        degradation_rows.append({
            "artifact": row.artifact,
            "accuracy_clean": clean.accuracy,
            "accuracy_artifact": row.accuracy,
            "accuracy_delta": row.accuracy - clean.accuracy,
            "macro_f1_clean": clean.macro_f1,
            "macro_f1_artifact": row.macro_f1,
            "macro_f1_delta": row.macro_f1 - clean.macro_f1,
            "macro_recall_clean": clean.macro_recall,
            "macro_recall_artifact": row.macro_recall,
            "macro_recall_delta": row.macro_recall - clean.macro_recall,
        })

    global_metrics.to_csv(
        output_dir / "global_oof_metrics.csv", index=False
    )
    pd.DataFrame(all_fold_metrics).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    pd.DataFrame(degradation_rows).to_csv(
        output_dir / "artifact_degradation.csv", index=False
    )
    oof_df.to_csv(
        output_dir / "oof_predictions_artifact.csv", index=False
    )

    config = {
        "protocol": "synthetic_artifact_robustness",
        "model": "SC-EfficientNet-B4",
        "backbone": "efficientnet_b4",
        "img_size": args.img_size,
        "checkpoint_dir": str(checkpoint_dir),
        "image_root": str(image_root),
        "split_file": str(split_path),
        "folds": 5,
        "seed": 42,
        "artifacts": artifacts,
        "artifact_definitions": {
            "hair": "3-6 deterministic thin dark hair-like strokes",
            "ruler": "one deterministic linear marking plus tick marks",
            "occlusion": "one deterministic semi-transparent circular/elliptical obstruction",
        },
        "note": (
            "Synthetic perturbations; this experiment does not quantify "
            "robustness to naturally occurring artifacts."
        ),
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("GLOBAL OOF ARTIFACT ROBUSTNESS")
    print("=" * 72)
    print(global_metrics.to_string(index=False))

    print("\nDegradation relative to clean:")
    print(pd.DataFrame(degradation_rows).to_string(index=False))

    print("\n" + "=" * 72)
    print("ARTIFACT ROBUSTNESS COMPLETED")
    print("=" * 72)
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
