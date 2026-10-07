import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sc_model_unified import SCEfficientNet  # noqa: F401
from models.factory import build_from_checkpoint
from data_utils.dataset import CLASS_NAMES


NUM_CLASSES = 7

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# DEVICE
# ============================================================

def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# DATASET
# ============================================================

class TTADataset(Dataset):

    def __init__(self, df, data_dir, img_size, transform):

        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, image_id):

        candidates = [
            self.data_dir / f"{image_id}.jpg",
            self.data_dir / f"{image_id}.jpeg",
            self.data_dir / f"{image_id}.png",
        ]

        for path in candidates:
            if path.exists():
                return path

        # Recursive fallback
        matches = list(
            self.data_dir.rglob(f"{image_id}.*")
        )

        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Image not found: {image_id}"
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_id = str(row["image_id"])
        label = int(row["label"])

        image_path = self._find_image(image_id)

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        return (
            image,
            label,
            image_id,
        )


# ============================================================
# TTA TRANSFORMS
# ============================================================

def build_tta_transforms(img_size):

    normalize = T.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    transforms_dict = {

        "original": T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize,
        ]),

        "hflip": T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            normalize,
        ]),

        "vflip": T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            normalize,
        ]),

        "hvflip": T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            normalize,
        ]),

        "rot10": T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomRotation(
                degrees=(10, 10)
            ),
            T.ToTensor(),
            normalize,
        ]),
    }

    return transforms_dict


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# SINGLE TTA VIEW
# ============================================================

@torch.no_grad()
def predict_view(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
):

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    probabilities = []
    labels = []
    image_ids = []

    for images, y, ids in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        probs = torch.softmax(
            logits,
            dim=1,
        )

        probabilities.append(
            probs.cpu().numpy()
        )

        labels.extend(
            y.numpy().tolist()
        )

        image_ids.extend(
            list(ids)
        )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    return (
        probabilities,
        labels,
        image_ids,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    y_pred = probabilities.argmax(
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": report,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SC-EfficientNet-B4 TTA evaluation "
            "using fixed lesion-level 5-fold split."
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing fold0.pt ... fold4.pt",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="sc_b4_384_tta",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else ROOT / "skin_cancer_data"
    )

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else ROOT / "HAM10000_metadata.csv"
    )

    split_path = (
        Path(args.split_file)
        if args.split_file
        else ROOT
        / "splits"
        / "stratified_group_5fold.csv"
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = select_device()

    print("=" * 70)
    print("TTA Evaluation (model type read from checkpoint)")
    print("=" * 70)
    print("Device:", device)
    print("Image size:", args.img_size)
    print("Checkpoint:", checkpoint_dir)
    print("Data:", data_dir)
    print()

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        metadata_path
    )

    required = {
        "image_id",
        "lesion_id",
        "dx",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    df["image_id"] = (
        df["image_id"]
        .astype(str)
    )

    df["lesion_id"] = (
        df["lesion_id"]
        .astype(str)
    )

    df["dx"] = (
        df["dx"]
        .astype(str)
        .str.upper()
    )

    class_to_idx = {
        name: i
        for i, name in enumerate(CLASS_NAMES)
    }

    df["label"] = df["dx"].map(
        class_to_idx
    )

    if df["label"].isna().any():
        raise ValueError(
            "Unknown class detected."
        )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Load fixed split
    # --------------------------------------------------------

    split_df = pd.read_csv(
        split_path
    )

    split_df["image_id"] = (
        split_df["image_id"]
        .astype(str)
    )

    if "fold" not in split_df.columns:
        raise ValueError(
            "Split file must contain 'fold'."
        )

    df = df.merge(
        split_df[
            ["image_id", "fold"]
        ],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if df["fold"].isna().any():
        raise ValueError(
            "Some images are missing from "
            "the fixed split file."
        )

    df["fold"] = (
        df["fold"]
        .astype(int)
    )

    # --------------------------------------------------------
    # TTA transforms
    # --------------------------------------------------------

    tta_transforms = (
        build_tta_transforms(
            args.img_size
        )
    )

    print(
        "TTA views:",
        ", ".join(
            tta_transforms.keys()
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        ROOT
        / "results"
        / f"{args.output_name}_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "model": "SC-EfficientNet-B4",
        "backbone": "b4",
        "input_size": "384x384",
        "tta_views": list(
            tta_transforms.keys()
        ),
        "num_tta_views": len(
            tta_transforms
        ),
        "aggregation": (
            "mean_probability"
        ),
        "split_protocol": (
            "fixed stratified group "
            "5-fold by lesion_id"
        ),
        "metadata": str(
            metadata_path
        ),
        "split_file": str(
            split_path
        ),
        "checkpoint_dir": str(
            checkpoint_dir
        ),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "timestamp": timestamp,
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    all_y = []
    all_probs = []
    all_rows = []

    fold_metrics = []

    # --------------------------------------------------------
    # Five folds
    # --------------------------------------------------------

    for fold in range(5):

        print()
        print("=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        val_df = (
            df[df["fold"] == fold]
            .copy()
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Leakage check
        # ----------------------------------------------------

        train_df = df[
            df["fold"] != fold
        ]

        overlap = set(
            train_df["lesion_id"]
        ).intersection(
            set(
                val_df["lesion_id"]
            )
        )

        if overlap:

            raise RuntimeError(
                f"Lesion leakage detected "
                f"in fold {fold}: "
                f"{len(overlap)} lesions."
            )

        print(
            "Validation images:",
            len(val_df),
        )

        print(
            "Validation lesions:",
            val_df[
                "lesion_id"
            ].nunique(),
        )

        # ----------------------------------------------------
        # Find checkpoint
        # ----------------------------------------------------

        candidates = [
            checkpoint_dir
            / f"fold{fold}.pt",

            checkpoint_dir
            / f"fold{fold}_final.pt",
        ]

        checkpoint_path = None

        for candidate in candidates:

            if candidate.exists():

                checkpoint_path = candidate
                break

        if checkpoint_path is None:

            raise FileNotFoundError(
                f"No checkpoint found for fold {fold} "
                f"in {checkpoint_dir}"
            )

        print(
            "Checkpoint:",
            checkpoint_path,
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model, model_type, backbone = build_from_checkpoint(
            checkpoint_path,
            device,
            num_classes=NUM_CLASSES,
        )
        print("Model:", model_type, backbone)

        # ----------------------------------------------------
        # Each TTA view
        # ----------------------------------------------------

        view_probs = []

        reference_labels = None
        reference_ids = None

        for view_name, transform in (
            tta_transforms.items()
        ):

            print(
                f"  Evaluating TTA view: "
                f"{view_name}"
            )

            dataset = TTADataset(
                val_df,
                data_dir,
                args.img_size,
                transform,
            )

            probs, labels, ids = (
                predict_view(
                    model,
                    dataset,
                    device,
                    args.batch_size,
                    args.num_workers,
                )
            )

            if reference_labels is None:

                reference_labels = labels
                reference_ids = ids

            else:

                if not np.array_equal(
                    reference_labels,
                    labels,
                ):

                    raise RuntimeError(
                        "Label order changed "
                        "between TTA views."
                    )

                if reference_ids != ids:

                    raise RuntimeError(
                        "Image order changed "
                        "between TTA views."
                    )

            view_probs.append(
                probs
            )

            # Save individual view probabilities
            view_output = pd.DataFrame(
                probs,
                columns=[
                    f"prob_{c}"
                    for c in CLASS_NAMES
                ],
            )

            view_output.insert(
                0,
                "image_id",
                ids,
            )

            view_output.insert(
                1,
                "label",
                labels,
            )

            view_output.to_csv(
                output_dir
                / f"fold{fold}_{view_name}.csv",
                index=False,
            )

        # ----------------------------------------------------
        # Probability averaging
        # ----------------------------------------------------

        stacked = np.stack(
            view_probs,
            axis=0,
        )

        tta_probs = stacked.mean(
            axis=0
        )

        y_true = reference_labels
        image_ids = reference_ids

        metrics = calculate_metrics(
            y_true,
            tta_probs,
        )

        print()
        print(
            f"Fold {fold} TTA Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Fold {fold} TTA Macro-F1: "
            f"{metrics['macro_f1']:.4f}"
        )

        print(
            f"Fold {fold} TTA Macro Precision: "
            f"{metrics['macro_precision']:.4f}"
        )

        print(
            f"Fold {fold} TTA Macro Recall: "
            f"{metrics['macro_recall']:.4f}"
        )

        # ----------------------------------------------------
        # Fold metrics
        # ----------------------------------------------------

        fold_metrics.append({
            "fold": fold,
            "accuracy": metrics[
                "accuracy"
            ],
            "macro_precision": metrics[
                "macro_precision"
            ],
            "macro_recall": metrics[
                "macro_recall"
            ],
            "macro_f1": metrics[
                "macro_f1"
            ],
            "n_validation": len(
                val_df
            ),
            "n_lesions": val_df[
                "lesion_id"
            ].nunique(),
        })

        # ----------------------------------------------------
        # Fold report
        # ----------------------------------------------------

        report_df = pd.DataFrame(
            metrics["report"]
        ).T

        report_df.to_csv(
            output_dir
            / f"classification_report_fold{fold}.csv"
        )

        cm_df = pd.DataFrame(
            metrics[
                "confusion_matrix"
            ],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )

        cm_df.to_csv(
            output_dir
            / f"confusion_matrix_fold{fold}.csv"
        )

        # ----------------------------------------------------
        # Fold OOF predictions
        # ----------------------------------------------------

        fold_output = pd.DataFrame(
            tta_probs,
            columns=[
                f"prob_{c}"
                for c in CLASS_NAMES
            ],
        )

        fold_output.insert(
            0,
            "image_id",
            image_ids,
        )

        fold_output.insert(
            1,
            "fold",
            fold,
        )

        fold_output.insert(
            2,
            "y_true",
            y_true,
        )

        fold_output.insert(
            3,
            "y_pred",
            metrics[
                "predictions"
            ],
        )

        fold_output.to_csv(
            output_dir
            / f"oof_tta_fold{fold}.csv",
            index=False,
        )

        all_y.extend(
            y_true.tolist()
        )

        all_probs.append(
            tta_probs
        )

        for i, image_id in enumerate(
            image_ids
        ):

            all_rows.append({
                "image_id": image_id,
                "fold": fold,
                "y_true": int(
                    y_true[i]
                ),
                "y_pred": int(
                    metrics[
                        "predictions"
                    ][i]
                ),
                **{
                    f"prob_{CLASS_NAMES[c]}":
                    float(
                        tta_probs[i, c]
                    )
                    for c in range(
                        NUM_CLASSES
                    )
                },
            })

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ========================================================
    # GLOBAL OOF
    # ========================================================

    print()
    print("=" * 70)
    print("GLOBAL OOF TTA")
    print("=" * 70)

    all_y = np.asarray(
        all_y,
        dtype=np.int64,
    )

    all_probs = np.concatenate(
        all_probs,
        axis=0,
    )

    global_metrics = calculate_metrics(
        all_y,
        all_probs,
    )

    print(
        f"Accuracy: "
        f"{global_metrics['accuracy']:.6f}"
    )

    print(
        f"Macro Precision: "
        f"{global_metrics['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall: "
        f"{global_metrics['macro_recall']:.6f}"
    )

    print(
        f"Macro F1: "
        f"{global_metrics['macro_f1']:.6f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    global_report = pd.DataFrame(
        global_metrics["report"]
    ).T

    global_report.to_csv(
        output_dir
        / "classification_report_oof_tta.csv"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    global_cm = pd.DataFrame(
        global_metrics[
            "confusion_matrix"
        ],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    global_cm.to_csv(
        output_dir
        / "confusion_matrix_oof_tta.csv"
    )

    # --------------------------------------------------------
    # OOF predictions
    # --------------------------------------------------------

    oof_df = pd.DataFrame(
        all_rows
    )

    oof_df.to_csv(
        output_dir
        / "oof_predictions_tta.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Fold metrics
    # --------------------------------------------------------

    fold_metrics_df = pd.DataFrame(
        fold_metrics
    )

    fold_metrics_df.to_csv(
        output_dir
        / "fold_metrics_tta.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "accuracy": global_metrics[
            "accuracy"
        ],
        "macro_precision": global_metrics[
            "macro_precision"
        ],
        "macro_recall": global_metrics[
            "macro_recall"
        ],
        "macro_f1": global_metrics[
            "macro_f1"
        ],
        "n_images": int(
            len(all_y)
        ),
        "n_tta_views": len(
            tta_transforms
        ),
        "tta_views": list(
            tta_transforms.keys()
        ),
        "aggregation": (
            "mean_probability"
        ),
        "fold_metrics": fold_metrics,
        "output_dir": str(
            output_dir
        ),
    }

    with open(
        output_dir / "summary_tta.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("TTA COMPLETED")
    print("=" * 70)
    print(
        "Results:",
        output_dir,
    )


if __name__ == "__main__":
    main()