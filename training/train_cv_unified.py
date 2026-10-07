"""Train EfficientNet baselines and SC-EfficientNet with lesion-level 5-fold CV.

The same script supports EfficientNet-B0/B4, cross-entropy or focal loss,
optional weighted sampling, and the SC consistency objective.
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_utils.dataset import CLASS_NAMES, SkinDataset  # noqa: E402
from models.sc_model_unified import EfficientNetBaseline, EfficientNetHeadless, SCEfficientNet  # noqa: E402
from utils.losses import FocalLoss  # noqa: E402
from utils.seed import seed_everything  # noqa: E402
from utils.trainer import predict, train_one_epoch  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(train: bool, img_size: int):
    common = [
        T.Resize((img_size, img_size)),
    ]
    if train:
        common.extend([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    common.extend([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return T.Compose(common)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unique_run_dir(base_dir: Path, requested_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / requested_name
    i = 1
    while candidate.exists():
        candidate = base_dir / f"{requested_name}_{i:02d}"
        i += 1
    candidate.mkdir()
    return candidate


def report_to_df(report):
    rows = []
    for key, value in report.items():
        if isinstance(value, dict):
            rows.append({"label": key, **value})
        else:
            rows.append({"label": key, "value": value})
    return pd.DataFrame(rows)


def make_sampler(train_df):
    counts = train_df["dx"].value_counts()
    weights = train_df["dx"].map(lambda x: 1.0 / counts[x]).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def verify_split(df):
    required = {"image_id", "dx", "lesion_id", "fold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")

    df = df.copy()
    df["image_id"] = df["image_id"].astype(str)
    df["lesion_id"] = df["lesion_id"].astype(str)
    df["dx"] = df["dx"].astype(str).str.upper()
    if not set(df["dx"].unique()).issubset(set(CLASS_NAMES)):
        raise ValueError("Split file contains an unknown diagnostic class.")

    # Exact leakage check used by the reported protocol.
    for fold in sorted(df["fold"].unique()):
        train_lesions = set(df.loc[df.fold != fold, "lesion_id"])
        val_lesions = set(df.loc[df.fold == fold, "lesion_id"])
        overlap = train_lesions & val_lesions
        if overlap:
            raise RuntimeError(
                f"Lesion leakage detected in fold {fold}: {len(overlap)} overlapping lesion_id values."
            )
    return df


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "headless", "sc"], default="baseline")
    ap.add_argument("--backbone", choices=["b0", "b4"], default="b0")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--sampler", action="store_true")
    ap.add_argument("--lambda_c", type=float, default=0.10)
    ap.add_argument("--objective", choices=["sc", "sc_v2"], default="sc",
                    help="sc: original objective; sc_v2: both paths supervised + mask area/binarization prior")
    ap.add_argument("--area_target", type=float, default=0.5)
    ap.add_argument("--w_masked_ce", type=float, default=1.0)
    ap.add_argument("--w_area", type=float, default=1.0)
    ap.add_argument("--w_bin", type=float, default=0.1)
    ap.add_argument("--fold_ids", type=str, default=None,
                    help="Comma-separated fold ids to run (e.g. '0' for a pilot). Overrides --folds.")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, choices=range(1, 6), default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "skin_cancer_data"))
    ap.add_argument(
        "--split_file",
        type=str,
        default=str(ROOT / "splits" / "stratified_group_5fold.csv"),
    )
    ap.add_argument("--pretrained", dest="pretrained", action="store_true")
    ap.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    ap.set_defaults(pretrained=True)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()

    split_path = Path(args.split_file).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    df = verify_split(pd.read_csv(split_path))
    available_folds = sorted(int(f) for f in df["fold"].unique())
    run_folds = available_folds[:args.folds]
    if args.fold_ids:
        run_folds = [int(f) for f in args.fold_ids.split(",")]
        assert set(run_folds) <= set(available_folds), run_folds

    attention_only = args.model == "sc" and args.lambda_c == 0.0
    if args.run_name:
        requested_name = args.run_name
    elif args.model == "sc":
        tag = "scv2" if args.objective == "sc_v2" else "sc"
        requested_name = f"{tag}_{args.backbone}_{args.loss}_lambda{args.lambda_c:g}"
        if args.objective == "sc_v2":
            requested_name += f"_area{args.area_target:g}"
        if attention_only:
            requested_name += "_attention_only"
        if args.sampler:
            requested_name += "_sampler"
    else:
        requested_name = f"{args.model}_{args.backbone}_{args.loss}"
        if args.sampler:
            requested_name += "_sampler"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested_name = f"{requested_name}_seed{args.seed}_{timestamp}"
    results_dir = unique_run_dir(ROOT / "results", requested_name)
    checkpoints_dir = unique_run_dir(ROOT / "checkpoints", requested_name)

    config = {
        "model": args.model,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "loss": args.loss,
        "sampler": args.sampler,
        "sampler_definition": "inverse class frequency over the training fold; replacement=True",
        "lambda_c": args.lambda_c,
        "attention_only_ablation": attention_only,
        "objective": args.objective if args.model == "sc" else None,
        "area_target": args.area_target,
        "w_masked_ce": args.w_masked_ce,
        "w_area": args.w_area,
        "w_bin": args.w_bin,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "epochs": args.epochs,
        "folds_requested": args.folds,
        "folds_run": run_folds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "seed": args.seed,
        "device": str(device),
        "num_classes": len(CLASS_NAMES),
        "num_workers": args.num_workers,
        "pretrained": args.pretrained,
        "class_names": CLASS_NAMES,
        "split_file": "splits/stratified_group_5fold.csv",
        "data_dir": "skin_cancer_data",
        "timestamp": timestamp,
    }
    (results_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Device:", device)
    print("Model:", args.model, args.backbone)
    print("Image size:", args.img_size)
    print("Results:", results_dir)
    print("Checkpoints:", checkpoints_dir)

    all_rows = []
    fold_metric_rows = []
    history_rows = []
    all_y, all_pred, all_prob = [], [], []

    for fold in run_folds:
        print(f"\n========== FOLD {fold} ==========")
        train_df = df[df.fold != fold].copy()
        val_df = df[df.fold == fold].copy()

        train_ds = SkinDataset(train_df, root_dir=data_dir, transform=make_transforms(True, args.img_size))
        val_ds = SkinDataset(val_df, root_dir=data_dir, transform=make_transforms(False, args.img_size))

        seed_everything(args.seed * 100 + int(fold))
        sampler = None
        if args.sampler:
            sampler = make_sampler(train_df)
            sampler.generator = torch.Generator().manual_seed(args.seed * 100 + int(fold))
        pin_memory = device.type == "cuda"
        g = torch.Generator()
        g.manual_seed(args.seed * 100 + int(fold))
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            generator=g,
            persistent_workers=args.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        if args.model == "headless":
            model = EfficientNetHeadless(
                num_classes=len(CLASS_NAMES),
                pretrained=args.pretrained,
                backbone=args.backbone,
            )
        elif args.model == "baseline":
            model = EfficientNetBaseline(
                num_classes=len(CLASS_NAMES),
                pretrained=args.pretrained,
                backbone=args.backbone,
            )
        else:
            model = SCEfficientNet(
                num_classes=len(CLASS_NAMES),
                pretrained=args.pretrained,
                backbone=args.backbone,
            )
        model.to(device)

        if args.loss == "focal":
            criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            lambda_for_run = args.lambda_c if args.model == "sc" else 0.0
            stats = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                criterion,
                lambda_c=lambda_for_run,
                attention_only=attention_only,
                objective=args.objective,
                area_target=args.area_target,
                w_masked_ce=args.w_masked_ce,
                w_area=args.w_area,
                w_bin=args.w_bin,
            )
            msg = (f"Epoch {epoch + 1:02d}/{args.epochs} "
                   f"loss={stats['loss']:.5f} cls={stats['loss_cls']:.5f} cons={stats['loss_cons']:.5f}")
            if args.model == "sc":
                msg += f" mask_mean={stats['mask_mean']:.3f} mask<0.5={stats['mask_frac_below_05']:.3f}"
            print(msg, flush=True)
            history_rows.append({"fold": fold, "epoch": epoch + 1, **stats})

        probs_t, y, ids, mask_stats = predict(model, val_loader, device)
        y = np.asarray(y, dtype=np.int64)
        probs = probs_t.numpy()
        pred = probs.argmax(1)

        report = classification_report(
            y,
            pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        acc = accuracy_score(y, pred)
        print(f"Fold {fold} accuracy: {acc:.4f}")

        report_to_df(report).to_csv(
            results_dir / f"classification_report_fold{fold}.csv", index=False
        )
        cm = confusion_matrix(y, pred, labels=np.arange(len(CLASS_NAMES)))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            results_dir / f"confusion_matrix_fold{fold}.csv"
        )

        fold_metric_rows.append({
            "fold": fold,
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "balanced_accuracy": report["macro avg"]["recall"],
            "mask_mean": float(mask_stats[:, 0].mean()) if mask_stats is not None else None,
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_train_lesions": train_df.lesion_id.nunique(),
            "n_val_lesions": val_df.lesion_id.nunique(),
        })

        lesion_map = val_df.set_index("image_id")["lesion_id"].to_dict()
        for k, (image_id, yt, yp, p) in enumerate(zip(ids, y, pred, probs)):
            row = {
                "image_id": image_id,
                "lesion_id": lesion_map.get(image_id, ""),
                "fold": fold,
                "y_true": int(yt),
                "y_pred": int(yp),
            }
            row.update({f"prob_{c}": float(p[i]) for i, c in enumerate(CLASS_NAMES)})
            if mask_stats is not None:
                row.update({"mask_mean": float(mask_stats[k, 0]),
                            "mask_frac_below_05": float(mask_stats[k, 1]),
                            "mask_min": float(mask_stats[k, 2]),
                            "mask_max": float(mask_stats[k, 3])})
            all_rows.append(row)

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": args.epochs,
                "fold": fold,
                "args": vars(args),
                "config": config,
            },
            checkpoints_dir / f"fold{fold}.pt",
        )

        all_y.extend(y.tolist())
        all_pred.extend(pred.tolist())
        all_prob.append(probs)

        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_y = np.asarray(all_y, dtype=np.int64)
    all_pred = np.asarray(all_pred, dtype=np.int64)
    all_prob = np.concatenate(all_prob, axis=0)

    oof_report = classification_report(
        all_y,
        all_pred,
        labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_to_df(oof_report).to_csv(
        results_dir / "classification_report_oof.csv", index=False
    )
    cm = confusion_matrix(all_y, all_pred, labels=np.arange(len(CLASS_NAMES)))
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        results_dir / "confusion_matrix_oof.csv"
    )
    pd.DataFrame(all_rows).to_csv(results_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metric_rows).to_csv(results_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(results_dir / "training_history.csv", index=False)

    summary = {
        "accuracy": float(oof_report["accuracy"]),
        "macro_precision": float(oof_report["macro avg"]["precision"]),
        "macro_recall": float(oof_report["macro avg"]["recall"]),
        "macro_f1": float(oof_report["macro avg"]["f1-score"]),
        "weighted_precision": float(oof_report["weighted avg"]["precision"]),
        "weighted_recall": float(oof_report["weighted avg"]["recall"]),
        "weighted_f1": float(oof_report["weighted avg"]["f1-score"]),
        "n_samples": int(len(all_y)),
        "fold_metrics": fold_metric_rows,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n========== OOF SUMMARY ==========")
    print(f"Accuracy: {summary['accuracy']:.6f}")
    print(
        f"Macro P/R/F1: {summary['macro_precision']:.4f} / "
        f"{summary['macro_recall']:.4f} / {summary['macro_f1']:.4f}"
    )
    print("Saved:", results_dir)


if __name__ == "__main__":
    main()
