"""Evaluate trained fold checkpoints on the official ISIC 2018 Task 3 test set (1,512 images).

No retraining. For every run (checkpoint directory with fold0.pt ... fold4.pt) the script reports
  * each fold model separately (mean ± SD over the five fold models), and
  * the five-fold ensemble (mean of the five softmax outputs),
using accuracy, balanced accuracy (= official ISIC 2018 ranking metric), and macro-F1.

Usage (PowerShell, repository root):
  python -m evaluation.evaluate_official_test --test_images <folder with ISIC_*.jpg> \
      --gt_csv <ISIC2018_Task3_Test_GroundTruth.csv> --runs "checkpoints\\baseline_b4_*" "checkpoints\\sc_b4_*" ...
Optional: --test_metadata <csv with image and lesion_id columns> checks lesion overlap with the training set.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_utils.dataset import CLASS_NAMES  # noqa: E402
from models.factory import build_from_checkpoint  # noqa: E402

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class TestSet(torch.utils.data.Dataset):
    def __init__(self, ids, folder, size):
        self.ids, self.folder = ids, Path(folder)
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.tf(Image.open(self.folder / f"{self.ids[i]}.jpg").convert("RGB")), i


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_images", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoint directories or glob patterns")
    ap.add_argument("--test_metadata", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_name", default="official_test")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt_csv)
    id_col = "image" if "image" in gt.columns else gt.columns[0]
    assert all(c in gt.columns for c in CLASS_NAMES), f"GT columns must include {CLASS_NAMES}"
    ids = gt[id_col].astype(str).tolist()
    y = gt[CLASS_NAMES].values.argmax(1)
    print(f"Test images: {len(ids)}; class counts:", dict(zip(CLASS_NAMES, np.bincount(y, minlength=7))))
    missing = [i for i in ids if not (Path(args.test_images) / f"{i}.jpg").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} test images missing, e.g. {missing[:3]}")

    out = ROOT / "results" / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    report = {"n_test_images": len(ids)}
    if args.test_metadata:
        tm = pd.read_csv(args.test_metadata)
        train_les = set(pd.read_csv(ROOT / "splits" / "stratified_group_5fold.csv").lesion_id)
        overlap = sorted(set(tm.lesion_id.dropna()) & train_les)
        report["lesions_shared_with_training"] = len(overlap)
        print(f"Test lesions also present in the training set: {len(overlap)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat) if Path(d).is_dir()})
    rows = []
    for rd in run_dirs:
        rd = Path(rd)
        cks = sorted(rd.glob("fold*.pt"))
        if len(cks) != 5:
            print(f"[skip] {rd.name}: {len(cks)} fold checkpoints"); continue
        probs = []
        for ck in cks:
            model, mtype, bb = build_from_checkpoint(ck, device, num_classes=len(CLASS_NAMES))
            size = 384 if bb == "b4" else 224
            dl = torch.utils.data.DataLoader(TestSet(ids, args.test_images, size), batch_size=args.batch_size,
                                             num_workers=args.num_workers, shuffle=False)
            P = np.zeros((len(ids), 7), dtype=np.float32)
            with torch.no_grad():
                for x, idx in dl:
                    P[idx.numpy()] = torch.softmax(model(x.to(device)), 1).cpu().numpy()
            probs.append(P)
            m = metrics(y, P.argmax(1))
            rows.append({"run": rd.name, "model": mtype, "backbone": bb, "fold_model": ck.stem, **m})
            print(f"{rd.name} {ck.stem}: acc {m['accuracy']:.4f} bacc {m['balanced_accuracy']:.4f} f1 {m['macro_f1']:.4f}")
            del model; torch.cuda.empty_cache() if device.type == "cuda" else None
        E = np.mean(probs, 0)
        m = metrics(y, E.argmax(1))
        rows.append({"run": rd.name, "model": mtype, "backbone": bb, "fold_model": "ensemble", **m})
        pd.DataFrame(E, columns=[f"prob_{c}" for c in CLASS_NAMES]).assign(image=ids, y_true=y, y_pred=E.argmax(1)) \
            .to_csv(out / f"{rd.name}_ensemble_predictions.csv", index=False)
        np.save(out / f"{rd.name}_fold_probs.npy", np.stack(probs))
        print(f"{rd.name} ENSEMBLE: acc {m['accuracy']:.4f} bacc {m['balanced_accuracy']:.4f} f1 {m['macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "official_test_metrics.csv", index=False)
    (out / "official_test_summary.json").write_text(json.dumps(report, indent=2))
    print("Saved to", out)


if __name__ == "__main__":
    main()
