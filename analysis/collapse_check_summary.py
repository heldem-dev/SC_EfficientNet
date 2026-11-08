"""Summarize the collapse checks of the KL-only formulation (manuscript Section 4.2).

Additional KL-only EfficientNet-B0 runs on fold 0 with seed 42 (run_collapse_check.ps1):
  lambda = 1.0, lambda = 5.0 (20 and 40 epochs), a head shared by both paths (--shared_head),
  and a zero-initialized attention convolution (--attn_init zero).
They are compared with fold 0 of the seed-42 multi-seed runs: the KL-only ablation (lambda = 0.10),
SC-EfficientNet-B0, and the attention-only configuration.

Mask statistics are computed on the held-out images of fold 0 (oof_predictions.csv, mask columns);
training trajectories are read from training_history.csv when present.

Usage:
  python analysis/collapse_check_summary.py --runs_dir results --out results/collapse_check/collapse_check_summary.csv
"""
import argparse
import csv
import glob
import os

RUNS = [  # (label, folder pattern relative to runs_dir)
    ("KL-only, lambda = 0.10 (reference)", "sc_b0_ce_lambda0.1_seed42_*"),
    ("KL-only, lambda = 1.0", "collapse_check/sc_b0_ce_lambda1_seed42_*"),
    ("KL-only, lambda = 5.0", "collapse_check/sc_b0_ce_lambda5_seed42_*"),
    ("KL-only, lambda = 5.0, 40 epochs", "collapse_check/sc_b0_ce_lambda5_ep40_seed42_*"),
    ("KL-only, shared head", "collapse_check/sc_b0_ce_lambda0.1_sharedhead_seed42_*"),
    ("KL-only, zero-initialized attention", "collapse_check/sc_b0_ce_lambda0.1_attnzero_seed42_*"),
    ("SC-EfficientNet-B0", "scv2_b0_ce_lambda0.1_area0.5_seed42_*"),
    ("Attention-only B0", "sc_b0_ce_lambda0_attention_only_seed42_*"),
]
N_CLASSES = 7


def find_run(runs_dir, pattern):
    hits = sorted(d for d in glob.glob(os.path.join(runs_dir, pattern)) if os.path.isdir(d))
    if not hits:  # also accept the run folders directly under runs_dir
        hits = sorted(d for d in glob.glob(os.path.join(runs_dir, os.path.basename(pattern))) if os.path.isdir(d))
    return hits[-1] if hits else None


def fold0_stats(run):
    rows = [r for r in csv.DictReader(open(os.path.join(run, "oof_predictions.csv"))) if r["fold"] == "0"]
    y = [int(r["y_true"]) for r in rows]
    p = [int(r["y_pred"]) for r in rows]
    recall, f1 = [], []
    for c in range(N_CLASSES):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        n_true = sum(1 for a in y if a == c)
        n_pred = sum(1 for b in p if b == c)
        rec = tp / n_true if n_true else 0.0
        prec = tp / n_pred if n_pred else 0.0
        recall.append(rec)
        f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    out = {
        "n_images": len(rows),
        "mask_mean": sum(float(r["mask_mean"]) for r in rows) / len(rows),
        "frac_below_05": sum(float(r["mask_frac_below_05"]) for r in rows) / len(rows),
        "accuracy": 100 * sum(a == b for a, b in zip(y, p)) / len(rows),
        "balanced_accuracy": 100 * sum(recall) / N_CLASSES,
        "macro_f1": 100 * sum(f1) / N_CLASSES,
    }
    hist_path = os.path.join(run, "training_history.csv")
    if os.path.exists(hist_path):
        hist = [r for r in csv.DictReader(open(hist_path)) if r["fold"] == "0"]
        out["epochs"] = len(hist)
        for e in (1, 10, 20, 40):
            if e <= len(hist):
                out[f"train_mask_ep{e}"] = float(hist[e - 1]["mask_mean"])
        out["train_cons_last"] = float(hist[-1]["loss_cons"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results")
    ap.add_argument("--out", default="results/collapse_check/collapse_check_summary.csv")
    args = ap.parse_args()
    cols = ["setting", "run", "n_images", "epochs", "mask_mean", "frac_below_05", "accuracy", "balanced_accuracy",
            "macro_f1", "train_mask_ep1", "train_mask_ep10", "train_mask_ep20", "train_mask_ep40", "train_cons_last"]
    rows = []
    for label, pattern in RUNS:
        run = find_run(args.runs_dir, pattern)
        if run is None:
            print(f"missing: {label} ({pattern})")
            continue
        s = fold0_stats(run)
        s.update(setting=label, run=os.path.basename(run))
        rows.append(s)
        print(f"{label:40s} mask {s['mask_mean']:.3f}  below 0.5 {100 * s['frac_below_05']:5.1f}%  "
              f"acc {s['accuracy']:.2f}  bacc {s['balanced_accuracy']:.2f}  macro-F1 {s['macro_f1']:.2f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print("->", args.out)


if __name__ == "__main__":
    main()
