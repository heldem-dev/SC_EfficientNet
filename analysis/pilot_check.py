"""Decision gate for the SC-v2 pilot (fold 0 only, B0, seed 42).

Usage: python analysis/pilot_check.py results/<pilot_run_dir> [results/<other_pilot_dir> ...]

A pilot passes when, on held-out fold 0:
  (1) mean raw mask value <= 0.70,
  (2) at least 20% of mask positions are below 0.5 (the mask is spatially selective), and
  (3) macro-F1 is at most 2.0 percentage points below the attention-only B0 run on fold 0.
Criteria were fixed before running the pilot and must be reported in the paper.
"""
import glob
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
ref = sorted(glob.glob(str(ROOT / "results/b0_ablation/sc_attention_noconsistency_*/fold_metrics.csv")))[0]
ref_f1 = pd.read_csv(ref).set_index("fold").loc[0, "macro_f1"]
print(f"Reference attention-only fold-0 macro-F1: {ref_f1:.4f}")
for d in sys.argv[1:]:
    d = Path(d)
    oof = pd.read_csv(d / "oof_predictions.csv")
    oof = oof[oof.fold == 0]
    f1 = f1_score(oof.y_true, oof.y_pred, average="macro")
    mm, fb = oof.mask_mean.mean(), oof.mask_frac_below_05.mean()
    ok = mm <= 0.70 and fb >= 0.20 and (ref_f1 - f1) <= 0.02
    print(f"{d.name}: macro-F1 {f1:.4f} (Δ {100*(f1-ref_f1):+.2f} pp), mask mean {mm:.3f}, "
          f"fraction<0.5 {fb:.3f} -> {'PASS' if ok else 'FAIL'}")
