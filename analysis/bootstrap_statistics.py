"""Post-hoc statistical analysis of the exported out-of-fold (OOF) predictions.

No model is retrained or re-evaluated. The script reads the OOF prediction files
in results/ and computes
  (i)  lesion-level bootstrap 95% confidence intervals for the primary B4 metrics,
  (ii) paired lesion-level bootstrap differences between configurations, and
  (iii) exact McNemar tests on the discordant OOF predictions.
Lesion-level resampling keeps all images of a lesion together, matching the
grouping used for cross-validation.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B = 2000
SEED = 42


def load(pattern):
    path = glob.glob(str(RES / pattern))[0]
    return pd.read_csv(path).set_index("image_id").sort_index()


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
    }


def main():
    split = pd.read_csv(ROOT / "splits" / "stratified_group_5fold.csv").set_index("image_id").sort_index()
    runs = {
        "SC-B0 attention-only": "b0_ablation/sc_attention_noconsistency*/oof_predictions.csv",
        "SC-B0 CE+KL l=0.05": "b0_ablation/sc_lambda005_*/oof_predictions.csv",
        "SC-B0 CE+KL l=0.10": "b0_ablation/sc_lambda010_*/oof_predictions.csv",
        "SC-B0 CE+KL l=0.25": "b0_ablation/sc_lambda025_*/oof_predictions.csv",
        "SC-B0 CE+KL l=0.50": "b0_ablation/sc_lambda050_*/oof_predictions.csv",
        "SC-B0 Focal+KL l=0.10": "b0_ablation/sc_focal_*/oof_predictions.csv",
        "SC-B4/384": "b4_384/oof_predictions.csv",
        "SC-B4/384 + TTA": "tta/oof_predictions_tta.csv",
    }
    preds = {k: load(v) for k, v in runs.items()}
    ids = split.index
    for k, d in preds.items():
        assert set(d.index) == set(ids), k
        preds[k] = d.loc[ids]
    y = split.loc[ids].dx.str.upper().map(
        {c: i for i, c in enumerate(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"])}).values
    for k, d in preds.items():
        assert (d.y_true.values == y).all(), k

    lesions = split.loc[ids].lesion_id.values
    uniq = np.unique(lesions)
    pos = {l: np.where(lesions == l)[0] for l in uniq}
    rng = np.random.default_rng(SEED)
    samples = [np.concatenate([pos[l] for l in rng.choice(uniq, len(uniq))]) for _ in range(B)]

    out = {"n_bootstrap": B, "resampling_unit": "lesion_id", "point_estimates_and_ci": {}, "paired": {}}

    for name in ["SC-B4/384", "SC-B4/384 + TTA"]:
        p = preds[name].y_pred.values
        point = metrics(y, p)
        boot = {m: [] for m in point}
        for s in samples:
            for m, v in metrics(y[s], p[s]).items():
                boot[m].append(v)
        out["point_estimates_and_ci"][name] = {
            m: [point[m], float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))]
            for m in point}

    comparisons = [
        ("SC-B0 CE+KL l=0.05", "SC-B0 attention-only"),
        ("SC-B0 CE+KL l=0.10", "SC-B0 attention-only"),
        ("SC-B0 CE+KL l=0.25", "SC-B0 attention-only"),
        ("SC-B0 CE+KL l=0.50", "SC-B0 attention-only"),
        ("SC-B0 Focal+KL l=0.10", "SC-B0 attention-only"),
        ("SC-B4/384 + TTA", "SC-B4/384"),
    ]
    for a, b in comparisons:
        pa, pb = preds[a].y_pred.values, preds[b].y_pred.values
        res = {}
        for m in ["accuracy", "macro_f1"]:
            f = (lambda yy, pp: accuracy_score(yy, pp)) if m == "accuracy" else (
                lambda yy, pp: f1_score(yy, pp, average="macro"))
            d0 = f(y, pa) - f(y, pb)
            ds = [f(y[s], pa[s]) - f(y[s], pb[s]) for s in samples]
            res[m] = [d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
        n01 = int(((pa == y) & (pb != y)).sum())
        n10 = int(((pa != y) & (pb == y)).sum())
        res["mcnemar"] = {"a_correct_b_wrong": n01, "a_wrong_b_correct": n10,
                          "p_exact": float(binomtest(n01, n01 + n10, 0.5).pvalue)}
        out["paired"][f"{a} vs {b}"] = res

    (RES / "statistics").mkdir(exist_ok=True)
    (RES / "statistics" / "bootstrap_statistics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
