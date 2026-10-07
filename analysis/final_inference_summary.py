"""Summaries of the inference-only analyses: official ISIC 2018 test set and multi-seed artifact robustness.

Usage: python analysis/final_inference_summary.py
Outputs (results/statistics/): official_test_summary.csv, official_test_comparisons.csv,
artifact_multiseed_summary.csv, artifact_multiseed_comparisons.csv
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from compare_runs import fast_acc, fast_bacc, fast_f1  # noqa: E402

RES, OUT = ROOT / "results", ROOT / "results" / "statistics"
CFG = {"baseline_b4_ce_sampler": "B4 baseline", "headless_b4_ce_sampler": "B4 headless",
       "sc_b4_ce_lambda0.1_sampler": "B4 SC-v1", "sc_b4_384_mac": "B4 SC-v1",
       "scv2_b4_ce_lambda0.1_area0.5_sampler": "B4 SC-v2"}
B = 2000


def cfg_seed(name):
    base = re.sub(r"_(seed\d+_)?\d{8}_\d{6}$", "", name)
    s = re.search(r"seed(\d+)", name)
    return CFG.get(base), int(s.group(1)) if s else 42


def official_test():
    m = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "official_test*" / "official_test_metrics.csv"))])
    m[["config", "seed"]] = m.run.apply(lambda r: pd.Series(cfg_seed(r)))
    m = m.dropna(subset=["config"])
    met = ["accuracy", "balanced_accuracy", "macro_f1"]
    fold = m[m.fold_model != "ensemble"].groupby(["config", "seed"])[met].mean().reset_index()
    ens = m[m.fold_model == "ensemble"][["config", "seed"] + met]
    rows = []
    for c in ["B4 baseline", "B4 headless", "B4 SC-v1", "B4 SC-v2"]:
        r = {"config": c, "n_seeds": int((fold.config == c).sum())}
        for k in met:
            r[f"fold_{k}_mean"] = 100 * fold[fold.config == c][k].mean(); r[f"fold_{k}_sd"] = 100 * fold[fold.config == c][k].std()
            r[f"ens_{k}_mean"] = 100 * ens[ens.config == c][k].mean(); r[f"ens_{k}_sd"] = 100 * ens[ens.config == c][k].std()
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "official_test_summary.csv", index=False)
    # paired comparisons of the five-fold ensembles (image-level bootstrap; lesion ids unavailable)
    P = {}
    for f in glob.glob(str(RES / "official_test*" / "*_ensemble_predictions.csv")):
        c, s = cfg_seed(Path(f).name.replace("_ensemble_predictions.csv", ""))
        if c: P[(c, s)] = pd.read_csv(f).sort_values("image")
    y = next(iter(P.values())).y_true.values
    rng = np.random.default_rng(42); S = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    comp = []
    for a, b in [("B4 headless", "B4 baseline"), ("B4 SC-v1", "B4 headless"), ("B4 SC-v2", "B4 headless"),
                 ("B4 SC-v2", "B4 SC-v1"), ("B4 SC-v1", "B4 baseline"), ("B4 SC-v2", "B4 baseline")]:
        for s in (42, 43, 44):
            if (a, s) not in P or (b, s) not in P: continue
            pa, pb = P[(a, s)].y_pred.values, P[(b, s)].y_pred.values
            r = {"config": a, "reference": b, "seed": s}
            for k, f in [("accuracy", fast_acc), ("macro_f1", fast_f1)]:
                d = [f(y[i], pa[i]) - f(y[i], pb[i]) for i in S]
                r[f"d_{k}"] = f(y, pa) - f(y, pb); r[f"d_{k}_lo"], r[f"d_{k}_hi"] = np.percentile(d, [2.5, 97.5])
            fa = fold[(fold.config == a) & (fold.seed == s)].iloc[0]; fb = fold[(fold.config == b) & (fold.seed == s)].iloc[0]
            r["d_fold_mean_accuracy"] = fa.accuracy - fb.accuracy; r["d_fold_mean_macro_f1"] = fa.macro_f1 - fb.macro_f1
            n01 = int(((pa == y) & (pb != y)).sum()); n10 = int(((pa != y) & (pb == y)).sum())
            r["mcnemar_p"] = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
            comp.append(r)
    pd.DataFrame(comp).to_csv(OUT / "official_test_comparisons.csv", index=False)


def artifacts():
    runs = {("B4 baseline", 42): "b4_baseline_artifact", ("B4 SC-v1", 42): "artifact", ("B4 SC-v2", 42): "b4_scv2_artifact"}
    for d in glob.glob(str(RES / "artifact_*")):
        c, s = cfg_seed(Path(d).name.replace("artifact_", "", 1))
        if c: runs[(c, s)] = Path(d).name
    split = pd.read_csv(ROOT / "splits" / "stratified_group_5fold.csv").set_index("image_id").sort_index()
    ids = split.index; les = split.lesion_id.values; u = np.unique(les); pos = {l: np.where(les == l)[0] for l in u}
    rng = np.random.default_rng(42); S = [np.concatenate([pos[l] for l in rng.choice(u, len(u))]) for _ in range(1000)]
    pred, rows = {}, []
    for (c, s), d in runs.items():
        a = pd.read_csv(RES / d / "oof_predictions_artifact.csv")
        pred[(c, s)] = {k: g.set_index("image_id").loc[ids] for k, g in a.groupby("artifact")}
        y = pred[(c, s)]["clean"].y_true.values
        for art in ["hair", "ruler", "occlusion"]:
            pc, pa = pred[(c, s)]["clean"].y_pred.values, pred[(c, s)][art].y_pred.values
            rows.append({"config": c, "seed": s, "artifact": art, "drop_accuracy": 100 * (fast_acc(y, pc) - fast_acc(y, pa)),
                         "drop_macro_f1": 100 * (fast_f1(y, pc) - fast_f1(y, pa))})
    df = pd.DataFrame(rows); df.to_csv(OUT / "artifact_multiseed_per_run.csv", index=False)
    df.groupby(["config", "artifact"])[["drop_accuracy", "drop_macro_f1"]].agg(["mean", "std", "count"]).to_csv(OUT / "artifact_multiseed_summary.csv")
    comp = []
    for a, b in [("B4 SC-v1", "B4 baseline"), ("B4 SC-v2", "B4 baseline"), ("B4 SC-v2", "B4 SC-v1")]:
        for s in (42, 43, 44):
            if (a, s) not in pred or (b, s) not in pred: continue
            A, Bp = pred[(a, s)], pred[(b, s)]; y = A["clean"].y_true.values
            for art in ["hair", "ruler", "occlusion"]:
                def dd(i):
                    return (fast_f1(y[i], A["clean"].y_pred.values[i]) - fast_f1(y[i], A[art].y_pred.values[i])) - \
                           (fast_f1(y[i], Bp["clean"].y_pred.values[i]) - fast_f1(y[i], Bp[art].y_pred.values[i]))
                full = dd(np.arange(len(y))); ds = [dd(i) for i in S]
                comp.append({"config": a, "reference": b, "seed": s, "artifact": art, "diff_drop_macro_f1": 100 * full,
                             "lo": 100 * np.percentile(ds, 2.5), "hi": 100 * np.percentile(ds, 97.5)})
    pd.DataFrame(comp).to_csv(OUT / "artifact_multiseed_comparisons.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    official_test(); artifacts(); print("done")
