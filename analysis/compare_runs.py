"""Multi-seed and cross-model comparison of exported OOF predictions (revision v2).

Usage:  python analysis/compare_runs.py --area 0.5 [--manifest analysis/manifest_v2.json]

Outputs (results/statistics/):
  v2_per_run.csv            metrics of every run (config, seed)
  v2_config_summary.csv     mean ± SD over seeds per configuration (+ mask statistics)
  v2_comparisons.csv        paired comparisons per seed: lesion-level bootstrap CI + McNemar,
                            plus the across-seed mean difference
  v2_artifact.csv           drop in macro-F1 / accuracy under each perturbation and the
                            paired bootstrap CI for the difference in drop between models
Missing runs are skipped with a warning, so the script can be run at any stage.
"""
import argparse
import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
B = 2000


def find(pattern):
    hits = sorted(glob.glob(str(RES / pattern)))
    hits = [h for h in hits if Path(h).is_dir()]
    if not hits:
        return None
    if len(hits) > 1:
        warnings.warn(f"{pattern}: {len(hits)} matches, using the newest: {hits[-1]}")
    return Path(hits[-1])


def load_oof(run_dir):
    f = run_dir / "oof_predictions.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).set_index("image_id").sort_index()


K = 7


def _cm(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)


def fast_acc(y, p):
    return float((y == p).mean())


def fast_bacc(y, p):
    cm = _cm(y, p); support = cm.sum(1)
    return float(np.mean(np.diag(cm)[support > 0] / support[support > 0]))


def fast_f1(y, p):
    # identical to sklearn f1_score(average="macro", zero_division=0) for labels 0..K-1 present in y or p
    cm = _cm(y, p); tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    present = (cm.sum(0) + cm.sum(1)) > 0
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1[present].mean())


def metrics(y, p):
    return {"accuracy": fast_acc(y, p), "balanced_accuracy": fast_bacc(y, p), "macro_f1": fast_f1(y, p)}


def lesion_samples(lesions, rng):
    uniq = np.unique(lesions)
    pos = {l: np.where(lesions == l)[0] for l in uniq}
    return [np.concatenate([pos[l] for l in rng.choice(uniq, len(uniq))]) for _ in range(B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "analysis" / "manifest_v2.json"))
    ap.add_argument("--area", default="0.5", help="area_target used for SC-v2 runs")
    args = ap.parse_args()
    man = json.loads(Path(args.manifest).read_text())
    split = pd.read_csv(ROOT / "splits" / "stratified_group_5fold.csv").set_index("image_id").sort_index()
    ids = split.index
    lesions = split.lesion_id.values
    rng = np.random.default_rng(42)
    samples = lesion_samples(lesions, rng)

    runs = {}  # (config, seed) -> oof df aligned to ids
    for cfg, pat in man["configs"].items():
        for s in man["seeds"]:
            d = find(pat.replace("AREA", args.area).format(seed=s))
            df = load_oof(d) if d else None
            if df is None:
                print(f"[missing] {cfg} seed {s}")
                continue
            runs[(cfg, s)] = df.loc[ids]
    for cfg, pat in man.get("b4_configs", {}).items():
        for s in man.get("b4_seeds", [42]):
            d = find(pat.replace("AREA", args.area).format(seed=s))
            if d is None and s == 42 and cfg in man.get("b4_seed42_fallback", {}):
                d = find(man["b4_seed42_fallback"][cfg])
            df = load_oof(d) if d else None
            if df is None:
                print(f"[missing] {cfg} seed {s}")
                continue
            runs[(cfg, s)] = df.loc[ids]
    for cfg, pat in man["single_seed_configs"].items():
        d = find(pat.replace("AREA", args.area))
        df = load_oof(d) if d else None
        if df is None:
            print(f"[missing] {cfg}")
            continue
        runs[(cfg, 42)] = df.loc[ids]

    y = next(iter(runs.values())).y_true.values if runs else None
    out = RES / "statistics"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for (cfg, s), df in runs.items():
        r = {"config": cfg, "seed": s, **metrics(y, df.y_pred.values)}
        assert abs(r["macro_f1"] - f1_score(y, df.y_pred.values, average="macro")) < 1e-9
        if "mask_mean" in df:
            r["mask_mean"] = df.mask_mean.mean()
            r["mask_frac_below_05"] = df.mask_frac_below_05.mean()
        rows.append(r)
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "v2_per_run.csv", index=False)
    if not per_run.empty:
        agg = per_run.groupby("config").agg(["mean", "std", "count"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.drop(columns=[c for c in agg.columns if c.startswith("seed_")]).to_csv(out / "v2_config_summary.csv")
        print(per_run.groupby("config")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"]).round(4))

    comp = []
    all_seeds = sorted(set(man["seeds"]) | set(man.get("b4_seeds", [])))
    for a, b in man["comparisons"]:
        seeds = [s for s in all_seeds if (a, s) in runs and (b, s) in runs]
        for s in seeds:
            pa, pb = runs[(a, s)].y_pred.values, runs[(b, s)].y_pred.values
            row = {"config": a, "reference": b, "seed": s}
            for m in ["accuracy", "balanced_accuracy", "macro_f1"]:
                f = {"accuracy": fast_acc, "balanced_accuracy": fast_bacc, "macro_f1": fast_f1}[m]
                ds = [f(y[i], pa[i]) - f(y[i], pb[i]) for i in samples]
                row[f"d_{m}"] = f(y, pa) - f(y, pb)
                row[f"d_{m}_ci_low"], row[f"d_{m}_ci_high"] = np.percentile(ds, [2.5, 97.5])
            n01 = int(((pa == y) & (pb != y)).sum()); n10 = int(((pa != y) & (pb == y)).sum())
            row.update({"only_config_correct": n01, "only_reference_correct": n10,
                        "mcnemar_p": binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0})
            comp.append(row)
    comp = pd.DataFrame(comp)
    if not comp.empty:
        g = comp.groupby(["config", "reference"])
        seedwise = g[["d_accuracy", "d_balanced_accuracy", "d_macro_f1"]].agg(["mean", "std", "count"])
        seedwise.columns = ["_".join(c) for c in seedwise.columns]
        seedwise.to_csv(out / "v2_comparisons_across_seeds.csv")
    comp.to_csv(out / "v2_comparisons.csv", index=False)
    if not comp.empty:
        print(comp[["config", "reference", "seed", "d_macro_f1", "d_macro_f1_ci_low", "d_macro_f1_ci_high", "mcnemar_p"]].round(4))

    # ---- artifact comparison
    art = {}
    for name, pat in man.get("artifact_runs", {}).items():
        d = find(pat)
        f = d / "oof_predictions_artifact.csv" if d else None
        if not f or not f.exists():
            print(f"[missing] artifact {name}")
            continue
        a = pd.read_csv(f)
        art[name] = {c: g.set_index("image_id").loc[ids].y_pred.values for c, g in a.groupby("artifact")}
    arows = []
    f1 = fast_f1
    for name, preds in art.items():
        for c in [c for c in preds if c != "clean"]:
            arows.append({"model": name, "artifact": c,
                          "drop_accuracy": accuracy_score(y, preds["clean"]) - accuracy_score(y, preds[c]),
                          "drop_macro_f1": f1(y, preds["clean"]) - f1(y, preds[c])})
    for a, b in man.get("artifact_comparisons", []):
        if a not in art or b not in art:
            continue
        for c in [c for c in art[a] if c != "clean" and c in art[b]]:
            A, Bp = art[a], art[b]
            def ddrop(i):
                return (f1(y[i], A["clean"][i]) - f1(y[i], A[c][i])) - (f1(y[i], Bp["clean"][i]) - f1(y[i], Bp[c][i]))
            full = ddrop(np.arange(len(y)))
            ds = [ddrop(i) for i in samples[:1000]]
            arows.append({"model": f"{a} minus {b}", "artifact": c, "diff_in_drop_macro_f1": full,
                          "ci_low": np.percentile(ds, 2.5), "ci_high": np.percentile(ds, 97.5)})
    pd.DataFrame(arows).to_csv(out / "v2_artifact.csv", index=False)
    if arows:
        print(pd.DataFrame(arows).round(4))
    print("Saved to", out)


if __name__ == "__main__":
    main()
