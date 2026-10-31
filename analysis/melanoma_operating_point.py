"""Post hoc melanoma operating-point analysis (manuscript Section 5.2).

Uses only stored predictions (no retraining):
  * out-of-fold predictions of each SC-EfficientNet-B4 run  (<runs_dir>/scv2_b4*seed{S}_*/oof_predictions.csv)
  * five-fold ensemble predictions on the official ISIC 2018 test set
    (<test_dir>/scv2_b4*seed{S}_*_ensemble_predictions.csv)
For each seed, a melanoma probability threshold is chosen on the OOF predictions so that
OOF melanoma sensitivity equals the target, and is applied unchanged to the test ensemble.

Example:
  python analysis/melanoma_operating_point.py --runs_dir results --test_dir results/official_test \
      --gt ISIC2018_Task3_Test_GroundTruth.csv --out results/melanoma_operating_point.csv
"""
import argparse, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs_dir', required=True); ap.add_argument('--test_dir', required=True)
    ap.add_argument('--gt', required=True); ap.add_argument('--out', default='melanoma_operating_point.csv')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--targets', type=float, nargs='+', default=[0.80, 0.90])
    a = ap.parse_args()

    gt = pd.read_csv(a.gt); gt['y'] = gt[CLASSES].values.argmax(1)
    rows = []
    for s in a.seeds:
        oof = pd.read_csv(glob.glob(f'{a.runs_dir}/scv2_b4*seed{s}_*/oof_predictions.csv')[0])
        te = pd.read_csv(glob.glob(f'{a.test_dir}/scv2_b4*seed{s}_*_ensemble_predictions.csv')[0]).merge(gt[['image', 'y']], on='image')
        ym, pm = (oof.y_true == 0).values, oof.prob_MEL.values
        yt, pt = (te.y == 0).values, te.prob_MEL.values
        r = dict(seed=s, oof_auc=roc_auc_score(ym, pm), test_auc=roc_auc_score(yt, pt),
                 oof_sens_argmax=((oof.y_pred == 0) & ym).sum() / ym.sum(),
                 oof_spec_argmax=((oof.y_pred != 0) & ~ym).sum() / (~ym).sum(),
                 test_sens_argmax=((te.y_pred == 0) & yt).sum() / yt.sum(),
                 test_spec_argmax=((te.y_pred != 0) & ~yt).sum() / (~yt).sum())
        for tgt in a.targets:
            k = int(round(tgt * 100)); t = np.quantile(pm[ym], 1 - tgt); flag = pt >= t
            r.update({f'thr{k}': t, f'oof_spec{k}': (pm[~ym] < t).mean(),
                      f'test_sens{k}': flag[yt].mean(), f'test_spec{k}': (~flag[~yt]).mean(),
                      f'test_flagged{k}': flag.mean(), f'test_ppv{k}': yt[flag].mean(),
                      f'test_flagged_per_detected{k}': flag.sum() / (flag & yt).sum()})
        rows.append(r)
    df = pd.DataFrame(rows)
    summary = pd.concat([df, df.drop(columns='seed').agg(['mean', 'std']).reset_index().rename(columns={'index': 'seed'})])
    summary.to_csv(a.out, index=False)
    print(summary.set_index('seed').T.round(4))

if __name__ == '__main__':
    main()
