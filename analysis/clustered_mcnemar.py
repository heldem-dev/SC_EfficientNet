"""Lesion-clustered paired tests for the out-of-fold comparisons (Table 5 and Section 4.4).

Images of the same lesion are not independent, so in addition to the standard exact McNemar
test (image level) we report the clustered-data test of Obuchowski (1998), which compares the
paired proportions of correctly classified images with a variance estimated over lesions.
The test of Durkalski et al. (2003), which weights lesions equally, is reported for reference.

Example:
  python analysis/clustered_mcnemar.py --results_dir results --out results/clustered_mcnemar.csv
"""
import argparse, glob
import numpy as np, pandas as pd
from scipy.stats import binomtest, chi2


def paired_tests(ca, cb, lesion):
    ca, cb = np.asarray(ca, bool), np.asarray(cb, bool)
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p_img = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    g = pd.DataFrame({'l': lesion, 'b': (ca & ~cb).astype(int), 'c': (~ca & cb).astype(int),
                      'xa': ca.astype(int), 'xb': cb.astype(int), 'n': 1}).groupby('l').sum()
    d = ((g.b - g.c) / g.n).values                                   # Durkalski et al. (2003)
    xd = d.sum() ** 2 / (d ** 2).sum() if (d ** 2).sum() > 0 else 0.0
    K, N = len(g), g.n.sum(); pa, pb = g.xa.sum() / N, g.xb.sum() / N   # Obuchowski (1998)
    ra, rb = g.xa - g.n * pa, g.xb - g.n * pb
    var = K / ((K - 1) * N ** 2) * ((ra ** 2).sum() + (rb ** 2).sum() - 2 * (ra * rb).sum())
    xo = (pa - pb) ** 2 / var if var > 0 else 0.0
    return dict(diff_acc=100 * (ca.mean() - cb.mean()), n10=n10, n01=n01, p_image=p_img,
                p_obuchowski=chi2.sf(xo, 1), p_durkalski=chi2.sf(xd, 1))


CFG = {('B0', 'baseline'): 'baseline_b0_ce_seed{s}_*', ('B0', 'headless'): 'headless_b0_ce_seed{s}_*',
       ('B0', 'attention-only'): 'sc_b0_ce_lambda0_attention_only_seed{s}_*', ('B0', 'KL-only'): 'sc_b0_ce_lambda0.1_seed{s}_*',
       ('B0', 'SC'): 'scv2_b0_ce_lambda0.1_area0.5_seed{s}_*', ('B0', 'SC without priors'): 'scnoprior_b0_ce_lambda0.1_seed{s}_*',
       ('B4', 'baseline'): 'baseline_b4_ce_sampler_seed{s}_*', ('B4', 'headless'): 'headless_b4_ce_sampler_seed{s}_*',
       ('B4', 'KL-only'): 'sc_b4_ce_lambda0.1_sampler_seed{s}_*', ('B4', 'SC'): 'scv2_b4_ce_lambda0.1_area0.5_sampler_seed{s}_*'}
SPECIAL = {('B4', 'KL-only', 42): 'b4_384'}   # seed-42 KL-only B4 run (earlier script version)
COMP = [('B0', 'headless', 'baseline'), ('B0', 'attention-only', 'headless'), ('B0', 'KL-only', 'headless'),
        ('B0', 'SC', 'headless'), ('B0', 'SC without priors', 'headless'), ('B0', 'SC', 'SC without priors'),
        ('B0', 'SC', 'attention-only'), ('B0', 'SC', 'KL-only'), ('B4', 'headless', 'baseline'),
        ('B4', 'KL-only', 'headless'), ('B4', 'SC', 'headless'), ('B4', 'SC', 'KL-only')]
TTA = {'baseline': ('baseline_b4_ce_sampler_seed42_*', 'b4_baseline_tta_*'),
       'SC': ('scv2_b4_ce_lambda0.1_area0.5_sampler_seed42_*', 'b4_scv2_tta_*'),
       'KL-only': ('b4_384', 'tta')}


def one(pattern, name='oof_predictions.csv'):
    f = glob.glob(f'{R}/{pattern}/{name}'); assert len(f) == 1, (pattern, f); return pd.read_csv(f[0])


def main():
    global R
    ap = argparse.ArgumentParser(); ap.add_argument('--results_dir', default='results')
    ap.add_argument('--out', default='results/clustered_mcnemar.csv'); a = ap.parse_args(); R = a.results_dir
    rows = []
    load = lambda sc, c, s: one(SPECIAL.get((sc, c, s), CFG[(sc, c)].format(s=s))).set_index('image_id')
    for sc, x, y in COMP:
        for s in (42, 43, 44):
            m = load(sc, x, s).join(load(sc, y, s), lsuffix='_a', rsuffix='_b', how='inner')
            rows.append(dict(analysis='OOF', comparison=f'{sc}: {x} - {y}', seed=s,
                             **paired_tests(m.y_pred_a == m.y_true_a, m.y_pred_b == m.y_true_b, m.lesion_id_a.values)))
    lesion = one(CFG[('B4', 'baseline')].format(s=42))[['image_id', 'lesion_id']]
    tt = {}
    for k, (plain, tta) in TTA.items():
        tt[k] = one(plain).drop(columns='lesion_id', errors='ignore').merge(lesion, on='image_id').set_index('image_id')
        tt[k + ' + TTA'] = one(tta, 'oof_predictions_tta.csv').drop(columns='lesion_id', errors='ignore').merge(lesion, on='image_id').set_index('image_id')
    for x, y in [('SC + TTA', 'SC'), ('baseline + TTA', 'baseline'), ('KL-only + TTA', 'KL-only'),
                 ('SC + TTA', 'baseline + TTA'), ('KL-only + TTA', 'baseline + TTA')]:
        m = tt[x].join(tt[y], lsuffix='_a', rsuffix='_b')
        rows.append(dict(analysis='TTA (B4, seed 42)', comparison=f'{x} - {y}', seed=42,
                         **paired_tests(m.y_pred_a == m.y_true_a, m.y_pred_b == m.y_true_b, m.lesion_id_a.values)))
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    f = lambda p: '<0.001' if p < 0.001 else f'{p:.3f}'
    for (an, c), g in df.groupby(['analysis', 'comparison'], sort=False):
        print(f'{c:40s} image: ' + ' / '.join(f(p) for p in g.p_image) + '   clustered (Obuchowski): ' + ' / '.join(f(p) for p in g.p_obuchowski))


if __name__ == '__main__':
    main()
