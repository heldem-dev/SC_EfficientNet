# Collapse checks of the KL-only formulation (manuscript Section 4.2)

EfficientNet-B0, 224 × 224, fold 0 of `splits/stratified_group_5fold.csv`, seed 42, KL-only objective
(`--model sc`, classification loss on the original path + λ · KL toward the masked path). Each run changes one
setting of the reference run (`results/sc_b0_ce_lambda0.1_seed42_*`, λ = 0.10):

| Folder | Change |
|---|---|
| `sc_b0_ce_lambda1_seed42_*` | λ = 1.0 |
| `sc_b0_ce_lambda5_seed42_*` | λ = 5.0 |
| `sc_b0_ce_lambda5_ep40_seed42_*` | λ = 5.0, 40 epochs |
| `sc_b0_ce_lambda0.1_sharedhead_seed42_*` | masked path uses the linear head of the original path (`--shared_head`) |
| `sc_b0_ce_lambda0.1_attnzero_seed42_*` | attention convolution initialized to zero, mask = 0.5 at the start (`--attn_init zero`) |

`collapse_check_summary.csv` is produced by `python analysis/collapse_check_summary.py --runs_dir results`.
Mask statistics are computed on the 2,014 held-out images of fold 0 (`oof_predictions.csv`, columns
`mask_mean` and `mask_frac_below_05`).
