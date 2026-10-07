# SC-EfficientNet: Preventing Attention Collapse in Consistency-Regularized Skin Lesion Classification

Code, fixed data splits, exported predictions, and analysis scripts for the manuscript
*“SC-EfficientNet: Preventing Attention Collapse in Consistency-Regularized Skin Lesion Classification”*.

Every metric, confidence interval, and statistical test in the manuscript can be recomputed from the
exported out-of-fold (OOF) predictions in `results/` **without retraining**.

## Overview

SC-EfficientNet (Supervised-Consistency EfficientNet) adds a learned spatial mask to the final EfficientNet
feature map and predicts from an attention-masked path. The masked path is supervised with the classification
loss, regularized toward a stop-gradient copy of the original prediction by a Kullback–Leibler (KL) term, and the
mask is constrained by an area prior and a binarization penalty.

The originally submitted formulation, in which the mask is trained **only** through the KL term, is retained as
the **KL-only ablation**. In every run it collapses to a nearly constant mask; SC-EfficientNet prevents this collapse.

All experiments use the ISIC 2018 Task 3 training set (10,015 images, 7,470 lesions) with **lesion-level**
five-fold Stratified Group K-Fold cross-validation (`splits/stratified_group_5fold.csv`, grouped by `lesion_id`),
three training seeds (42, 43, 44), EfficientNet-B0 (224 × 224) and EfficientNet-B4 (384 × 384) backbones, and the
official ISIC 2018 Task 3 test set (1,512 images) for independent evaluation.

## Model and folder names

Folder and command-line names in this repository predate the final terminology of the manuscript:

| Manuscript | Command line | Result folders |
|---|---|---|
| Standard baseline (EfficientNet with 1×1 head conv) | `--model baseline` | `baseline_b0_*`, `baseline_b4_*` |
| Headless control (final-stage features → GAP → linear) | `--model headless` | `headless_b0_*`, `headless_b4_*` |
| Attention-only ablation (B0) | `--model sc --lambda_c 0` | `sc_b0_ce_lambda0_attention_only_*` |
| **KL-only ablation** (originally submitted objective) | `--model sc --lambda_c 0.1` | `sc_b0_ce_lambda0.1_*`, `sc_b4_ce_lambda0.1_*`; seed-42 B4 run: `b4_384` |
| **SC-EfficientNet** (proposed) | `--model sc --objective sc_v2 --area_target 0.5 --w_bin 0.1` | `scv2_b0_*`, `scv2_b4_*` |
| SC-EfficientNet without mask priors (B0) | `--model sc --objective sc_v2 --w_area 0 --w_bin 0` | `scnoprior_b0_*` |
| Pre-specified pilot (fold 0) | see `run_v2.ps1`, phase 2 | `pilot_scv2_*` |

In the analysis scripts, `SC-v1` denotes the KL-only ablation and `SC-v2` denotes SC-EfficientNet.

## Repository structure

```
models/        SC-EfficientNet, baseline, headless control, spatial attention, checkpoint factory
training/      train_cv_unified.py (all configurations), train_cv.py (legacy)
utils/         losses / training loop (incl. SC-EfficientNet objective), seeding
data_utils/    dataset and class definitions
evaluation/    TTA, synthetic artifacts, XAI (attention vs. Grad-CAM), official test set, latency
analysis/      statistics (bootstrap, McNemar), multi-seed summaries, pilot gate, model complexity
figures/       figure-generation scripts and exported figures
splits/        fixed lesion-level fold assignment (no lesion in more than one fold)
configs/       configuration files of the original runs
results/       exported OOF predictions, metrics, and statistics of all runs
logs/          training and evaluation logs
run_v2.ps1     the complete revision experiment plan (PowerShell, phases 1–7)
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch with CUDA, and timm. `fvcore` is needed only for `analysis/model_complexity.py`.

### Data (not included)

Download from the ISIC Challenge data page (https://challenge.isic-archive.com/data/#2018), Task 3:

* Training images and ground truth, and the lesion-grouping metadata (`HAM10000_metadata.csv`, containing `lesion_id`).
  Place the images in `skin_cancer_data/<CLASS>/<image_id>.jpg` and the metadata file in the repository root.
* For the independent evaluation: test images (`ISIC2018_Task3_Test_Input/`) and `ISIC2018_Task3_Test_GroundTruth.csv`,
  e.g. in `isic2018_test/`.

Please cite the dataset as Tschandl et al. (2018) and Codella et al. (2019).

## Reproducing the analyses from the exported results (no GPU needed)

```bash
python analysis/compare_runs.py --area 0.5        # Tables 4–5, mask statistics -> results/statistics/v2_*.csv
python analysis/final_inference_summary.py         # official test set (Table 10), multi-seed artifacts (Table 8)
python analysis/model_complexity.py                # parameters and FLOPs (Table 3)
python figures/make_revision_v2_figures.py         # Figures 4, 11, 12, 13
python figures/make_detailed_figures_sc.py         # Figures 5–10 and Tables 6–7 (SC-EfficientNet-B4, seed 42)
python figures/make_schematics.py                  # Figures 1 and 3
```

Bootstrap analyses resample whole lesions (2,000 resamples, seed 42). McNemar tests are exact binomial tests on
discordant predictions.

## Reproducing the experiments (GPU)

Examples (one configuration, one seed; five folds are trained in sequence):

```bash
# SC-EfficientNet-B4
python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b4 --img_size 384 --sampler \
    --batch_size 16 --lambda_c 0.1 --area_target 0.5 --w_bin 0.1 --seed 42
# KL-only ablation, B0
python training/train_cv_unified.py --model sc --backbone b0 --img_size 224 --lambda_c 0.1 --seed 42
# Headless control, B4
python training/train_cv_unified.py --model headless --backbone b4 --img_size 384 --sampler --batch_size 16 --seed 42
```

Evaluation of trained checkpoints:

```bash
python -m evaluation.evaluate_tta --checkpoint_dir checkpoints/<run> --img_size 384 --output_name <name>
python -m evaluation.evaluate_artifact_robustness --checkpoint_dir checkpoints/<run> --img_size 384 --output_name <name>
python -m evaluation.evaluate_xai_attention_gradcam --checkpoint_dir checkpoints/<run> --output_name <name>
python -m evaluation.evaluate_official_test --test_images isic2018_test/ISIC2018_Task3_Test_Input \
    --gt_csv isic2018_test/ISIC2018_Task3_Test_GroundTruth.csv --runs "checkpoints/<pattern>*"
python -m evaluation.benchmark_latency
```

`run_v2.ps1` contains the full sequence used for the revision (Windows PowerShell; `.\run_v2.ps1 -Phase <n>`).
The model type and backbone are read from each checkpoint, so all evaluation scripts work for every configuration.

## Training protocol

Adam, learning rate 1 × 10⁻⁴ (constant), 20 epochs, no backbone freezing, FP32; batch size 32 (B0) or 16 (B4);
weighted random sampling for B4 (inverse class frequency of the training portion of each fold). The model after the
final epoch of each fold is evaluated; no checkpoint is selected on held-out data. The seed also controls data order
and sampling. SC-EfficientNet: λ = 0.10, ρ = 0.5, β_area = 1, β_bin = 0.1, fixed before the pilot
(`analysis/pilot_check.py`, criteria in the manuscript, Section 4.1).

## Notes and limitations

* Multi-seed runs were executed on a single NVIDIA GeForce RTX 5090 Laptop GPU. The seed-42 KL-only B4 run
  (`results/b4_384`) is the original run of that configuration, trained with an earlier version of the training
  script; it does not contain raw mask statistics. An earlier single-seed λ sweep (`results/b0_ablation/`) was run on
  a different workstation. Bit-level reproducibility across hardware is not expected.
* Trained checkpoints are not included because of their size.
* Lesion identifiers are not provided for the official test set, so lesion overlap with the training set cannot be excluded.

## Citation

If you use this code, please cite the manuscript (citation details will be added upon publication) and the ISIC 2018 / HAM10000 dataset papers.
