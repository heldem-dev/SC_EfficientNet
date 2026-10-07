# Reproducibility notes

The reported revision experiments use a fixed lesion-level five-fold split in `splits/stratified_group_5fold.csv` and seed 42.

The canonical training entry point is `training/train_cv_unified.py`. It supports EfficientNet-B0/B4, cross-entropy or focal loss, weighted random sampling, and the SC-EfficientNet consistency objective.

The primary configuration is:

- SC-EfficientNet-B4
- 384 × 384 input
- cross-entropy + KL consistency
- λ = 0.10
- inverse-frequency WeightedRandomSampler with replacement
- 20 epochs
- Adam, learning rate 1e-4
- batch size 16
- seed 42
- five lesion-level folds

The repository contains exported predictions and metrics for the reported runs. Trained checkpoints and the image dataset are not distributed.

Exact package versions from the original runs were not captured in a lock file. A future rerun should record the environment with `python -m pip freeze`.

Post-hoc analyses (`analysis/bootstrap_statistics.py`, `analysis/model_complexity.py`) run on the exported files only and write to `results/statistics/`. The bootstrap uses seed 42 and 2,000 lesion-level resamples.

The final-epoch (epoch 20) model of each fold is evaluated; no checkpoint is selected on the held-out fold.
