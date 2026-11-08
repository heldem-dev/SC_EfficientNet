# Collapse checks of the KL-only formulation (manuscript Section 4.2): EfficientNet-B0, fold 0, seed 42.
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\run_collapse_check.ps1
# Four runs of 20 epochs (fold 0 only) and an extended 40-epoch run with lambda = 5.0; FP32 and default settings, as in the B0 runs.
# Summary afterwards: python analysis/collapse_check_summary.py --runs_dir results  (run folders in results/ or results/collapse_check/)
param([int]$Workers = 6)
$start = Get-Date
$common = "--model sc --objective sc --backbone b0 --img_size 224 --fold_ids 0 --seed 42 --num_workers $Workers"

# 1-2) larger consistency weights (default in the paper: lambda = 0.10)
python training/train_cv_unified.py $common.Split(" ") --lambda_c 1.0  2>&1 | Tee-Object -FilePath collapse_check.log -Append
python training/train_cv_unified.py $common.Split(" ") --lambda_c 5.0  2>&1 | Tee-Object -FilePath collapse_check.log -Append
# 3) masked path shares the linear head of the original path
python training/train_cv_unified.py $common.Split(" ") --lambda_c 0.1 --shared_head  2>&1 | Tee-Object -FilePath collapse_check.log -Append
# 4) attention convolution initialised to zero (mask = 0.5 everywhere at the start)
python training/train_cv_unified.py $common.Split(" ") --lambda_c 0.1 --attn_init zero  2>&1 | Tee-Object -FilePath collapse_check.log -Append
# 5) lambda = 5.0 extended to 40 epochs (the mask was still drifting after 20 epochs)
python training/train_cv_unified.py $common.Split(" ") --lambda_c 5.0 --epochs 40 --run_name sc_b0_ce_lambda5_ep40  2>&1 | Tee-Object -FilePath collapse_check.log -Append

$dirs = Get-ChildItem results -Directory | Where-Object { $_.CreationTime -gt $start -and $_.Name -like "sc_b0_*" }
$dirs | ForEach-Object { Write-Host "result folder:" $_.Name }
Compress-Archive -Path (@($dirs.FullName) + (Resolve-Path collapse_check.log).Path) -DestinationPath collapse_check_results.zip -Force
Write-Host "Done -> collapse_check_results.zip"
