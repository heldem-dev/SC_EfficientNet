# Revision v2 experiment plan for Windows / PowerShell, single RTX 5090 (laptop).
# Run each phase separately from the repository root, e.g.:  .\run_v2.ps1 -Phase 1
# Keep the laptop on AC power, in performance mode, with sleep and automatic restarts disabled.
# Do NOT enable mixed precision: existing results were trained in FP32 and must stay comparable.
param([int]$Phase = 1, [string]$Area = "0.5", [int]$Workers = 6)
$ErrorActionPreference = "Stop"
function Newest($pattern) { (Get-ChildItem -Directory $pattern | Sort-Object LastWriteTime | Select-Object -Last 1).FullName }

if ($Phase -eq 1) {
  # B4/384 baseline without SC components (closes R2.3 / R2.4), then TTA and artifact test on it.
  python training/train_cv_unified.py --model baseline --backbone b4 --img_size 384 --sampler --batch_size 16 --seed 42 --num_workers $Workers
  $ck = Newest "checkpoints\baseline_b4_ce_sampler_seed42_*"
  python -m evaluation.evaluate_artifact_robustness --checkpoint_dir $ck --img_size 384 --batch_size 16 --num_workers $Workers --output_name b4_baseline_artifact
  python -m evaluation.evaluate_tta --checkpoint_dir $ck --img_size 384 --batch_size 16 --num_workers $Workers --output_name b4_baseline_tta
}
if ($Phase -eq 2) {
  # Pilot: SC-v2 on fold 0 only, B0, seed 42. Three pre-specified variants.
  python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b0 --img_size 224 --lambda_c 0.1 --area_target 0.5 --w_bin 0.1 --fold_ids 0 --seed 42 --num_workers $Workers --run_name pilot_scv2_area0.5_bin0.1
  python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b0 --img_size 224 --lambda_c 0.1 --area_target 0.3 --w_bin 0.1 --fold_ids 0 --seed 42 --num_workers $Workers --run_name pilot_scv2_area0.3_bin0.1
  python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b0 --img_size 224 --lambda_c 0.1 --area_target 0.5 --w_bin 0.5 --fold_ids 0 --seed 42 --num_workers $Workers --run_name pilot_scv2_area0.5_bin0.5
  python analysis/pilot_check.py (Newest "results\pilot_scv2_area0.5_bin0.1_*") (Newest "results\pilot_scv2_area0.3_bin0.1_*") (Newest "results\pilot_scv2_area0.5_bin0.5_*")
}
if ($Phase -eq 3) {
  # B0 multi-seed for existing configurations (independent of the pilot outcome). 45 fold runs.
  foreach ($s in 42, 43, 44) {
    python training/train_cv_unified.py --model baseline --backbone b0 --img_size 224 --seed $s --num_workers $Workers
    python training/train_cv_unified.py --model sc --backbone b0 --img_size 224 --lambda_c 0 --seed $s --num_workers $Workers
    python training/train_cv_unified.py --model sc --backbone b0 --img_size 224 --lambda_c 0.1 --seed $s --num_workers $Workers
  }
}
if ($Phase -eq 4) {
  # Only if the pilot passed. Use the variant chosen in Phase 2 (-Area, and edit --w_bin if needed).
  foreach ($s in 42, 43, 44) {
    python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b0 --img_size 224 --lambda_c 0.1 --area_target $Area --w_bin 0.1 --seed $s --num_workers $Workers
  }
  python training/train_cv_unified.py --model sc --objective sc_v2 --backbone b4 --img_size 384 --sampler --batch_size 16 --lambda_c 0.1 --area_target $Area --w_bin 0.1 --seed 42 --num_workers $Workers
  $ck = Newest "checkpoints\scv2_b4_ce_lambda0.1_area$($Area)_sampler_seed42_*"
  python -m evaluation.evaluate_artifact_robustness --checkpoint_dir $ck --img_size 384 --batch_size 16 --num_workers $Workers --output_name b4_scv2_artifact
  python -m evaluation.evaluate_tta --checkpoint_dir $ck --img_size 384 --batch_size 16 --num_workers $Workers --output_name b4_scv2_tta
  python -m evaluation.evaluate_xai_attention_gradcam --checkpoint_dir $ck --output_name b4_scv2_xai
}
if ($Phase -eq 6) {
  # Control: B4/384 with the SC feature path (no 1x1 head conv) but no attention/consistency.
  python training/train_cv_unified.py --model headless --backbone b4 --img_size 384 --sampler --batch_size 16 --seed 42 --num_workers $Workers
}
if ($Phase -eq 7) {
  # Additional seeds for the B4 comparison (baseline, headless, SC-v1).
  foreach ($s in 43, 44) {
    python training/train_cv_unified.py --model baseline --backbone b4 --img_size 384 --sampler --batch_size 16 --seed $s --num_workers $Workers
    python training/train_cv_unified.py --model headless --backbone b4 --img_size 384 --sampler --batch_size 16 --seed $s --num_workers $Workers
    python training/train_cv_unified.py --model sc --backbone b4 --img_size 384 --sampler --batch_size 16 --lambda_c 0.1 --seed $s --num_workers $Workers
  }
}
if ($Phase -eq 5) {
  python analysis/compare_runs.py --area $Area
}
