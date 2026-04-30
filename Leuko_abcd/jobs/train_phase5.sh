#!/bin/bash
#SBATCH --job-name=leuko_phase5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase5_%j.out
#SBATCH --error=logs/phase5_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE="/mnt/home/lbardou/leukoaraiosis-detection/Leuko_abcd"

# Edit these two variables before submitting
CHECKPOINT="runs/phase3_YYYYMMDD_HHMMSS/best_model.pt"   # path to Phase 3 checkpoint
HEATMAP_DIR="heatmaps/gradcam_YYYYMMDD"                   # path to Phase 4 heatmap output

source "$WORKSPACE/activate_env.sh"
cd "$WORKSPACE"
mkdir -p logs

# ── Run ───────────────────────────────────────────────────────────────────────
python phase5_pseudomask.py all \
    --heatmap_dir  "$HEATMAP_DIR" \
    --mask_dir     heatmaps/pseudo_masks \
    --manifest     manifest.csv \
    --checkpoint   "$CHECKPOINT" \
    --out_dir      runs/phase5 \
    --epochs       50 \
    --lr           5e-5 \
    --batch_size   4 \
    --feature_size 48

echo "Phase 5 complete."
