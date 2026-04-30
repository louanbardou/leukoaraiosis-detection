#!/bin/bash
#SBATCH --job-name=leuko_gradcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gradcam_%j.out
#SBATCH --error=logs/gradcam_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE="/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd"

# Edit this variable before submitting: fill in the path to your best_model.pt
CHECKPOINT="runs/phase3_YYYYMMDD_HHMMSS/best_model.pt"

source "$WORKSPACE/activate_env.sh"
cd "$WORKSPACE"
mkdir -p logs

# ── Run ───────────────────────────────────────────────────────────────────────
OUT_DIR="heatmaps/gradcam_$(date +%Y%m%d)"

python phase4_gradcam.py \
    --checkpoint   "$CHECKPOINT" \
    --manifest     manifest.csv \
    --out_dir      "$OUT_DIR" \
    --feature_size 48

echo "Heatmaps saved to $OUT_DIR"
