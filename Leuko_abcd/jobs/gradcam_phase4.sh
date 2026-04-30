#!/bin/bash
#SBATCH --job-name=leuko_gradcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/gradcam_%j.out
#SBATCH --error=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/gradcam_%j.err

# Edit this variable before submitting: fill in the path to your best_model.pt
CHECKPOINT="runs/phase3_YYYYMMDD_HHMMSS/best_model.pt"

source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh
cd /wynton/home/sugrue/loubard/workspace/Leuko_abcd

OUT_DIR="heatmaps/gradcam_$(date +%Y%m%d)"

python phase4_gradcam.py \
    --checkpoint   "$CHECKPOINT" \
    --manifest     manifest.csv \
    --out_dir      "$OUT_DIR" \
    --feature_size 48

echo "Heatmaps saved to $OUT_DIR"
