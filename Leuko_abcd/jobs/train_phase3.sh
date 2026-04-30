#!/bin/bash
#SBATCH --job-name=leuko_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/train_%j.out
#SBATCH --error=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/train_%j.err

source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh
cd /wynton/home/sugrue/loubard/workspace/Leuko_abcd

RUN_DIR="runs/phase3_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

python phase3_train.py \
    --manifest    manifest.csv \
    --out_dir     "$RUN_DIR" \
    --epochs      100 \
    --batch_size  4 \
    --lr          1e-4 \
    --feature_size 48 \
    --fold        0 \
    --seed        42

echo "Done. Checkpoint in $RUN_DIR/best_model.pt"
