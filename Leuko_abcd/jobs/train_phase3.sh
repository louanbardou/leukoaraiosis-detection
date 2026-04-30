#!/bin/bash
#SBATCH --job-name=leuko_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE="/mnt/home/lbardou/leukoaraiosis-detection/Leuko_abcd"

source "$WORKSPACE/activate_env.sh" || { echo "ERROR: activate_env.sh failed"; exit 1; }
cd "$WORKSPACE"              || { echo "ERROR: cannot cd to $WORKSPACE"; exit 1; }
mkdir -p logs

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

# ── Run ───────────────────────────────────────────────────────────────────────
RUN_DIR="runs/phase3_$(date +%Y%m%d_%H%M%S)"

python "$WORKSPACE/phase3_train.py" \
    --manifest    "$WORKSPACE/manifest_balanced.csv" \
    --out_dir     "$WORKSPACE/$RUN_DIR" \
    --epochs      100 \
    --batch_size  4 \
    --lr          1e-4 \
    --feature_size 48 \
    --fold        0 \
    --seed        42

echo "Done. Checkpoint in $RUN_DIR/best_model.pt"
