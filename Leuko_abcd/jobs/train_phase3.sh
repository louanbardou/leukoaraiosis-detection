#!/bin/bash
#SBATCH --job-name=leuko_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd
#SBATCH --output=/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd/logs/train_%j.out
#SBATCH --error=/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd/logs/train_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd"

source "$SCRIPT_DIR/activate_env.sh"
# activate_env.sh overwrites WORKSPACE — use SCRIPT_DIR for our paths
WORKSPACE="$SCRIPT_DIR"
mkdir -p "$WORKSPACE/logs"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

# ── Run ───────────────────────────────────────────────────────────────────────
RUN_DIR="$WORKSPACE/runs/phase3_$(date +%Y%m%d_%H%M%S)"

python "$WORKSPACE/phase3_train.py" \
    --manifest    "$WORKSPACE/manifest_balanced.csv" \
    --out_dir     "$RUN_DIR" \
    --epochs      100 \
    --batch_size  4 \
    --lr          1e-5 \
    --feature_size 48 \
    --fold        0 \
    --seed        42

echo "Done. Checkpoint in $RUN_DIR/best_model.pt"
