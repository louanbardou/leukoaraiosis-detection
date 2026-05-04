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

# W&B authentication — set via environment variable (never hardcode in scripts)
# Run once on login node to store permanently:
#   wandb login <your_api_key>
# Or export here if ~/.netrc is not available on compute nodes:
# export WANDB_API_KEY="<your_api_key>"
# activate_env.sh overwrites WORKSPACE — use SCRIPT_DIR for our paths
WORKSPACE="$SCRIPT_DIR"
mkdir -p "$WORKSPACE/logs"

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

# ── Step 1: Build manifest from disk (disk-first, catches all labeled subjects) ──
echo "Building manifest from disk..."
python "$WORKSPACE/scripts/build_manifest_from_disk.py" --data_root "$ABCD_IMAGING" --labels_dir "$WORKSPACE/data" --out_csv "$WORKSPACE/data/manifest_full.csv"

if [ ! -f "$WORKSPACE/data/manifest_full.csv" ]; then
    echo "ERROR: manifest_full.csv was not created. Aborting."
    exit 1
fi
echo "Manifest ready: $(wc -l < "$WORKSPACE/data/manifest_full.csv") rows"

# ── Step 2: Train ─────────────────────────────────────────────────────────────
RUN_DIR="$WORKSPACE/runs/phase3_$(date +%Y%m%d_%H%M%S)"

python "$WORKSPACE/phase3_train.py" \
    --manifest      "$WORKSPACE/data/manifest_full.csv" \
    --out_dir       "$RUN_DIR" \
    --epochs        60 \
    --batch_size    4 \
    --lr            1e-4 \
    --feature_size  48 \
    --fold          0 \
    --seed          42 \
    --cache_dir     /mnt/scratch/user/lbardou/leuko_cache \
    --dropout       0.5 \
    --weight_decay  1e-3 \
    --freeze_epochs 15 \
    --wandb_project leuko-abcd

echo "Done. Checkpoint in $RUN_DIR/best_model.pt"
