#!/bin/bash
#SBATCH --job-name=leuko_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd
#SBATCH --output=/mnt/scratch/user/lbardou/leuko_logs/train_%j.out
#SBATCH --error=/mnt/scratch/user/lbardou/leuko_logs/train_%j.err

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="/home/remote/lbardou/leukoaraiosis-detection/Leuko_abcd"

source "$SCRIPT_DIR/activate_env.sh"

# W&B — offline mode (compute nodes have no outbound internet on CHPC)
# All run data is saved locally in the out_dir/wandb/ folder.
# After the job finishes, sync from the login node:
#   wandb sync <RUN_DIR>/wandb/offline-run-*
export WANDB_MODE=offline
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
# Save runs to scratch (home quota is too small for model checkpoints)
RUN_DIR="/mnt/scratch/user/lbardou/leuko_runs/phase3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# SSL pretrained SwinUNETR weights (download once on login node if missing):
#   wget -O "$SSL_WEIGHTS" \
#     "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
SSL_WEIGHTS="/mnt/scratch/user/lbardou/swin_ssl_pretrained.pth"
if [ ! -f "$SSL_WEIGHTS" ]; then
    echo "WARNING: SSL pretrained weights not found at $SSL_WEIGHTS — training from scratch"
    PRETRAINED_ARG=""
else
    echo "SSL pretrained weights found: $SSL_WEIGHTS"
    PRETRAINED_ARG="--pretrained_weights $SSL_WEIGHTS"
fi

python "$WORKSPACE/phase3_train.py" \
    --manifest           "$WORKSPACE/data/manifest_full.csv" \
    --out_dir            "$RUN_DIR" \
    --epochs             80 \
    --batch_size         4 \
    --lr                 1e-4 \
    --feature_size       48 \
    --fold               0 \
    --seed               42 \
    --cache_dir          /mnt/scratch/user/lbardou/leuko_cache \
    --dropout            0.4 \
    --weight_decay       5e-4 \
    --freeze_epochs      10 \
    --aploss_gamma       0.1 \
    --epoch_decay        1e-3 \
    --wandb_project      leuko-abcd \
    $PRETRAINED_ARG

echo "Done. Checkpoint in $RUN_DIR/best_model.pt"
