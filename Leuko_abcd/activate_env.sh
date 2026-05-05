#!/bin/bash
# activate_env.sh
# ---------------
# Activates the Python environment and exports the standard path variables
# used by all pipeline scripts.
#
# Source this at the top of every interactive session and at the top of
# every SLURM job script:
#
#     source /mnt/fac/CX500007_DS1/bardou/leukoaraiosis-detection/Leuko_abcd/activate_env.sh

WORKSPACE="/home/remote/lbardou"
VENV="/mnt/scratch/user/lbardou/leuko_env"

# Load the CUDA module. Required on compute nodes; the warning is suppressed
# on login nodes where the module may not be available.
module load cuda/12.5 2>/dev/null || echo "[warn] cuda/12.5 module not available, skipping"

# Activate the Python virtual environment
source "$VENV/bin/activate"

# Export convenience path variables so scripts can reference them without
# hardcoding cluster-specific paths
export LEUKO_WORKSPACE="$WORKSPACE/leukoaraiosis-detection/Leuko_abcd"
export ABCD_IMAGING="/mnt/scratch/user/lbardou/abcd_leuko"

echo "Environment activated"
echo "  Python : $(python --version 2>&1 | cut -d' ' -f2)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not visible on login node")')"
