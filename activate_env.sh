#!/bin/bash
# Activate the Leuko pipeline environment.
# Source this at the top of any training script or SLURM job:
#
#   source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh
#
# What it does:
#   1. Loads CUDA 12.5 module (matches PyTorch cu124 wheel — driver compatible)
#   2. Activates the Python venv
#   3. Exports convenience paths

WORKSPACE="/wynton/home/sugrue/loubard/workspace"
VENV="$WORKSPACE/leuko_env"

# Load CUDA (required on compute nodes; harmless on login node)
module load cuda/12.5 2>/dev/null || echo "[warn] cuda/12.5 module not available — skipping"

# Activate venv
source "$VENV/bin/activate"

# Convenience exports
export LEUKO_WORKSPACE="$WORKSPACE/Leuko_abcd"
export ABCD_TABULATED="/wynton/group/abcd/6.1/tabulated"
export ABCD_IMAGING="/wynton/group/abcd/6.0/imaging/derivatives/mproc"
export ABCD_CONCAT="/wynton/group/abcd/6.1/concat"

echo "Leuko env activated — Python $(python --version 2>&1 | cut -d' ' -f2)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not visible (login node)")')"
