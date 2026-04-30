#!/bin/bash
# install_env.sh
# --------------
# Installs all Python dependencies for the leukoaraiosis detection pipeline
# into a virtual environment.
#
# Run ONCE on a login node (no GPU needed for installation):
#
#     bash install_env.sh 2>&1 | tee install_log.txt

set -e

WORKSPACE="/mnt/fac/CX500007_DS1/bardou"
VENV="$WORKSPACE/leuko_env"

# Create the virtual environment if it does not exist
if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    python3 -m venv "$VENV"
fi

echo "Installing pipeline dependencies"
echo "  Virtual env : $VENV"
echo "  Python      : $(python3 --version)"
echo "  Date        : $(date)"
echo ""

source "$VENV/bin/activate"

# Step 1: Upgrade build tools
echo "[1/5] Upgrading pip, setuptools, wheel"
pip install --upgrade pip setuptools wheel

# Step 2: PyTorch with CUDA 12.4 wheels.
# The cluster runs CUDA driver 12.5, which is forward-compatible with the
# CUDA 12.4 runtime. The cu124 index provides the correct binaries.
echo "[2/5] Installing PyTorch 2.5.1 (cu124 wheels)"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 3: MONAI with the extras required by the Swin UNETR and the
# transform pipeline (nibabel for NIfTI, einops for tensor reshaping,
# skimage for Otsu thresholding in Phase 5)
echo "[3/5] Installing MONAI with nibabel, einops, skimage, tqdm, pandas"
pip install "monai[nibabel,skimage,tqdm,einops,pandas,pillow]"

# Step 4: LibAUC provides APLoss (differentiable AUPREC surrogate) and the
# SOAP optimizer that is designed to work with it
echo "[4/5] Installing LibAUC"
pip install libauc

# Step 5: Nilearn for neuroimaging visualisation and matplotlib for figures
echo "[5/5] Installing nilearn, matplotlib, scipy, scikit-learn"
pip install nilearn matplotlib scipy scikit-learn

# Verification: confirm all key imports work correctly
echo ""
echo "Verification"
python3 - << 'PYEOF'
import sys
print(f"Python : {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"  CUDA available (expected False on login node): {torch.cuda.is_available()}")

import monai
print(f"MONAI  : {monai.__version__}")

import libauc
print(f"LibAUC : {libauc.__version__}")

import nilearn
print(f"Nilearn: {nilearn.__version__}")

from monai.networks.nets import SwinUNETR
print("SwinUNETR import: OK")

from libauc.losses import APLoss
print("APLoss import: OK")

from libauc.optimizers import SOAP
print("SOAP optimizer import: OK")

print("\nAll checks passed.")
PYEOF

echo ""
echo "Installation complete. Activate the environment with:"
echo "  source $VENV/bin/activate"
