#!/bin/bash
# Phase 1.1 — Environment setup for Leukoaraiosis detection pipeline
# Run this on a login node (no GPU needed for install).
# All packages are installed into workspace/leuko_env (write-safe location).
#
# Usage: bash install_env.sh 2>&1 | tee install_log.txt

set -e
WORKSPACE="/wynton/home/sugrue/loubard/workspace"
VENV="$WORKSPACE/leuko_env"
LOG="$WORKSPACE/Leuko_abcd/install_log.txt"

echo "====================================================="
echo " Leuko pipeline — environment install"
echo " Venv : $VENV"
echo " Python: $(python3 --version)"
echo " Date  : $(date)"
echo "====================================================="

# Activate the venv
source "$VENV/bin/activate"

# ── 1. Upgrade pip / build tools ─────────────────────────────────────────────
echo ""
echo "[1/5] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ── 2. PyTorch with CUDA 12.4 (compatible with CUDA 12.5 driver on cluster) ──
echo ""
echo "[2/5] Installing PyTorch 2.5.1 + torchvision (cu124 wheel)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ── 3. MONAI with required extras for Swin UNETR ─────────────────────────────
echo ""
echo "[3/5] Installing MONAI (nibabel, einops, skimage, tqdm, pandas)..."
pip install "monai[nibabel,skimage,tqdm,einops,pandas,pillow]"

# ── 4. LibAUC — differentiable AUPREC loss + SOAP optimizer ──────────────────
echo ""
echo "[4/5] Installing LibAUC..."
pip install libauc

# ── 5. Nilearn — 3D neuroimaging visualization ───────────────────────────────
echo ""
echo "[5/5] Installing nilearn + matplotlib..."
pip install nilearn matplotlib scipy scikit-learn

# ── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "====================================================="
echo " Verification"
echo "====================================================="
python3 - << 'PYEOF'
import sys
print(f"Python : {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"  CUDA available (CPU check): {torch.cuda.is_available()} "
      f"(False on login node — OK)")

import monai
print(f"MONAI  : {monai.__version__}")

import libauc
print(f"LibAUC : {libauc.__version__}")

import nilearn
print(f"Nilearn: {nilearn.__version__}")

# Swin UNETR importable?
from monai.networks.nets import SwinUNETR
print("SwinUNETR: importable OK")

# LibAUC AUPREC loss importable?
from libauc.losses import APLoss
print("APLoss (AUPREC surrogate): importable OK")

# LibAUC SOAP optimizer importable?
from libauc.optimizers import SOAP
print("SOAP optimizer: importable OK")

# Nilearn plotting importable?
from nilearn import plotting
print("nilearn.plotting: importable OK")

print()
print("All checks passed.")
PYEOF

echo ""
echo "Done. Activate environment with:"
echo "  source $VENV/bin/activate"
