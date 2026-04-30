# Leukoaraiosis Detection Pipeline — Full Implementation Plan
# Adapted to ABCD Release 6.1 on Wynton HPC

**Date:** 2026-04-10  
**Author:** loubard  
**Cluster:** Wynton (UCSF), SLURM scheduler, A100/V100 GPUs  
**Framework:** PyTorch 2.5.1 + MONAI 1.5.2 + LibAUC 1.4.0  
**Working directory:** `/wynton/home/sugrue/loubard/workspace/Leuko_abcd/`

---

## 1. Scientific Objective

Detect leukoaraiosis (white matter rarefaction) from T1w + T2w MRI pairs using **weakly supervised learning**: a Swin UNETR backbone trained with image-level clinical labels produces Grad-CAM heatmaps that spatially localize pathology — no manual voxel masks required.

**Deliverables:**
1. Binary classifier (AUPREC-optimized) with held-out ROC/PR curves  
2. Per-subject 3D Grad-CAM heatmaps in NIfTI format (MNI space, same resolution as input)  
3. Aggregate group-level heatmap (mean Grad-CAM across positives) for publication  
4. Phase 5 pseudo-mask model for voxel-level probability maps

---

## 2. Dataset — ABCD Release 6.1 on Wynton

### 2.1 Data Locations

| Resource | Path |
|---|---|
| Tabulated data (parquet) | `/wynton/group/abcd/6.1/tabulated/` |
| T1w / T2w NIfTI (mproc) | `/wynton/group/abcd/6.0/imaging/derivatives/mproc/` |
| Concatenated matrices | `/wynton/group/abcd/6.1/concat/` |
| Clinical findings | `mr_y_qc__clfind.parquet` |
| WMH ASEG volumes | `mr_y_smri__vol__aseg.parquet` |
| Demographics | `abcd_p_demo.parquet` |

and at other locations to check in /abcd/

### 2.2 Study Scale

- **Subjects:** 11,868 unique participants  
- **Sessions:** ses-00A through ses-06A (7 waves, ~2 years apart)  
- **Imaging format:** NIfTI `.nii.gz`, motion-corrected, skull-stripped where applicable  
- **T1w path pattern:** `mproc/sub-{NDAR}/ses-{SES}/anat/sub-{NDAR}_ses-{SES}_run-01_T1w.nii.gz`  
- **T2w path pattern:** `mproc/sub-{NDAR}/ses-{SES}/anat/sub-{NDAR}_ses-{SES}_run-01_T2w.nii.gz`

### 2.3 Leukoaraiosis Prevalence
to check


## 3. Pipeline Architecture Overview

```
Phase 1.2  Manifest builder      →  manifest.csv (subject, paths, label)
Phase 1.3  MONAI transforms      →  augmented 3D tensor pairs (T1+T2)
Phase 2    Swin UNETR backbone   →  encoder extracts 3D feature maps
Phase 3    GAP + classifier head →  binary classification, AUPREC training
Phase 4    Grad-CAM 3D           →  heatmaps saved as NIfTI
Phase 5    Pseudo-mask training  →  voxel probability maps, DiceFocal loss
```

---

## 4. Phase 1.2 — Manifest Builder

**Goal:** Produce `manifest.csv` linking each subject-session to its T1w path, T2w path, and binary label. Scan must exist on disk for both modalities; clfind_score must be 1–4 (exclude 0).

**Script:** `Leuko_abcd/phase1_2_build_manifest.py`

```python
#!/usr/bin/env python3
"""
Phase 1.2 — Build manifest.csv
Matches clfind labels to available NIfTI files on mproc.

Output: /wynton/home/sugrue/loubard/workspace/Leuko_abcd/manifest.csv
Columns: subject, session, t1w_path, t2w_path, label, clfind_score, wmh_mm3
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
TABULATED   = Path("/wynton/group/abcd/6.1/tabulated")
MPROC       = Path("/wynton/group/abcd/6.0/imaging/derivatives/mproc")
OUT_DIR     = Path("/wynton/home/sugrue/loubard/workspace/Leuko_abcd")
OUT_CSV     = OUT_DIR / "manifest.csv"

# ── Load clinical labels ──────────────────────────────────────────────────────
log.info("Loading clfind_score ...")
clfind = pd.read_parquet(TABULATED / "mr_y_qc__clfind.parquet",
                         columns=["src_subject_id", "eventname", "clfind_score"])

# ABCD eventname → session map
EVENTNAME_TO_SESSION = {
    "baseline_year_1_arm_1":    "ses-00A",
    "1_year_follow_up_y_arm_1": "ses-01A",
    "2_year_follow_up_y_arm_1": "ses-02A",
    "3_year_follow_up_y_arm_1": "ses-03A",
    "4_year_follow_up_y_arm_1": "ses-04A",
    "5_year_follow_up_y_arm_1": "ses-05A",
    "6_year_follow_up_y_arm_1": "ses-06A",
}
clfind["session"] = clfind["eventname"].map(EVENTNAME_TO_SESSION)
clfind = clfind.dropna(subset=["session"])

# ── Load WMH volumes (supplementary) ─────────────────────────────────────────
log.info("Loading WMH volumes ...")
wmh = pd.read_parquet(TABULATED / "mr_y_smri__vol__aseg.parquet",
                      columns=["src_subject_id", "eventname",
                               "mr_y_smri__vol__aseg__wmh_sum"])
wmh["session"] = wmh["eventname"].map(EVENTNAME_TO_SESSION)
wmh = wmh.dropna(subset=["session"])
wmh = wmh.rename(columns={"mr_y_smri__vol__aseg__wmh_sum": "wmh_mm3"})

# ── Merge ─────────────────────────────────────────────────────────────────────
df = clfind.merge(wmh[["src_subject_id", "session", "wmh_mm3"]],
                  on=["src_subject_id", "session"], how="left")

# Exclude unreadable (score 0)
df = df[df["clfind_score"] >= 1].copy()

# Binary label
df["label"] = (df["clfind_score"] >= 3).astype(int)

# ── Resolve NIfTI paths ───────────────────────────────────────────────────────
def resolve_paths(row):
    subj   = row["src_subject_id"]          # e.g. NDARINV00000001
    ses    = row["session"]                  # e.g. ses-00A
    anat   = MPROC / f"sub-{subj}" / ses / "anat"
    t1     = anat / f"sub-{subj}_{ses}_run-01_T1w.nii.gz"
    t2     = anat / f"sub-{subj}_{ses}_run-01_T2w.nii.gz"
    return str(t1) if t1.exists() else None, str(t2) if t2.exists() else None

log.info("Resolving NIfTI paths (this takes ~2 min for full dataset) ...")
paths = df.apply(resolve_paths, axis=1, result_type="expand")
df["t1w_path"] = paths[0]
df["t2w_path"] = paths[1]

# Keep only rows where BOTH modalities exist
n_before = len(df)
df = df.dropna(subset=["t1w_path", "t2w_path"])
log.info(f"Rows with both T1+T2 on disk: {len(df)} / {n_before}")

# ── Summary ───────────────────────────────────────────────────────────────────
log.info("\n=== Manifest Summary ===")
log.info(f"  Total rows:    {len(df)}")
log.info(f"  Positive (1):  {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
log.info(f"  Negative (0):  {(df['label']==0).sum()}")
log.info(f"  Unique subjects: {df['src_subject_id'].nunique()}")

# ── Save ─────────────────────────────────────────────────────────────────────
df[["src_subject_id", "session", "t1w_path", "t2w_path",
    "label", "clfind_score", "wmh_mm3"]].to_csv(OUT_CSV, index=False)
log.info(f"Saved: {OUT_CSV}")
```

**Run on login node (no GPU needed):**
```bash
source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh
python phase1_2_build_manifest.py 2>&1 | tee logs/manifest_build.log
```

Expected output: ~12,000–15,000 rows, ~500–700 positive across all sessions.

---

## 5. Phase 1.3 — MONAI Transform Pipeline

**Goal:** Load T1w + T2w as a 2-channel 3D tensor; apply spatial and intensity augmentations appropriate for brain MRI; crop to 96×96×96 patches centered on the brain.

**Script:** `Leuko_abcd/phase1_3_transforms.py`

```python
"""
Phase 1.3 — MONAI transform pipeline for T1+T2 brain MRI.
Import this module from the training script.
"""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    ConcatItemsd,
    DeleteItemsd,
    ToTensord,
)

PATCH_SIZE   = (96, 96, 96)   # Swin UNETR operating resolution
TARGET_VOXEL = (1.0, 1.0, 1.0)  # 1mm isotropic


def get_train_transforms():
    return Compose([
        # ── Load ──────────────────────────────────────────────────────────────
        LoadImaged(keys=["t1w", "t2w"]),
        EnsureChannelFirstd(keys=["t1w", "t2w"]),

        # ── Standardize orientation + spacing ─────────────────────────────────
        Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
        Spacingd(keys=["t1w", "t2w"],
                 pixdim=TARGET_VOXEL,
                 mode=("bilinear", "bilinear")),

        # ── Intensity normalization (per-volume percentile) ───────────────────
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),

        # ── Crop to brain foreground, then pad to at least patch size ─────────
        CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),
        SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),

        # ── Random 96³ patch ──────────────────────────────────────────────────
        RandSpatialCropd(keys=["t1w", "t2w"],
                         roi_size=PATCH_SIZE,
                         random_size=False),

        # ── Spatial augmentation ──────────────────────────────────────────────
        RandFlipd(keys=["t1w", "t2w"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["t1w", "t2w"], prob=0.5, max_k=3),
        RandAffined(
            keys=["t1w", "t2w"],
            prob=0.3,
            rotate_range=(0.1, 0.1, 0.1),
            shear_range=(0.05, 0.05, 0.05),
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "bilinear"),
            padding_mode="border",
        ),

        # ── Intensity augmentation ────────────────────────────────────────────
        RandGaussianNoised(keys=["t1w", "t2w"], prob=0.2, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["t1w", "t2w"], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandScaleIntensityd(keys=["t1w", "t2w"], prob=0.3, factors=0.1),

        # ── Concatenate into 2-channel tensor → shape (2, 96, 96, 96) ─────────
        ConcatItemsd(keys=["t1w", "t2w"], name="image"),
        DeleteItemsd(keys=["t1w", "t2w"]),
        ToTensord(keys=["image"]),
    ])


def get_val_transforms():
    """Deterministic transforms for validation / inference."""
    return Compose([
        LoadImaged(keys=["t1w", "t2w"]),
        EnsureChannelFirstd(keys=["t1w", "t2w"]),
        Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
        Spacingd(keys=["t1w", "t2w"],
                 pixdim=TARGET_VOXEL,
                 mode=("bilinear", "bilinear")),
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),
        SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),
        RandSpatialCropd(keys=["t1w", "t2w"],
                         roi_size=PATCH_SIZE,
                         random_size=False,
                         random_center=False),   # center crop for val
        ConcatItemsd(keys=["t1w", "t2w"], name="image"),
        DeleteItemsd(keys=["t1w", "t2w"]),
        ToTensord(keys=["image"]),
    ])
```

---

## 6. Phase 2 — Swin UNETR Backbone (MONAI 1.5.2 API)

### 6.1 Architecture Decision

We use the **encoder only** during classification training. The Swin UNETR decoder is activated in Phase 5 (pseudo-mask self-training). This two-stage approach forces the encoder to learn global semantic representations before the decoder learns spatial detail.

**Critical MONAI 1.5.2 note:** `img_size` parameter was removed from `SwinUNETR.__init__()`. The model is now resolution-agnostic; dimensions are inferred at forward pass time from the input tensor shape.

```python
from monai.networks.nets import SwinUNETR

# Correct instantiation for MONAI 1.5.2
backbone = SwinUNETR(
    in_channels=2,          # T1 + T2
    out_channels=2,         # ignored during Phase 2–4; used in Phase 5
    feature_size=48,        # controls Swin window embedding dim; 48 → ~62M params
    use_checkpoint=True,    # gradient checkpointing: ~40% VRAM reduction
    spatial_dims=3,
)
```

### 6.2 GAP Classification Head

```python
import torch
import torch.nn as nn

class LeukoBinaryClassifier(nn.Module):
    """
    Swin UNETR encoder → Global Average Pool → binary classifier.
    The encoder is shared with the Phase 5 decoder.
    """

    def __init__(self, feature_size: int = 48, dropout: float = 0.3):
        super().__init__()

        # Swin UNETR backbone (encoder + decoder; decoder frozen/unused in Phase 2-4)
        self.backbone = SwinUNETR(
            in_channels=2,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3,
        )

        # The encoder's final hidden dim is feature_size * 32 = 1536 for feature_size=48
        # (Swin UNETR uses 4 stages: ×2, ×4, ×8, ×16 → final dim = feature_size * 16 * 2)
        hidden_dim = feature_size * 32   # 1536

        # GAP + MLP head
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),   # raw logit; sigmoid applied by loss
        )

    def forward(self, x):
        """
        x: (B, 2, D, H, W) tensor
        Returns: (B, 1) logit
        """
        # Extract encoder hidden states
        # SwinUNETR.swinViT returns a list of hidden states at each resolution
        hidden_states = self.backbone.swinViT(x, normalize=True)
        # hidden_states[-1]: (B, feature_size*32, D/32, H/32, W/32) = (B,1536,3,3,3)
        feat = hidden_states[-1]
        pooled = self.gap(feat)   # (B, 1536, 1, 1, 1)
        return self.head(pooled)  # (B, 1)

    def get_cam_target(self):
        """Returns the final encoder block for Grad-CAM hook registration."""
        return self.backbone.swinViT.layers4[-1]
```

---

## 7. Phase 3 — Training Loop (AUPREC / APLoss)

### 7.1 SLURM Job Script

Save as `Leuko_abcd/jobs/train_phase3.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=leuko_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/train_%j.out
#SBATCH --error=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/train_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh

# ── Run ───────────────────────────────────────────────────────────────────────
cd /wynton/home/sugrue/loubard/workspace/Leuko_abcd
python phase3_train.py \
    --manifest manifest.csv \
    --out_dir runs/phase3_$(date +%Y%m%d_%H%M%S) \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --feature_size 48 \
    --seed 42
```

Submit with:
```bash
mkdir -p /wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs
sbatch jobs/train_phase3.sh
```

### 7.2 Training Script

**Script:** `Leuko_abcd/phase3_train.py`

```python
#!/usr/bin/env python3
"""
Phase 3 — Binary leukoaraiosis classifier training.
Optimizer: SOAP (LibAUC)
Loss:      APLoss (surrogate AUPREC, LibAUC)
Metric:    AUPREC (primary) + AUROC
"""

import argparse
import random
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from monai.data import CacheDataset
from libauc.losses import APLoss
from libauc.optimizers import SOAP
from sklearn.metrics import average_precision_score, roc_auc_score

# Import local modules
from phase1_3_transforms import get_train_transforms, get_val_transforms
from phase2_model import LeukoBinaryClassifier


# ── Dataset ───────────────────────────────────────────────────────────────────

class LeukoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {
            "t1w": row["t1w_path"],
            "t2w": row["t2w_path"],
        }
        data = self.transform(data)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return data["image"], label


# ── Training utilities ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray):
    """For SOAP margin: estimate prior positive probability."""
    return float(labels.mean())


# ── Main ─────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load manifest ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    print(f"Manifest: {len(df)} rows, {df['label'].sum()} positive ({df['label'].mean()*100:.1f}%)")

    # Stratified group k-fold (grouped by subject — no data leakage across sessions)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    groups = df["src_subject_id"].values
    labels = df["label"].values

    # Use fold 0 for initial training
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        if fold_idx != 0:
            continue
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]
        break

    print(f"Train: {len(train_df)} rows ({train_df['label'].sum()} pos)")
    print(f"Val:   {len(val_df)} rows ({val_df['label'].sum()} pos)")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = LeukoDataset(train_df, get_train_transforms())
    val_ds   = LeukoDataset(val_df,   get_val_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LeukoBinaryClassifier(feature_size=args.feature_size).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    # ── Loss + Optimizer ──────────────────────────────────────────────────────
    pos_prior = compute_class_weights(train_df["label"].values)
    print(f"Positive prior (p): {pos_prior:.4f}")

    # APLoss: differentiable surrogate for Average Precision
    loss_fn = APLoss(
        pos_len=int(train_df["label"].sum()),
        num_labels=1,
        margin=1.0,
        gamma=0.1,
    )

    # SOAP: Stochastic Optimization for AUC-Precision
    optimizer = SOAP(
        model.parameters(),
        loss_fn=loss_fn,
        lr=args.lr,
        epoch_decay=2e-4,
        weight_decay=1e-5,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    best_auprec = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_logits, train_labels = [], []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)   # (B,)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_logits.extend(logits.detach().cpu().float().numpy())
            train_labels.extend(labels.cpu().numpy())

        optimizer.update_regularizer(decay_factor=10)

        # Validate
        model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images).squeeze(1)
                val_logits.extend(logits.cpu().float().numpy())
                val_labels.extend(labels.numpy())

        # Metrics
        train_probs = torch.sigmoid(torch.tensor(train_logits)).numpy()
        val_probs   = torch.sigmoid(torch.tensor(val_logits)).numpy()

        train_auprec = average_precision_score(train_labels, train_probs)
        val_auprec   = average_precision_score(val_labels,   val_probs)
        val_auroc    = roc_auc_score(val_labels, val_probs)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train AUPREC={train_auprec:.4f} | "
              f"Val AUPREC={val_auprec:.4f} | "
              f"Val AUROC={val_auroc:.4f} | "
              f"Time={elapsed:.0f}s")

        row = {
            "epoch": epoch,
            "train_auprec": train_auprec,
            "val_auprec": val_auprec,
            "val_auroc": val_auroc,
        }
        history.append(row)

        # Checkpoint
        if val_auprec > best_auprec:
            best_auprec = val_auprec
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprec": val_auprec,
                "val_auroc": val_auroc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  → New best AUPREC={best_auprec:.4f}, saved to {ckpt_path}")

    # Save history
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    print(f"\nTraining complete. Best Val AUPREC: {best_auprec:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",     default="manifest.csv")
    parser.add_argument("--out_dir",      default="runs/phase3")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--feature_size", type=int,   default=48)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()
    train(args)
```

### 7.3 Expected VRAM Budget (A100 80GB)

| Component | VRAM |
|---|---|
| Model weights (62M params, fp32) | ~240 MB |
| Model weights (bf16 training) | ~120 MB |
| Activations, batch=4, 96³ input | ~18–24 GB |
| Gradient checkpointing saving | ~40% reduction |
| **Total estimated** | **~20–28 GB** |

Gradient checkpointing (`use_checkpoint=True`) is essential; recomputes activations during backward pass at the cost of ~35% slower training.

---

## 8. Phase 4 — Grad-CAM Heatmap Generation

### 8.1 Concept

Grad-CAM backpropagates the classification gradient to the final Swin Transformer block's output feature map. Channels are globally average-pooled to obtain weights, then the weighted sum of the feature map produces a coarse 3D attention map that is resized (via trilinear interpolation) to the full input resolution.

### 8.2 Script

**Script:** `Leuko_abcd/phase4_gradcam.py`

```python
#!/usr/bin/env python3
"""
Phase 4 — Grad-CAM 3D heatmap generation.

For each subject in val/test set:
  1. Forward pass → classification logit
  2. Backward pass on positive class
  3. Compute weighted activation map at final encoder stage
  4. Upsample to input resolution (96³) then to native MRI resolution
  5. Save as NIfTI .nii.gz with same affine as T1w

Aggregate map: mean Grad-CAM across all true positives (TP subjects).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Orientation, Spacing, \
    ScaleIntensityRangePercentiles, NormalizeIntensity, CropForeground, SpatialPad, \
    RandSpatialCrop, ConcatItems
from sklearn.metrics import average_precision_score, roc_auc_score

from phase2_model import LeukoBinaryClassifier


PATCH_SIZE = (96, 96, 96)


class GradCAM3D:
    """
    Register forward/backward hooks on target_layer.
    Call .compute() to get the Grad-CAM map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: (B, C, D, H, W) or (B, tokens, C) for Swin
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        """
        Returns Grad-CAM map as numpy array, shape (D, H, W), values in [0, 1].
        """
        self.model.zero_grad()
        logit = self.model(input_tensor)         # (1, 1)
        score = logit[0, class_idx]
        score.backward()

        grads = self.gradients     # (1, C, D', H', W') or (1, tokens, C)
        acts  = self.activations   # same shape

        # Swin Transformer outputs tokens: reshape to spatial
        if grads.dim() == 3:
            # (B, tokens, C) → need to infer spatial dims
            B, T, C = grads.shape
            D = H = W = round(T ** (1/3))
            grads = grads.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            acts  = acts.view( B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Global average pooling over spatial dims → weights per channel
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)   # (1, C, 1, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)      # (1, 1, D', H', W')
        cam = F.relu(cam)

        # Upsample to patch size
        cam_up = F.interpolate(cam,
                               size=PATCH_SIZE,
                               mode="trilinear",
                               align_corners=False)

        cam_np = cam_up.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np


def load_single_volume(t1w_path: str, t2w_path: str) -> torch.Tensor:
    """Load and preprocess a single T1+T2 pair → (1, 2, 96, 96, 96) tensor."""
    from phase1_3_transforms import get_val_transforms
    transform = get_val_transforms()
    data = transform({"t1w": t1w_path, "t2w": t2w_path})
    return data["image"].unsqueeze(0)   # add batch dim


def run_gradcam(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LeukoBinaryClassifier(feature_size=args.feature_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(Val AUPREC={ckpt['val_auprec']:.4f})")

    # Register Grad-CAM on final encoder stage
    target_layer = model.get_cam_target()
    gradcam = GradCAM3D(model, target_layer)

    # Load manifest (use validation set or full set for inference)
    df = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs, all_labels = [], []
    aggregate_cam = None
    n_aggregate   = 0

    for _, row in df.iterrows():
        subj, ses = row["src_subject_id"], row["session"]
        label = int(row["label"])

        try:
            img = load_single_volume(row["t1w_path"], row["t2w_path"]).to(device)
        except Exception as e:
            print(f"SKIP {subj} {ses}: {e}")
            continue

        with torch.enable_grad():
            cam = gradcam.compute(img, class_idx=0)

        with torch.no_grad():
            logit = model(img)
            prob  = torch.sigmoid(logit).item()

        all_probs.append(prob)
        all_labels.append(label)

        # Save individual heatmap as NIfTI
        ref_img = nib.load(row["t1w_path"])
        # cam is (96,96,96); we embed it in native space at 1mm iso affine
        # (full-resolution upsampling would require storing the crop metadata)
        cam_nii = nib.Nifti1Image(cam.astype(np.float32), affine=ref_img.affine)
        nii_path = out_dir / f"{subj}_{ses}_gradcam.nii.gz"
        nib.save(cam_nii, nii_path)

        # Aggregate (only true positives)
        if label == 1:
            if aggregate_cam is None:
                aggregate_cam = cam.copy()
            else:
                aggregate_cam += cam
            n_aggregate += 1

    # Global metrics
    if len(all_labels) > 0:
        auprec = average_precision_score(all_labels, all_probs)
        auroc  = roc_auc_score(all_labels,  all_probs)
        print(f"\nInference complete: N={len(all_labels)}")
        print(f"  AUPREC: {auprec:.4f}")
        print(f"  AUROC:  {auroc:.4f}")

    # Save aggregate heatmap
    if aggregate_cam is not None and n_aggregate > 0:
        agg = aggregate_cam / n_aggregate
        agg = (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)
        # Use identity affine for group-level map (registered space)
        agg_nii = nib.Nifti1Image(agg.astype(np.float32), affine=np.eye(4))
        agg_path = out_dir / "aggregate_gradcam_positives.nii.gz"
        nib.save(agg_nii, agg_path)
        print(f"\nAggregate Grad-CAM ({n_aggregate} TP subjects): {agg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--manifest",     default="manifest.csv")
    parser.add_argument("--out_dir",      default="heatmaps/")
    parser.add_argument("--feature_size", type=int, default=48)
    args = parser.parse_args()
    run_gradcam(args)
```

### 8.3 SLURM Job for Heatmap Generation

```bash
#!/bin/bash
#SBATCH --job-name=leuko_gradcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/wynton/home/sugrue/loubard/workspace/Leuko_abcd/logs/gradcam_%j.out

source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh
cd /wynton/home/sugrue/loubard/workspace/Leuko_abcd

python phase4_gradcam.py \
    --checkpoint runs/phase3_*/best_model.pt \
    --manifest manifest.csv \
    --out_dir heatmaps/gradcam_$(date +%Y%m%d)
```

### 8.4 Heatmap Visualization

```python
"""
Visualize aggregate Grad-CAM heatmap overlaid on MNI T1 template.
Save publication-quality figure.
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

# Load aggregate heatmap
cam_nii = nib.load("heatmaps/gradcam_YYYYMMDD/aggregate_gradcam_positives.nii.gz")

# Plot on glass brain (sagittal/coronal/axial projections)
fig = plt.figure(figsize=(14, 4))
display = plotting.plot_glass_brain(
    cam_nii,
    colorbar=True,
    cmap="hot",
    symmetric_cbar=False,
    title="Mean Grad-CAM: Leukoaraiosis (clfind≥3, N=444)",
    figure=fig,
)
plt.savefig("figures/aggregate_gradcam_glass_brain.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/aggregate_gradcam_glass_brain.png", dpi=300, bbox_inches="tight")

# Axial slice mosaic
display2 = plotting.plot_stat_map(
    cam_nii,
    threshold=0.3,
    cmap="hot",
    colorbar=True,
    display_mode="z",
    cut_coords=8,
    title="Grad-CAM attention — axial slices (threshold 0.3)",
)
plt.savefig("figures/aggregate_gradcam_axial.pdf", dpi=300, bbox_inches="tight")

print("Figures saved.")
```

---

## 9. Phase 5 — Pseudo-Mask Self-Training

### 9.1 Concept

After Phase 3–4 confirm the classifier works (Val AUPREC > 0.6), we use Grad-CAM heatmaps as **noisy supervision masks** to fine-tune the full Swin UNETR decoder for voxel-level segmentation. This is a self-supervised loop:

```
Epoch k: Train decoder with pseudo-masks from Grad-CAM epoch k-1
Epoch k+1: Generate refined Grad-CAM from updated model → new pseudo-masks
```

### 9.2 Pseudo-Mask Preparation

```python
"""
Convert Grad-CAM heatmaps → binary pseudo-masks via Otsu thresholding.
Only for predicted-positive subjects (prob > 0.5).
"""

import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
from pathlib import Path

heatmap_dir = Path("heatmaps/gradcam_YYYYMMDD")
mask_dir    = Path("heatmaps/pseudo_masks")
mask_dir.mkdir(parents=True, exist_ok=True)

for cam_path in heatmap_dir.glob("*_gradcam.nii.gz"):
    nii = nib.load(cam_path)
    cam = nii.get_fdata()

    # Otsu threshold on non-zero voxels
    nonzero = cam[cam > 0.01]
    if len(nonzero) < 100:
        continue
    thresh = threshold_otsu(nonzero)
    mask = (cam >= thresh).astype(np.uint8)

    out_path = mask_dir / cam_path.name.replace("_gradcam.nii.gz", "_pseudo_mask.nii.gz")
    nib.save(nib.Nifti1Image(mask, nii.affine), out_path)
```

### 9.3 Full Decoder Fine-tuning

```python
"""
Phase 5 — Fine-tune full Swin UNETR (encoder + decoder) with pseudo-masks.
Loss: DiceFocalLoss (Dice for global shape, Focal for hard boundary voxels).
"""

from monai.losses import DiceFocalLoss
from monai.networks.nets import SwinUNETR
import torch

# Load encoder weights from Phase 3 checkpoint
backbone = SwinUNETR(
    in_channels=2,
    out_channels=2,         # background + leukoaraiosis
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=3,
)
ckpt = torch.load("runs/phase3/best_model.pt", map_location="cuda")
# Transfer only encoder weights
encoder_state = {k.replace("backbone.", ""): v
                 for k, v in ckpt["model_state_dict"].items()
                 if k.startswith("backbone.")}
backbone.load_state_dict(encoder_state, strict=False)

# Loss: Dice + Focal
loss_fn = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    gamma=2.0,
    lambda_dice=1.0,
    lambda_focal=1.0,
)

# Fine-tune full model with lower LR
optimizer = torch.optim.AdamW(backbone.parameters(), lr=5e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

---

## 10. Longitudinal Extension (All Sessions)

After the baseline model is validated, extend to all 7 sessions to study WMH progression:

```python
# Use all sessions in manifest.csv (not just ses-00A)
# Each subject-session is a separate training example
# Stratify by SUBJECT (not session) in cross-validation splits
# This prevents session-level leakage while allowing longitudinal variation

# Mixed-effects modeling of heatmap centroid displacement over time:
# For each TP subject with multiple sessions, compute centroid of Grad-CAM per session
# Fit linear mixed model: centroid_x ~ session + (session | subject_id)
```

---

## 11. Directory Structure

```
/wynton/home/sugrue/loubard/workspace/Leuko_abcd/
├── activate_env.sh              # source before any script
├── install_env.sh               # re-runnable package installer
├── implementation_plan.md       # this document
├── ABCD_data_overview.md        # 239 tables, full data reference
├── project_state.md             # completed work log
│
├── phase1_2_build_manifest.py   # generates manifest.csv
├── phase1_3_transforms.py       # MONAI transform pipeline
├── phase2_model.py              # LeukoBinaryClassifier (Swin UNETR + GAP)
├── phase3_train.py              # APLoss + SOAP training loop
├── phase4_gradcam.py            # Grad-CAM heatmap generation
│
├── jobs/
│   ├── train_phase3.sh          # SLURM training job
│   └── gradcam_phase4.sh        # SLURM heatmap job
│
├── manifest.csv                 # (generated) subject × session × paths × label
├── logs/                        # SLURM stdout/stderr
├── runs/                        # training checkpoints + history
│   └── phase3_YYYYMMDD_HHMMSS/
│       ├── best_model.pt
│       └── training_history.csv
├── heatmaps/
│   ├── gradcam_YYYYMMDD/        # per-subject NIfTI heatmaps
│   ├── pseudo_masks/            # binarized Grad-CAM for Phase 5
│   └── aggregate_gradcam_positives.nii.gz
└── figures/                     # publication-quality plots
    ├── aggregate_gradcam_glass_brain.pdf
    └── aggregate_gradcam_axial.pdf
```

---

## 12. Execution Sequence

```bash
# Step 1: Activate environment (every session)
source /wynton/home/sugrue/loubard/workspace/Leuko_abcd/activate_env.sh

# Step 2: Build manifest (login node, ~5 minutes)
cd /wynton/home/sugrue/loubard/workspace/Leuko_abcd
python phase1_2_build_manifest.py

# Step 3: Verify transforms on one sample (login node, CPU only)
python -c "
from phase1_3_transforms import get_val_transforms
import pandas as pd
df = pd.read_csv('manifest.csv').head(1)
t = get_val_transforms()
d = t({'t1w': df.t1w_path[0], 't2w': df.t2w_path[0]})
print('Image shape:', d['image'].shape)
"

# Step 4: Submit training job
mkdir -p logs
sbatch jobs/train_phase3.sh

# Monitor
squeue -u $USER
tail -f logs/train_*.out

# Step 5: Generate heatmaps (after training converges)
# Edit jobs/gradcam_phase4.sh with correct checkpoint path, then:
sbatch jobs/gradcam_phase4.sh

# Step 6: Visualize
python -c "
import nibabel as nib
from nilearn import plotting
nii = nib.load('heatmaps/gradcam_YYYYMMDD/aggregate_gradcam_positives.nii.gz')
plotting.plot_glass_brain(nii, colorbar=True, cmap='hot')
plotting.show()
"
```

---

## 13. Key Hyperparameters and Justifications

| Parameter | Value | Rationale |
|---|---|---|
| `feature_size` | 48 | Standard Swin UNETR config; ~62M params; fits A100 with gradient checkpointing |
| `patch_size` | 96³ | Captures periventricular and subcortical WM in single patch at 1mm iso |
| `batch_size` | 4 | VRAM constraint with gradient checkpointing; increase to 8 on A100 80GB if available |
| `lr` | 1e-4 | SOAP default; higher LR causes instability with APLoss |
| APLoss `margin` | 1.0 | Squared hinge; standard for binary AUPREC optimization |
| APLoss `gamma` | 0.1 | Moving average window for surrogate gradient |
| `dropout` | 0.3 | Regularization given 3.8% positive rate and small effective N |
| `clfind_score` threshold | >=3 | Radiologist referral; strongest clinical label available |
| Cross-validation | StratifiedGroupKFold(k=5) | Group=subject to prevent session leakage |
| Optimizer | SOAP | Designed for AUPREC-surrogate optimization; outperforms SGD+APLoss |
