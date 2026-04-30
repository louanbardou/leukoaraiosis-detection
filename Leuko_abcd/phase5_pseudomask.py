#!/usr/bin/env python3
"""
phase5_pseudomask.py
--------------------
Pseudo-mask self-training: converts Grad-CAM heatmaps into binary segmentation
masks and uses them to fine-tune the full Swin UNETR decoder.

This is an optional refinement step that should only be run if the Phase 3
classifier achieved Val AUPREC > 0.60. Below that threshold, the Grad-CAM
heatmaps are too noisy to serve as reliable training targets.

Two-step process
----------------
Step A: Pseudo-mask preparation
    For each subject whose Grad-CAM heatmap was saved by Phase 4, apply Otsu's
    method to threshold the continuous heatmap into a binary mask. Otsu's method
    chooses the threshold that maximises the inter-class variance between the
    "background" and "lesion" distributions of heatmap intensities. This avoids
    the need to manually tune a threshold.

    Only heatmaps with at least 100 non-zero voxels are processed. Heatmaps
    with fewer active voxels indicate that the classifier did not find any clear
    evidence of pathology and are not reliable enough to use as masks.

Step B: Decoder fine-tuning
    The full Swin UNETR (encoder + decoder) is loaded. The encoder weights are
    transferred from the Phase 3 classification checkpoint. The decoder weights
    are initialised randomly (they were never trained in Phase 3).

    The model is then fine-tuned on the pseudo-masks using DiceFocalLoss, which
    combines two complementary objectives:

    Dice loss: measures the volumetric overlap between the predicted mask and
    the pseudo-mask. It is invariant to the absolute size of the lesion and
    ensures that the overall shape of the segmentation is correct.

    Focal loss: a modified cross-entropy that down-weights easy (correctly
    classified) voxels by a factor of (1-p)^gamma. This concentrates gradient
    signal on the hard voxels at the lesion boundary, where the model is most
    uncertain. With gamma=2, the loss for voxels with p=0.9 is 100x smaller
    than the loss for voxels with p=0.5.

    A lower learning rate (5e-5 vs 1e-4 in Phase 3) is used for fine-tuning to
    avoid disrupting the encoder representations that were learned during Phase 3.
    Cosine annealing smoothly reduces the learning rate to near-zero by the end
    of training.

Why self-training works
-----------------------
The Phase 3 classifier already locates leukoaraiosis accurately (that is what
a high AUPREC score means). The Grad-CAM maps coarsely outline those regions.
By treating these outlines as ground truth and training a segmentation model on
them, the model learns to produce precise, voxel-level probability maps. This
is a form of label propagation: weak image-level labels propagate to approximate
voxel-level labels through the trained classifier.

Usage
-----
    # Prepare pseudo-masks only
    python phase5_pseudomask.py prepare \\
        --heatmap_dir heatmaps/gradcam_YYYYMMDD \\
        --mask_dir    heatmaps/pseudo_masks \\
        --manifest    manifest.csv

    # Fine-tune decoder only (masks must already exist)
    python phase5_pseudomask.py train \\
        --manifest    manifest.csv \\
        --mask_dir    heatmaps/pseudo_masks \\
        --checkpoint  runs/phase3/best_model.pt \\
        --out_dir     runs/phase5

    # Run both steps sequentially
    python phase5_pseudomask.py all \\
        --heatmap_dir heatmaps/gradcam_YYYYMMDD \\
        --mask_dir    heatmaps/pseudo_masks \\
        --manifest    manifest.csv \\
        --checkpoint  runs/phase3/best_model.pt \\
        --out_dir     runs/phase5
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceFocalLoss
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    ConcatItemsd,
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset

PATCH_SIZE   = (96, 96, 96)
TARGET_VOXEL = (1.0, 1.0, 1.0)


# -------------------------------------------------------------------------
# Step A: Pseudo-mask preparation
# -------------------------------------------------------------------------

def prepare_pseudo_masks(
    heatmap_dir: str,
    mask_dir: str,
    manifest: str,
    min_nonzero_voxels: int = 100,
) -> None:
    """
    Convert Grad-CAM heatmaps to binary pseudo-masks using Otsu thresholding.

    For each subject-session pair in the manifest, look for the corresponding
    Grad-CAM NIfTI file in heatmap_dir. If found, apply Otsu's threshold on
    the non-zero voxels and save the binary mask.

    Parameters
    ----------
    heatmap_dir : str
        Directory containing *_gradcam.nii.gz files from Phase 4.
    mask_dir : str
        Output directory for the binary pseudo-masks.
    manifest : str
        Path to manifest.csv (used to iterate over all subject-session pairs).
    min_nonzero_voxels : int
        Minimum number of non-zero voxels required in a heatmap for it to be
        converted. Heatmaps below this threshold are skipped.
    """
    heatmap_dir = Path(heatmap_dir)
    mask_dir    = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    df    = pd.read_csv(manifest)
    saved = 0

    for _, row in df.iterrows():
        subj     = row["subject_id"]
        ses      = row["session"]
        cam_path = heatmap_dir / f"{subj}_{ses}_gradcam.nii.gz"

        if not cam_path.exists():
            continue

        nii = nib.load(cam_path)
        cam = nii.get_fdata().astype(np.float32)

        nonzero_voxels = cam[cam > 0.01]

        if len(nonzero_voxels) < min_nonzero_voxels:
            print(
                f"Skipping {subj} {ses}: only {len(nonzero_voxels)} non-zero voxels "
                f"(minimum required: {min_nonzero_voxels})"
            )
            continue

        # Otsu's method finds the threshold that best separates background from
        # foreground by maximising inter-class variance. We compute it on the
        # non-zero voxels only to avoid the large background mass biasing the
        # threshold towards zero.
        threshold = threshold_otsu(nonzero_voxels)
        mask      = (cam >= threshold).astype(np.uint8)

        out_path = mask_dir / f"{subj}_{ses}_pseudo_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine=nii.affine), out_path)
        saved += 1

    print(f"Saved {saved} pseudo-masks to {mask_dir}")


# -------------------------------------------------------------------------
# Step B: Decoder fine-tuning
# -------------------------------------------------------------------------

def get_seg_transforms() -> Compose:
    """
    Transform pipeline for segmentation training.

    Identical to the classification training transforms but includes the
    pseudo-mask as a third key ("mask"). The mask is loaded and resampled with
    nearest-neighbour interpolation (mode="nearest") to preserve binary values.
    """
    return Compose([
        LoadImaged(keys=["t1w", "t2w", "mask"]),
        EnsureChannelFirstd(keys=["t1w", "t2w", "mask"]),
        Orientationd(keys=["t1w", "t2w", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["t1w", "t2w", "mask"],
            pixdim=TARGET_VOXEL,
            mode=("bilinear", "bilinear", "nearest"),  # nearest for binary mask
        ),
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1, upper=99,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),
        # Crop using the T1w foreground; apply the same bounding box to T2 and mask.
        CropForegroundd(keys=["t1w", "t2w", "mask"], source_key="t1w"),
        SpatialPadd(keys=["t1w", "t2w", "mask"], spatial_size=PATCH_SIZE),
        RandSpatialCropd(
            keys=["t1w", "t2w", "mask"],
            roi_size=PATCH_SIZE,
            random_size=False,
        ),
        RandFlipd(keys=["t1w", "t2w", "mask"], prob=0.5, spatial_axis=0),
        RandRotate90d(keys=["t1w", "t2w", "mask"], prob=0.5, max_k=3),
        ConcatItemsd(keys=["t1w", "t2w"], name="image"),
        DeleteItemsd(keys=["t1w", "t2w"]),
        ToTensord(keys=["image", "mask"]),
    ])


class SegDataset(Dataset):
    """
    Dataset for segmentation fine-tuning.

    Each record is a dict with t1w_path, t2w_path, mask_path.
    Returns (image, mask) pairs where:
        image : (2, 96, 96, 96) tensor
        mask  : (1, 96, 96, 96) float tensor with values 0.0 or 1.0
    """

    def __init__(self, records: list, transform: Compose):
        self.records   = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r    = self.records[idx]
        data = self.transform({
            "t1w":  r["t1w_path"],
            "t2w":  r["t2w_path"],
            "mask": r["mask_path"],
        })
        return data["image"], data["mask"].float()


def finetune_decoder(
    manifest: str,
    mask_dir: str,
    checkpoint: str,
    out_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    feature_size: int,
) -> None:
    """
    Fine-tune the full Swin UNETR (encoder + decoder) on pseudo-masks.

    The encoder is initialised from the Phase 3 classification checkpoint.
    The decoder starts from random initialisation.

    Parameters
    ----------
    manifest : str
        Path to manifest.csv.
    mask_dir : str
        Directory containing pseudo-mask NIfTI files from Step A.
    checkpoint : str
        Path to the Phase 3 best_model.pt checkpoint.
    out_dir : str
        Output directory for the Phase 5 best segmentation checkpoint.
    epochs : int
        Number of training epochs.
    lr : float
        Initial learning rate. Lower than Phase 3 to avoid disrupting the
        pre-trained encoder representations.
    batch_size : int
        Number of volumes per gradient step.
    feature_size : int
        Must match the value used in Phase 3.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(mask_dir)

    # Build the record list: only subjects for which a pseudo-mask exists
    df      = pd.read_csv(manifest)
    records = []
    for _, row in df.iterrows():
        mask_path = mask_dir / f"{row['subject_id']}_{row['session']}_pseudo_mask.nii.gz"
        if mask_path.exists():
            records.append({
                "t1w_path":  row["t1w_path"],
                "t2w_path":  row["t2w_path"],
                "mask_path": str(mask_path),
            })

    print(f"Subjects with pseudo-mask available: {len(records)}")
    if len(records) == 0:
        raise RuntimeError(
            "No pseudo-masks found in the mask_dir. "
            "Run 'python phase5_pseudomask.py prepare' first."
        )

    loader = DataLoader(
        SegDataset(records, get_seg_transforms()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Instantiate the full Swin UNETR (with decoder)
    model = SwinUNETR(
        in_channels=2,
        out_channels=2,    # background and leukoaraiosis channels
        feature_size=feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    # Transfer encoder weights from the Phase 3 checkpoint.
    # The Phase 3 checkpoint stores weights under "backbone.*" keys (the backbone
    # attribute of LeukoBinaryClassifier). We strip the "backbone." prefix to
    # match the top-level keys expected by SwinUNETR.
    ckpt = torch.load(checkpoint, map_location=device)
    encoder_state = {
        k.replace("backbone.", ""): v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(
        f"Encoder weights transferred. "
        f"Missing keys (decoder, expected): {len(missing)}. "
        f"Unexpected keys: {len(unexpected)}."
    )

    # DiceFocalLoss: Dice handles global overlap, Focal handles boundary voxels.
    # to_onehot_y=True converts the single-channel integer mask to a 2-channel
    # one-hot tensor before computing the loss.
    loss_fn = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        gamma=2.0,        # focusing parameter: down-weights easy voxels
        lambda_dice=1.0,
        lambda_focal=1.0,
    )

    # AdamW with cosine annealing. Lower LR than Phase 3 to preserve the
    # encoder representations learned during classification training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)            # (B, 2, D, H, W)
            loss  = loss_fn(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        avg_loss = running_loss / len(loader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:3d}/{epochs}  DiceFocal={avg_loss:.4f}  {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = out_dir / "best_segmodel.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  New best loss={best_loss:.4f}, checkpoint saved.")

    print(f"\nPhase 5 complete. Best DiceFocal: {best_loss:.4f}")
    print(f"Segmentation checkpoint: {out_dir / 'best_segmodel.pt'}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5: prepare binary pseudo-masks from Grad-CAM heatmaps "
            "and/or fine-tune the Swin UNETR decoder on those masks."
        )
    )
    parser.add_argument(
        "command",
        choices=["prepare", "train", "all"],
        help=(
            "prepare: convert heatmaps to masks only. "
            "train: fine-tune decoder only (masks must already exist). "
            "all: run prepare then train."
        ),
    )
    parser.add_argument("--heatmap_dir",  default="heatmaps/gradcam",
                        help="Directory of *_gradcam.nii.gz files from Phase 4.")
    parser.add_argument("--mask_dir",     default="heatmaps/pseudo_masks",
                        help="Output directory for binary pseudo-masks.")
    parser.add_argument("--manifest",     default="manifest.csv")
    parser.add_argument("--checkpoint",   default="runs/phase3/best_model.pt",
                        help="Phase 3 classification checkpoint.")
    parser.add_argument("--out_dir",      default="runs/phase5")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--feature_size", type=int,   default=48)
    args = parser.parse_args()

    if args.command in ("prepare", "all"):
        prepare_pseudo_masks(args.heatmap_dir, args.mask_dir, args.manifest)

    if args.command in ("train", "all"):
        finetune_decoder(
            manifest=args.manifest,
            mask_dir=args.mask_dir,
            checkpoint=args.checkpoint,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            feature_size=args.feature_size,
        )


if __name__ == "__main__":
    main()
