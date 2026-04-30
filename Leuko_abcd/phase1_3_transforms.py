"""
phase1_3_transforms.py
----------------------
MONAI transform pipelines for loading and preprocessing T1w + T2w brain MRI pairs.

This module defines two transform pipelines:
    get_train_transforms()   -- used during training; includes random augmentations
    get_val_transforms()     -- used during validation and inference; deterministic

Both pipelines produce a single output tensor with shape (2, 96, 96, 96):
    channel 0 : T1-weighted volume
    channel 1 : T2-weighted volume

Design rationale
----------------
The pipeline performs several non-trivial preprocessing steps, each with a specific
scientific justification:

1. Orientation (RAS)
   ABCD data is acquired at 21 sites with different scanners and acquisition
   protocols. Enforcing a canonical Right-Anterior-Superior orientation ensures
   that spatial relationships (left/right, front/back, top/bottom) are consistent
   across all subjects in a training batch.

2. Spacing resampling to 1 mm isotropic
   The native resolution of ABCD T1w scans is approximately 1 mm isotropic,
   but T2w scans and data from some sites differ slightly. Resampling both
   modalities to a common 1 mm grid ensures that the spatial alignment between
   the T1 and T2 channels is voxel-exact after ConcatItems.

3. Intensity normalisation (two-step)
   Step A: ScaleIntensityRangePercentiles clips values outside the [1st, 99th]
   percentile range to [0, 1]. This removes scanner-specific outlier intensities
   (e.g. bright eyeballs, RF coil hotspots) that would otherwise dominate
   gradient updates.
   Step B: NormalizeIntensity applies per-channel Z-score normalisation. This
   zero-centres and scales each modality independently, which is important
   because T1 and T2 intensities live on completely different scales.

4. CropForeground + SpatialPad
   CropForeground removes the air voxels surrounding the skull (based on the T1w
   foreground mask). This reduces the patch size needed to cover the brain and
   concentrates model capacity on brain tissue. SpatialPad then guarantees the
   cropped volume is at least as large as the patch size; it adds zero-padding
   at the borders if the brain is smaller than 96 voxels in any dimension
   (uncommon but possible for very small or incomplete scans).

5. Patch extraction at 96x96x96
   The Swin UNETR operates at this fixed spatial resolution. Training on full
   volumes (typically ~180x220x180 voxels) would exceed GPU memory.
   During training: a random patch is drawn (random_size=False means the patch
   is always exactly 96^3; random_center=True means the centre is sampled
   uniformly from valid positions).
   During validation: the patch is always taken from the centre of the volume
   (random_center=False) for reproducibility.

6. Spatial augmentations (training only)
   Random flips, 90-degree rotations, and affine deformations artificially
   increase the effective training set size and improve generalisation across
   subject-to-subject anatomical variability.

7. Intensity augmentations (training only)
   Gaussian noise, Gaussian blur, and random intensity scaling simulate scanner
   noise and contrast differences. These reduce the model's tendency to overfit
   to scanner-specific intensity distributions.

Import
------
    from phase1_3_transforms import get_train_transforms, get_val_transforms

    # Example usage
    transform = get_val_transforms()
    sample = transform({"t1w": "/path/T1w.nii.gz", "t2w": "/path/T2w.nii.gz"})
    image = sample["image"]   # torch.Tensor, shape (2, 96, 96, 96)
"""

from monai.transforms import (
    ConcatItemsd,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Compose,
)

PATCH_SIZE   = (96, 96, 96)      # spatial dimensions fed to Swin UNETR
TARGET_VOXEL = (1.0, 1.0, 1.0)  # 1 mm isotropic resampling target


def get_train_transforms() -> Compose:
    """
    Build the training transform pipeline.

    Returns a MONAI Compose object. Call it with a dict:
        {"t1w": "/path/T1w.nii.gz", "t2w": "/path/T2w.nii.gz"}
    It returns a dict with key "image" containing a (2, 96, 96, 96) tensor.

    The augmentations (random flip, rotate, affine, noise, blur, scale) are
    applied stochastically. Probabilities are set conservatively to avoid
    producing anatomically implausible images.
    """
    return Compose([
        # Load NIfTI files from disk. MONAI LoadImaged handles .nii.gz automatically.
        LoadImaged(keys=["t1w", "t2w"]),

        # MONAI loads volumes as (D, H, W) by default. Add a channel dimension
        # so subsequent transforms see shape (1, D, H, W).
        EnsureChannelFirstd(keys=["t1w", "t2w"]),

        # Enforce Right-Anterior-Superior voxel ordering across all sites.
        Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),

        # Resample to 1 mm isotropic so T1 and T2 are on the same spatial grid.
        # Bilinear interpolation is appropriate for continuous intensity volumes.
        Spacingd(keys=["t1w", "t2w"], pixdim=TARGET_VOXEL, mode=("bilinear", "bilinear")),

        # Clip intensity outliers (top and bottom 1%) to [0, 1].
        # This is done before Z-score normalisation to avoid outliers skewing
        # the mean and standard deviation.
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # Z-score normalisation per channel, computed only on non-zero (brain) voxels.
        # channel_wise=True normalises T1 and T2 independently because they have
        # different tissue contrast profiles.
        NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),

        # Remove air background based on the T1w foreground mask. The T2w volume
        # is cropped with the same bounding box (source_key="t1w") to keep alignment.
        CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),

        # Zero-pad if the cropped brain is smaller than 96^3 in any dimension.
        SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),

        # Draw a random 96^3 patch from within the brain volume.
        RandSpatialCropd(keys=["t1w", "t2w"], roi_size=PATCH_SIZE, random_size=False),

        # Randomly flip along the left-right axis (axial symmetry of the brain).
        RandFlipd(keys=["t1w", "t2w"], prob=0.5, spatial_axis=0),

        # Randomly rotate by 0, 90, 180, or 270 degrees along any axis.
        RandRotate90d(keys=["t1w", "t2w"], prob=0.5, max_k=3),

        # Random affine deformation: small rotations, shears, translations, scales.
        # Probability 0.3 keeps augmentation moderate to avoid extreme distortions.
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

        # Add low-amplitude Gaussian noise to simulate scanner noise.
        RandGaussianNoised(keys=["t1w", "t2w"], prob=0.2, mean=0.0, std=0.05),

        # Random Gaussian blur to simulate partial-volume effects and slight
        # out-of-focus acquisition.
        RandGaussianSmoothd(
            keys=["t1w", "t2w"],
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),

        # Randomly scale global intensity to simulate gain differences between
        # scanners and sessions.
        RandScaleIntensityd(keys=["t1w", "t2w"], prob=0.3, factors=0.1),

        # Concatenate T1 and T2 along the channel dimension: (1, D, H, W) x2 -> (2, D, H, W)
        ConcatItemsd(keys=["t1w", "t2w"], name="image"),

        # Remove the now-redundant individual t1w/t2w keys from the dict.
        DeleteItemsd(keys=["t1w", "t2w"]),

        ToTensord(keys=["image"]),
    ])


def get_val_transforms() -> Compose:
    """
    Build the validation and inference transform pipeline.

    Identical to the training pipeline except:
    - No random augmentations (flip, rotate, affine, noise, blur, scale).
    - The spatial crop always takes the centre of the volume (random_center=False),
      so the same subject always yields the same patch across evaluation runs.

    Returns a MONAI Compose object with the same call signature as the training
    pipeline.
    """
    return Compose([
        LoadImaged(keys=["t1w", "t2w"]),
        EnsureChannelFirstd(keys=["t1w", "t2w"]),
        Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
        Spacingd(keys=["t1w", "t2w"], pixdim=TARGET_VOXEL, mode=("bilinear", "bilinear")),
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1,
            upper=99,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["t1w", "t2w"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["t1w", "t2w"], source_key="t1w"),
        SpatialPadd(keys=["t1w", "t2w"], spatial_size=PATCH_SIZE),

        # Centre crop: random_center=False always extracts the central 96^3 patch.
        # This makes validation metrics reproducible across runs.
        RandSpatialCropd(
            keys=["t1w", "t2w"],
            roi_size=PATCH_SIZE,
            random_size=False,
            random_center=False,
        ),

        ConcatItemsd(keys=["t1w", "t2w"], name="image"),
        DeleteItemsd(keys=["t1w", "t2w"]),
        ToTensord(keys=["image"]),
    ])
