#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from phase1_3_transforms import get_val_transforms, PATCH_SIZE
from phase2_model import LeukoBinaryClassifier

# Grad-CAM (copied here so infer.py is self-contained)

class GradCAM3D:
    """Registers hooks on target_layer and computes 3D Grad-CAM maps."""

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Returns a float32 numpy array of shape PATCH_SIZE with values in [0, 1].
        Higher values indicate regions that most strongly support a positive prediction.
        """
        self.model.zero_grad()
        logit = self.model(input_tensor)
        logit[0, 0].backward()

        grads = self.gradients
        acts  = self.activations

        # Swin Transformer outputs token sequences (B, T, C) rather than spatial
        # tensors. Reshape to (B, C, D, H, W) before computing channel weights.
        if grads.dim() == 3:
            B, T, C = grads.shape
            D = H = W = round(T ** (1 / 3))
            grads = grads.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            acts  = acts.view( B, D, H, W, C).permute(0, 4, 1, 2, 3)

        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam     = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam_up  = F.interpolate(cam, size=PATCH_SIZE, mode="trilinear", align_corners=False)
        cam_np  = cam_up.squeeze().cpu().numpy().astype(np.float32)

        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        return cam_np

# Subject discovery

def find_pairs(directory: str) -> list:
    """
    Recursively find all T1w/T2w NIfTI pairs under the given directory.

    A pair is valid when:
    - A file ending in _T1w.nii.gz exists
    - A file with the same name but _T2w.nii.gz exists in the same folder
    Returns
    
    list of dict, each with keys 't1w', 't2w', 'name'
        'name' is the stem of the T1w file, used for output naming.
    """
    root  = Path(directory)
    pairs = []

    for t1_path in sorted(root.rglob("*_T1w.nii.gz")):
        t2_path = t1_path.parent / t1_path.name.replace("_T1w.nii.gz", "_T2w.nii.gz")
        if t2_path.exists():
            # Use everything up to _T1w as the subject name for output files
            name = t1_path.name.replace("_T1w.nii.gz", "")
            pairs.append({"t1w": str(t1_path), "t2w": str(t2_path), "name": name})
        else:
            print(f"Warning: T1w found but T2w missing, skipping: {t1_path.name}")

    return pairs


# Volume estimation
def estimate_volume_mm3(cam: np.ndarray, threshold: float) -> float:
    """
    Estimate lesion volume by counting supra-threshold Grad-CAM voxels.
    All volumes are processed at 1 mm isotropic resolution, so each voxel = 1 mm3.

    Parameters
    cam : np.ndarray
        Grad-CAM map, values in [0, 1], shape PATCH_SIZE.
    threshold : float
        Voxels above this value are counted as lesion.
    float
        Estimated volume in mm3.
    """
    return float((cam >= threshold).sum()) * 1.0   # 1 mm3 per voxel



# Visualisation


def _pick_slices(volume: np.ndarray, n: int = 5) -> list:
    """
    Pick n equally spaced axial slice indices, avoiding the top and bottom 10%
    of the volume where there is often little brain tissue.
    """
    d = volume.shape[2]
    lo, hi = int(d * 0.15), int(d * 0.85)
    return [int(lo + i * (hi - lo) / (n - 1)) for i in range(n)]


def save_figure(
    t1_vol: np.ndarray,
    t2_vol: np.ndarray,
    cam:    np.ndarray,
    prob:   float,
    volume_mm3: float,
    name:   str,
    out_path: Path,
    cam_threshold: float,
) -> None:
    """
    Save a diagnostic PNG figure with three rows:
        Row 1: T1w axial slices
        Row 2: T2w axial slices
        Row 3: Grad-CAM heatmap overlaid on T1w

    Parameters
    ----------
    t1_vol : np.ndarray
        T1w volume after resampling (D, H, W) or (C, D, H, W).
    t2_vol : np.ndarray
        T2w volume after resampling.
    cam : np.ndarray
        Grad-CAM heatmap, values in [0, 1], shape PATCH_SIZE.
    prob : float
        Classification probability (0-1).
    volume_mm3 : float
        Estimated lesion volume in mm3.
    name : str
        Subject name, used in the figure title.
    out_path : Path
        Path where the PNG will be saved.
    cam_threshold : float
        The threshold used for volume estimation, shown as a contour line.
    """
    # Squeeze channel dimension if present
    if t1_vol.ndim == 4:
        t1_vol = t1_vol[0]
    if t2_vol.ndim == 4:
        t2_vol = t2_vol[0]

    slices = _pick_slices(t1_vol, n=5)
    n_cols = len(slices)
    n_rows = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    status = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    fig.suptitle(
        f"{name}\n"
        f"Prediction: {status}   Confidence: {prob * 100:.1f}%   "
        f"Estimated volume: {volume_mm3:.0f} mm3 ({volume_mm3 / 1000:.2f} cm3)",
        fontsize=11,
        y=1.01,
    )

    for col_idx, sl in enumerate(slices):
        t1_slice  = t1_vol[:, :, sl].T
        t2_slice  = t2_vol[:, :, sl].T
        cam_slice = cam[:, :, sl].T

        # Row 0: T1w
        ax = axes[0, col_idx]
        ax.imshow(t1_slice, cmap="gray", aspect="auto", origin="lower")
        ax.set_title(f"z={sl}", fontsize=8)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("T1w", fontsize=9)

        # Row 1: T2w
        ax = axes[1, col_idx]
        ax.imshow(t2_slice, cmap="gray", aspect="auto", origin="lower")
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("T2w", fontsize=9)

        # Row 2: Grad-CAM overlay on T1w
        ax = axes[2, col_idx]
        ax.imshow(t1_slice, cmap="gray", aspect="auto", origin="lower")
        overlay = ax.imshow(
            cam_slice,
            cmap="hot",
            alpha=0.55,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        # Draw the threshold contour so the estimated volume region is visible
        if cam_slice.max() >= cam_threshold:
            ax.contour(cam_slice, levels=[cam_threshold], colors=["cyan"], linewidths=0.8)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("Grad-CAM", fontsize=9)

    # Shared colorbar for the heatmap row
    cbar = fig.colorbar(overlay, ax=axes[2, :], fraction=0.02, pad=0.02)
    cbar.set_label("Attention", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



# Per-subject inference


def run_one(
    t1w_path:      str,
    t2w_path:      str,
    name:          str,
    model:         LeukoBinaryClassifier,
    gradcam:       GradCAM3D,
    device:        torch.device,
    out_dir:       Path,
    cam_threshold: float,
) -> dict:
    """
    Run inference on a single T1w/T2w pair and save outputs.

    Returns a dict with keys: name, t1w, t2w, probability, volume_mm3, prediction.
    """
    transform = get_val_transforms()

    try:
        data = transform({"t1w": t1w_path, "t2w": t2w_path})
    except Exception as exc:
        print(f"  ERROR loading {name}: {exc}")
        return {"name": name, "t1w": t1w_path, "t2w": t2w_path,
                "probability": None, "volume_mm3": None, "prediction": "ERROR"}

    img = data["image"].unsqueeze(0).to(device)   # (1, 2, 96, 96, 96)

    # Grad-CAM requires gradients; model is in eval mode but autograd is enabled
    with torch.enable_grad():
        cam = gradcam.compute(img)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    volume_mm3 = estimate_volume_mm3(cam, cam_threshold)
    prediction = "POSITIVE" if prob >= 0.5 else "NEGATIVE"

    # Save heatmap as NIfTI using the affine from the T1w file so it can be
    # loaded alongside the original MRI in FSLeyes or ITK-SNAP
    ref_nii  = nib.load(t1w_path)
    heatmap_path = out_dir / f"{name}_heatmap.nii.gz"
    nib.save(nib.Nifti1Image(cam, affine=ref_nii.affine), heatmap_path)

    # Extract individual modality volumes for the figure.
    # data["image"] shape: (2, D, H, W); channel 0 = T1, channel 1 = T2.
    t1_vol = data["image"][0].numpy()
    t2_vol = data["image"][1].numpy()
    figure_path = out_dir / f"{name}_report.png"
    save_figure(t1_vol, t2_vol, cam, prob, volume_mm3, name, figure_path, cam_threshold)

    print(
        f"  {name}\n"
        f"    Prediction : {prediction}\n"
        f"    Confidence : {prob * 100:.1f}%\n"
        f"    Est. volume: {volume_mm3:.0f} mm3  ({volume_mm3 / 1000:.2f} cm3)\n"
        f"    Heatmap    : {heatmap_path}\n"
        f"    Figure     : {figure_path}"
    )

    return {
        "name": name,
        "t1w": t1w_path,
        "t2w": t2w_path,
        "probability": round(prob, 4),
        "volume_mm3": round(volume_mm3, 1),
        "prediction": prediction,
    }



# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leukoaraiosis inference: classify a T1w+T2w pair and produce a heatmap."
    )
    # Input: either a single pair or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--t1w",
        help="Path to the T1w NIfTI file. Must be used with --t2w.",
    )
    input_group.add_argument(
        "--dir",
        help=(
            "Directory to scan recursively for T1w/T2w pairs. "
            "The script finds every *_T1w.nii.gz and pairs it with "
            "*_T2w.nii.gz in the same folder."
        ),
    )
    parser.add_argument(
        "--t2w",
        help="Path to the T2w NIfTI file. Required when using --t1w.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the Phase 3 best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        default="inference_results",
        help="Directory where heatmaps and figures are saved (default: inference_results/).",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=48,
        help="Must match the value used during Phase 3 training (default: 48).",
    )
    parser.add_argument(
        "--cam_threshold",
        type=float,
        default=0.30,
        help=(
            "Grad-CAM threshold used for volume estimation and figure contours. "
            "Voxels with Grad-CAM >= threshold are counted as lesion. "
            "Default: 0.30. Raise this value to count only the most activated regions."
        ),
    )
    args = parser.parse_args()

    # Validate --t1w requires --t2w
    if args.t1w and not args.t2w:
        parser.error("--t1w requires --t2w.")

    # Build the list of subject pairs to process
    if args.t1w:
        subjects = [{"t1w": args.t1w, "t2w": args.t2w,
            "name": Path(args.t1w).name.replace("_T1w.nii.gz", "")}]
    else:
        subjects = find_pairs(args.dir)
        if not subjects:
            print(f"No T1w/T2w pairs found under: {args.dir}")
            sys.exit(1)
        print(f"Found {len(subjects)} T1w/T2w pair(s) under {args.dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load trained model
    model = LeukoBinaryClassifier(feature_size=args.feature_size).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Checkpoint loaded: epoch {ckpt['epoch']}, "
        f"Val AUPREC={ckpt['val_auprec']:.4f}\n"
    )

    gradcam = GradCAM3D(model, model.get_cam_target())

    # Run inference
    results = []
    for subj in subjects:
        result = run_one(
            t1w_path=subj["t1w"],
            t2w_path=subj["t2w"],
            name=subj["name"],
            model=model,
            gradcam=gradcam,
            device=device,
            out_dir=out_dir,
            cam_threshold=args.cam_threshold,
        )
        results.append(result)
        print()

    # Write a summary CSV with all results
    csv_path = out_dir / "results.csv"
    if results:
        fieldnames = ["name", "prediction", "probability", "volume_mm3", "t1w", "t2w"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()


"""
For each subject, the script produces:
    - A Grad-CAM heatmap saved as a NIfTI file (.nii.gz)
    - A PNG figure showing T1w/T2w slices with the heatmap overlay
    - A confidence score (probability 0-1) printed to the console
    - An estimated lesion volume in mm3, derived by thresholding the heatmap

All volumes are resampled to 1 mm isotropic before processing, so each voxel
represents exactly 1 mm3. The lesion volume is estimated by counting the voxels
where the Grad-CAM value exceeds --cam_threshold (default 0.30) and multiplying
by 1 mm3.

Single subject (specify both files explicitly):
    python infer.py \\
        --t1w        /path/to/sub-XXXX_ses-00A_T1w.nii.gz \\
        --t2w        /path/to/sub-XXXX_ses-00A_T2w.nii.gz \\
        --checkpoint runs/phase3_YYYYMMDD/best_model.pt \\
        --out_dir    results/

Directory (scans recursively for T1w/T2w pairs):
    python infer.py \\
        --dir        group/abcd/subdir/sub-XXXX/ \\
        --checkpoint runs/phase3_YYYYMMDD/best_model.pt \\
        --out_dir    results/

The --dir mode finds every *_T1w.nii.gz file under the given path and pairs it
with the corresponding *_T2w.nii.gz in the same folder.
"""
