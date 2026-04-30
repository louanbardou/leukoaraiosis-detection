#!/usr/bin/env python3
"""
phase4_gradcam.py
-----------------
Generates 3D Grad-CAM heatmaps for every subject in the manifest, then saves
them as NIfTI files that can be viewed in any neuroimaging tool (FSLeyes, ITK-SNAP).

Grad-CAM algorithm
------------------
Gradient-weighted Class Activation Mapping (Grad-CAM) localises the brain
regions that drove the classifier's decision without requiring voxel-level labels.

The algorithm works as follows:

1. Register forward and backward hooks on the target layer (the last encoder
   block of the Swin UNETR, returned by model.get_cam_target()).

2. Run a forward pass. The hook captures the activation tensor A of shape
   (B, C, D', H', W'), where D'xH'xW' is the spatial resolution of the
   deepest encoder feature map.

3. Backpropagate the positive-class logit to the target layer. The hook captures
   the gradient tensor G of the same shape.

4. Compute per-channel importance weights by globally average-pooling the
   gradients: w_c = (1 / (D'*H'*W')) * sum(G_c over spatial dims).
   Each weight w_c represents how much channel c contributed to the classification.

5. Compute the weighted sum of activations: CAM = ReLU(sum_c(w_c * A_c)).
   The ReLU discards channels that would decrease the classification score,
   keeping only the positively contributing features.

6. Upsample the coarse CAM (D' x H' x W') to the patch size (96^3) using
   trilinear interpolation.

7. Normalise to [0, 1] so the maps from different subjects are comparable.

Handling Swin Transformer token sequences
------------------------------------------
Standard CNNs produce spatial tensors (B, C, D, H, W) at every layer, which
Grad-CAM can process directly. Swin Transformers instead output flattened token
sequences of shape (B, T, C) where T = D*H*W. Before computing the weighted sum,
the token sequence is reshaped into a spatial tensor. The spatial dimensions are
inferred by taking the cube root of T (valid because the encoder uses cubic
windows and the input is isotropic).

Aggregate heatmap
-----------------
After all subjects are processed, the per-subject CAMs of all true-positive
subjects are averaged and saved as a group-level NIfTI. This aggregate map
shows which brain regions most consistently drive positive classifications
across the cohort and can be used as a publication figure.

Usage
-----
    python phase4_gradcam.py \\
        --checkpoint runs/phase3_YYYYMMDD/best_model.pt \\
        --manifest   manifest.csv \\
        --out_dir    heatmaps/gradcam_$(date +%Y%m%d)
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1_3_transforms import get_val_transforms
from phase2_model import LeukoBinaryClassifier

PATCH_SIZE = (96, 96, 96)


# -------------------------------------------------------------------------
# Grad-CAM implementation
# -------------------------------------------------------------------------

class GradCAM3D:
    """
    Computes Grad-CAM heatmaps for 3D volumetric inputs.

    Hooks are registered on the target layer at construction time.
    Call compute() once per sample to produce the heatmap for that sample.

    Parameters
    ----------
    model : nn.Module
        The classifier. Must be in eval() mode before calling compute().
    target_layer : nn.Module
        The layer on which to register the hooks. Use model.get_cam_target()
        to get the last encoder block, which gives the richest abstract features.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # The forward hook runs after the forward pass through target_layer.
        # We store a detached copy of the activation to avoid holding a reference
        # to the computation graph (which would prevent garbage collection).
        target_layer.register_forward_hook(self._save_activation)

        # The full backward hook runs during backpropagation when the gradient
        # flows through target_layer. grad_output[0] is the gradient with respect
        # to the output of target_layer.
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def compute(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a single input volume.

        Gradients are enabled inside this method; the caller does not need to
        manage torch.enable_grad() explicitly.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, 2, D, H, W). Batch size must be 1.

        Returns
        -------
        np.ndarray
            Float32 array of shape (96, 96, 96) with values in [0, 1].
            Higher values indicate regions that most strongly contributed to
            a positive (leukoaraiosis) classification.
        """
        self.model.zero_grad()

        # Forward pass. The hooks capture activations and will capture gradients
        # when we call backward() below.
        logit = self.model(input_tensor)   # shape (1, 1)

        # Backpropagate the raw logit of the positive class.
        # We do not apply sigmoid before backward because sigmoid is a monotone
        # transformation; it does not change which spatial regions have the largest
        # gradient magnitude.
        logit[0, 0].backward()

        grads = self.gradients    # (1, C, D', H', W')  or  (1, T, C) for Swin
        acts  = self.activations  # same shape

        # Swin Transformer layers output token sequences of shape (B, T, C)
        # rather than spatial tensors. We need to reshape them to (B, C, D', H', W')
        # before computing the global average pool over spatial dimensions.
        if grads.dim() == 3:
            B, T, C = grads.shape
            # Spatial dimensions are equal (cubic window) so cube root gives each dim.
            D = H = W = round(T ** (1 / 3))
            # Reshape from (B, D*H*W, C) to (B, C, D, H, W)
            grads = grads.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
            acts  = acts.view( B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # Per-channel importance weights: global average pool over spatial dims.
        # Shape: (1, C, 1, 1, 1)
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)

        # Weighted sum of activations. ReLU keeps only positive contributions.
        # Shape: (1, 1, D', H', W')
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))

        # Upsample from the coarse encoder resolution to the patch size (96^3).
        cam_upsampled = F.interpolate(
            cam,
            size=PATCH_SIZE,
            mode="trilinear",
            align_corners=False,
        )

        cam_np = cam_upsampled.squeeze().cpu().numpy().astype(np.float32)

        # Normalise to [0, 1] so maps from different subjects are comparable.
        if cam_np.max() > 0:
            cam_np /= cam_np.max()

        return cam_np


# -------------------------------------------------------------------------
# Volume loading
# -------------------------------------------------------------------------

def load_volume(t1w_path: str, t2w_path: str, device: torch.device) -> torch.Tensor:
    """
    Apply the validation transform pipeline to one T1+T2 pair and return
    the resulting tensor with a batch dimension added.

    Parameters
    ----------
    t1w_path : str
        Absolute path to the T1w NIfTI file.
    t2w_path : str
        Absolute path to the T2w NIfTI file.
    device : torch.device
        Device to move the tensor to before returning.

    Returns
    -------
    torch.Tensor
        Shape (1, 2, 96, 96, 96).
    """
    transform = get_val_transforms()
    data      = transform({"t1w": t1w_path, "t2w": t2w_path})
    return data["image"].unsqueeze(0).to(device)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def run_gradcam(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the trained classifier
    model = LeukoBinaryClassifier(feature_size=args.feature_size).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint: epoch {ckpt['epoch']}, "
        f"Val AUPREC={ckpt['val_auprec']:.4f}"
    )

    # Register Grad-CAM hooks on the deepest encoder block
    gradcam = GradCAM3D(model, model.get_cam_target())

    df      = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_probs, all_labels = [], []
    aggregate_cam = None     # running sum of CAMs for true-positive subjects
    n_positive    = 0

    for _, row in df.iterrows():
        subj  = row["subject_id"]
        ses   = row["session"]
        label = int(row["label"])

        try:
            img = load_volume(row["t1w_path"], row["t2w_path"], device)
        except Exception as exc:
            print(f"Skipping {subj} {ses}: {exc}")
            continue

        # Grad-CAM needs gradients even in eval mode.
        # torch.enable_grad() re-enables the autograd engine inside a no_grad context.
        with torch.enable_grad():
            cam = gradcam.compute(img)

        # Compute classification probability without tracking gradients.
        with torch.no_grad():
            prob = torch.sigmoid(model(img)).item()

        all_probs.append(prob)
        all_labels.append(label)

        # Save individual heatmap as a NIfTI file in the same affine space as T1w.
        # The affine encodes the voxel-to-world coordinate mapping, which allows
        # neuroimaging tools to overlay the heatmap on the original MRI.
        ref_nii  = nib.load(row["t1w_path"])
        cam_nii  = nib.Nifti1Image(cam, affine=ref_nii.affine)
        nii_path = out_dir / f"{subj}_{ses}_gradcam.nii.gz"
        nib.save(cam_nii, nii_path)

        # Accumulate the CAM for this subject into the group-level average,
        # but only for subjects who are truly positive (ground-truth label=1).
        if label == 1:
            aggregate_cam  = cam.copy() if aggregate_cam is None else aggregate_cam + cam
            n_positive    += 1

    # Report global classification metrics over the processed subjects
    if len(all_labels) >= 2 and sum(all_labels) > 0:
        auprec = average_precision_score(all_labels, all_probs)
        auroc  = roc_auc_score(all_labels, all_probs)
        print(f"\nSubjects processed: {len(all_labels)}")
        print(f"AUPREC: {auprec:.4f}   AUROC: {auroc:.4f}")

    # Save the group-level aggregate heatmap
    if aggregate_cam is not None and n_positive > 0:
        mean_cam = aggregate_cam / n_positive

        # Rescale to [0, 1] for display
        lo, hi = mean_cam.min(), mean_cam.max()
        mean_cam = (mean_cam - lo) / (hi - lo + 1e-8)

        # Use an identity affine because the aggregate map is in patch space,
        # not tied to any individual subject's scanner coordinates.
        agg_path = out_dir / "aggregate_gradcam_positives.nii.gz"
        nib.save(
            nib.Nifti1Image(mean_cam.astype(np.float32), affine=np.eye(4)),
            agg_path,
        )
        print(f"Aggregate Grad-CAM ({n_positive} true-positive subjects): {agg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D Grad-CAM heatmaps from a trained Phase 3 checkpoint."
    )
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to best_model.pt from Phase 3 training.")
    parser.add_argument("--manifest",     default="manifest.csv")
    parser.add_argument("--out_dir",      default="heatmaps/gradcam")
    parser.add_argument("--feature_size", type=int, default=48)
    args = parser.parse_args()
    run_gradcam(args)
