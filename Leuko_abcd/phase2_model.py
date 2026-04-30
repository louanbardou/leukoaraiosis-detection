"""
phase2_model.py
---------------
Defines the binary leukoaraiosis classifier used in Phase 3 training.

Architecture overview
---------------------
The model is a modified Swin UNETR with the decoder disabled. Only the encoder
is used during classification training (Phases 3 and 4). The decoder is
reactivated in Phase 5 for voxel-level segmentation fine-tuning.

    Input (B, 2, D, H, W)
        |
        +-- Swin UNETR encoder (swinViT)
        |       4-stage hierarchical vision transformer
        |       Each stage: windowed multi-head self-attention + patch merging
        |       Output: list of feature maps at decreasing spatial resolution
        |
        +-- hidden_states[-1]   shape (B, feature_size*32, D/32, H/32, W/32)
        |       The deepest encoder feature map. At 96^3 input with feature_size=48:
        |       shape = (B, 1536, 3, 3, 3)
        |
        +-- Global Average Pool   (B, 1536, 1, 1, 1)
        |       Collapses spatial dimensions to a single vector per channel.
        |       This operation is what makes weakly supervised training possible:
        |       the network is forced to summarise the entire volume into one vector
        |       before making a binary decision, so it must attend to the spatial
        |       regions that are most discriminative for leukoaraiosis.
        |
        +-- MLP classification head   (B, 1)
                LayerNorm -> Dropout -> Linear(1536, 256) -> GELU -> Dropout -> Linear(256, 1)
                Raw logit output; sigmoid is applied externally by the loss function.

Why Swin UNETR
--------------
Standard CNNs use local convolutional kernels, which limits their ability to
model long-range spatial relationships. White matter tracts span large distances
across the brain, making global context important for detecting diffuse lesions.
Swin Transformers compute multi-head self-attention within local windows and then
shift those windows between layers, which propagates information globally while
keeping computational complexity linear (not quadratic) in the number of voxels.

Why Global Average Pooling for weak supervision
-----------------------------------------------
The ABCD dataset provides image-level clinical labels, not voxel-level masks.
With a standard segmentation head the network would need pixel-level supervision.
By inserting a Global Average Pool before the classification head, we force the
encoder to produce high activations specifically at the lesion regions (because
those are the regions that distinguish positive from negative examples). Disabling
the GAP during Grad-CAM inference (Phase 4) then reveals those high-activation
regions as a spatial heatmap.

Why use_checkpoint=True
-----------------------
Gradient checkpointing trades compute for memory: instead of storing all
intermediate activations during the forward pass, it recomputes them during
backpropagation. This reduces peak VRAM usage by roughly 40%, which is necessary
to fit the model and a batch of 4 volumetric inputs into a single A100 GPU.

Hidden dimension arithmetic
---------------------------
Swin UNETR doubles the embedding dimension at each of its 4 encoder stages:
    Stage 1:  feature_size * 1   =  48
    Stage 2:  feature_size * 2   =  96
    Stage 3:  feature_size * 4   = 192
    Stage 4:  feature_size * 8   = 384
    Bottleneck projection: feature_size * 16 * 2 = feature_size * 32 = 1536
The factor of 32 is the product of the 4 doublings (2^4 = 16) times the final
projection factor of 2 applied by the Swin UNETR implementation.

MONAI 1.5.2 compatibility
--------------------------
The img_size parameter was removed from SwinUNETR.__init__() in MONAI 1.5.
The model is now fully resolution-agnostic; spatial dimensions are inferred at
the first forward pass from the input tensor shape.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class LeukoBinaryClassifier(nn.Module):
    """
    Swin UNETR encoder with a Global Average Pool and MLP classification head.

    Parameters
    ----------
    feature_size : int
        Controls the embedding width of the Swin Transformer blocks.
        48  => approximately 62 million parameters (recommended for A100 training)
        24  => approximately 15 million parameters (useful for debugging on CPU)
    dropout : float
        Dropout probability applied in the MLP head. 0.3 is appropriate given
        the small effective positive class size and 5-fold cross-validation splits.
    """

    def __init__(self, feature_size: int = 48, dropout: float = 0.3):
        super().__init__()

        # The full Swin UNETR is instantiated (encoder + decoder) but the decoder
        # is not called during Phases 3 and 4. This design allows Phase 5 to
        # reuse the same object and simply re-enable the decoder without any
        # weight surgery.
        self.backbone = SwinUNETR(
            in_channels=2,            # T1w and T2w concatenated as two channels
            out_channels=2,           # background + lesion (used only in Phase 5)
            feature_size=feature_size,
            use_checkpoint=True,      # gradient checkpointing to reduce VRAM
            spatial_dims=3,
        )

        # The encoder's final hidden dimension is feature_size * 32.
        # See module docstring for the derivation.
        hidden_dim = feature_size * 32   # 1536 for feature_size=48

        # Global Average Pool collapses the spatial dimensions of the feature map
        # to a single scalar per channel: (B, C, D', H', W') -> (B, C, 1, 1, 1)
        self.gap = nn.AdaptiveAvgPool3d(1)

        # MLP head converts the pooled feature vector to a single logit.
        # LayerNorm stabilises the distribution of the pooled features before
        # the linear projection. GELU is used instead of ReLU because it is
        # smoother near zero, which helps with gradient flow in the early epochs.
        self.head = nn.Sequential(
            nn.Flatten(),                     # (B, C, 1, 1, 1) -> (B, C)
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),                # raw logit, no sigmoid here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the encoder and classification head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 2, D, H, W).
            Channel 0 is T1w, channel 1 is T2w.

        Returns
        -------
        torch.Tensor
            Raw logit of shape (B, 1). To get a probability, apply torch.sigmoid().
            The sigmoid is left to the caller so that loss functions that accept
            raw logits (e.g. APLoss, BCEWithLogitsLoss) can operate correctly.
        """
        # swinViT returns a list of hidden state tensors from each encoder stage.
        # hidden_states[0] through hidden_states[3] have increasing channels and
        # decreasing spatial resolution. hidden_states[-1] is the deepest feature map.
        hidden_states = self.backbone.swinViT(x, normalize=True)

        # Deepest feature map: (B, feature_size*32, D/32, H/32, W/32)
        # For 96^3 input with feature_size=48: shape (B, 1536, 3, 3, 3)
        deepest_features = hidden_states[-1]

        pooled = self.gap(deepest_features)   # (B, 1536, 1, 1, 1)
        return self.head(pooled)              # (B, 1)

    def get_cam_target(self) -> nn.Module:
        """
        Return the final Swin Transformer encoder block for Grad-CAM hook
        registration in Phase 4.

        The Grad-CAM algorithm registers forward and backward hooks on a target
        layer to capture activations and gradients. We target the last block of
        the deepest encoder stage (layers4) because it contains the most globally
        abstract feature representations of the input volume.
        """
        return self.backbone.swinViT.layers4[-1]
