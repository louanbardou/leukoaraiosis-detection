#!/usr/bin/env python3
"""
phase3_train.py
---------------
Binary leukoaraiosis classifier training loop.

This script trains the LeukoBinaryClassifier (Swin UNETR encoder + GAP head)
using a differentiable surrogate loss for the Average Precision (AUPREC) metric,
optimised with the SOAP optimizer from LibAUC.

Why AUPREC and not cross-entropy or AUROC
-----------------------------------------
Leukoaraiosis is rare in the ABCD adolescent cohort (roughly 3-5% of scans are
positive). With such heavy class imbalance, cross-entropy loss would assign most
of its gradient signal to the majority negative class, causing the model to learn
to predict "negative" for almost every input. AUROC is also unreliable here
because it accounts for the True Negative rate, which is trivially high when the
model simply outputs low probabilities for everything.

AUPREC (area under the Precision-Recall curve) evaluates only how well the model
ranks positive examples above negative ones. It is insensitive to the size of
the negative class and therefore gives an honest measure of minority-class
detection performance.

Why APLoss (LibAUC)
-------------------
AUPREC is not directly differentiable because it relies on sorting operations.
LibAUC's APLoss computes a smooth surrogate using a squared-hinge relaxation of
the pairwise ranking objective. Concretely, for every (positive, negative) pair
in the batch, the loss penalises cases where the positive score is not higher
than the negative score by at least the margin. The gamma parameter controls a
moving average that stabilises the gradient estimate across batches.

Why SOAP
--------
SOAP (Stochastic Optimization for Average Precision) is the companion optimizer
for APLoss in LibAUC. It maintains an internal surrogate state that is coupled to
the APLoss objective and uses epoch_decay to anneal the regularisation term over
training. Using SOAP with APLoss is strongly recommended over Adam+APLoss because
SOAP accounts for the non-i.i.d. structure of the pairwise ranking objective.

Cross-validation strategy
--------------------------
StratifiedGroupKFold is used with groups set to subject_id. The "group" constraint
ensures that all sessions belonging to the same subject land entirely in either the
training or validation fold, never split across both. Without this constraint,
the model could learn subject-level anatomy rather than lesion-level features,
inflating validation metrics.

Usage
-----
    # From within the Leuko_abcd directory, with the environment activated:
    python phase3_train.py \\
        --manifest    manifest.csv \\
        --out_dir     runs/phase3_$(date +%Y%m%d_%H%M%S) \\
        --epochs      100 \\
        --batch_size  4 \\
        --lr          1e-4 \\
        --feature_size 48 \\
        --fold        0
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from libauc.losses import APLoss
from libauc.optimizers import SOAP
from libauc.sampler import DualSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from phase1_3_transforms import get_train_transforms, get_val_transforms
from phase2_model import LeukoBinaryClassifier


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class LeukoDataset(Dataset):
    """
    Map-style dataset that reads subject-session rows from the manifest
    and applies a MONAI transform pipeline to load the MRI volumes.

    Parameters
    ----------
    df : pd.DataFrame
        Rows from manifest.csv. Must contain t1w_path, t2w_path, label.
    transform : MONAI Compose
        The transform pipeline to apply to each sample.
    """

    def __init__(self, df: pd.DataFrame, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        # DualSampler looks for a .targets attribute to identify class labels
        self.targets   = self.df["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Load and transform one sample.

        Returns
        -------
        image : torch.Tensor, shape (2, 96, 96, 96)
        label : torch.Tensor, scalar float (0.0 or 1.0)
        """
        row  = self.df.iloc[idx]
        data = self.transform({"t1w": row["t1w_path"], "t2w": row["t2w_path"]})
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return data["image"], label, idx


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility across Python, NumPy, and PyTorch.
    Fixed seeds ensure that the same StratifiedGroupKFold split is produced
    across runs, which is important for comparing experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_fold_splits(df: pd.DataFrame, n_splits: int, fold: int, seed: int):
    """
    Return train and validation DataFrames for one StratifiedGroupKFold fold.

    The groups are set to subject_id so that all sessions from one subject
    go to either train or val, never both. Stratification preserves the
    positive/negative ratio in each fold.

    Parameters
    ----------
    df : pd.DataFrame
        Full manifest with subject_id and label columns.
    n_splits : int
        Number of folds (5 recommended).
    fold : int
        Which fold to use as the validation set (0-indexed).
    seed : int
        Random seed passed to StratifiedGroupKFold.

    Returns
    -------
    (train_df, val_df) : tuple of pd.DataFrame
    """
    sgkf   = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["subject_id"].values
    labels = df["label"].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        if fold_idx == fold:
            return df.iloc[train_idx], df.iloc[val_idx]

    raise ValueError(f"Fold {fold} not found in {n_splits}-fold split.")


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(args) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifest and report class balance
    df = pd.read_csv(args.manifest)
    print(
        f"Manifest: {len(df)} rows, "
        f"{df['label'].sum()} positive ({df['label'].mean() * 100:.1f}%)"
    )

    train_df, val_df = get_fold_splits(df, n_splits=5, fold=args.fold, seed=args.seed)
    print(
        f"Fold {args.fold}: "
        f"train={len(train_df)} ({train_df['label'].sum()} pos), "
        f"val={len(val_df)} ({val_df['label'].sum()} pos)"
    )

    # Build data loaders.
    # persistent_workers=True keeps the worker processes alive between epochs,
    # avoiding the overhead of re-spawning them at each epoch start.
    # DualSampler guarantees at least num_pos=1 positive per batch, which is
    # required by APLoss (it asserts pos_mask.sum() > 0 every forward call).
    train_dataset = LeukoDataset(train_df, get_train_transforms())
    train_sampler = DualSampler(
        train_dataset,
        batch_size=args.batch_size,
        num_pos=1,
        sampling_rate=None,
        random_seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        LeukoDataset(val_df, get_val_transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model = LeukoBinaryClassifier(feature_size=args.feature_size).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M")

    # APLoss needs to know the number of positive examples in the training set
    # to correctly scale the margin term. pos_len is also used to initialise the
    # surrogate moving-average buffer.
    n_pos = int(train_df["label"].sum())
    print(f"Positive prior: {n_pos / len(train_df):.4f} ({n_pos} positives)")

    loss_fn = APLoss(
        data_len=len(train_df),
        margin=1.0,
        gamma=0.9,
    )

    # SOAP maintains an internal dual variable coupled to the APLoss objective.
    # epoch_decay slowly reduces the regularisation strength over training.
    optimizer = SOAP(
        model.parameters(),
        lr=args.lr,
        epoch_decay=1e-6,
        weight_decay=1e-5,
    )

    history     = []
    best_auprec = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Training pass
        model.train()
        tr_logits, tr_labels = [], []

        for images, batch_labels, batch_idx in train_loader:
            images       = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_idx    = batch_idx.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)       # (B,)
            loss   = loss_fn(logits, batch_labels, batch_idx)
            loss.backward()

            # Gradient clipping prevents exploding gradients during the first
            # few epochs when the model weights are far from optimum.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_logits.extend(logits.detach().cpu().float().tolist())
            tr_labels.extend(batch_labels.cpu().tolist())

        # Validation pass (no gradients needed)
        model.eval()
        va_logits, va_labels = [], []

        with torch.no_grad():
            for images, batch_labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images).squeeze(1)
                va_logits.extend(logits.cpu().float().tolist())
                va_labels.extend(batch_labels.tolist())

        # Convert logits to probabilities for metric computation
        tr_probs = torch.sigmoid(torch.tensor(tr_logits)).numpy()
        va_probs = torch.sigmoid(torch.tensor(va_logits)).numpy()

        tr_auprec = average_precision_score(tr_labels, tr_probs)
        va_auprec = average_precision_score(va_labels, va_probs)
        va_auroc  = roc_auc_score(va_labels, va_probs)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"Train AUPREC={tr_auprec:.4f}  "
            f"Val AUPREC={va_auprec:.4f}  "
            f"Val AUROC={va_auroc:.4f}  "
            f"{elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_auprec": tr_auprec,
            "val_auprec": va_auprec,
            "val_auroc": va_auroc,
        })

        # Save checkpoint whenever validation AUPREC improves
        if va_auprec > best_auprec:
            best_auprec = va_auprec
            ckpt_path   = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auprec": va_auprec,
                "val_auroc": va_auroc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  New best AUPREC={best_auprec:.4f}, checkpoint saved.")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    _plot_history(hist_df, out_dir, best_auprec)
    print(f"\nTraining complete. Best Val AUPREC: {best_auprec:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Plots:      {out_dir / 'training_curves.png'}")


def _plot_history(hist: pd.DataFrame, out_dir: Path, best_auprec: float) -> None:
    epochs = hist["epoch"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves  (best Val AUPREC={best_auprec:.4f})", fontsize=12)

    # ── AUPREC ────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, hist["train_auprec"], label="Train AUPREC", color="steelblue")
    ax.plot(epochs, hist["val_auprec"],   label="Val AUPREC",   color="tomato")
    ax.axhline(y=best_auprec, color="tomato", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPREC")
    ax.set_title("Average Precision (AUPREC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── AUROC ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, hist["val_auroc"], label="Val AUROC", color="seagreen")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("ROC AUC (AUROC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the leukoaraiosis binary classifier (APLoss + SOAP)."
    )
    parser.add_argument("--manifest",      default="manifest.csv")
    parser.add_argument("--out_dir",       default="runs/phase3")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--feature_size",  type=int,   default=48)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index for StratifiedGroupKFold (0-4). Default is 0.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
