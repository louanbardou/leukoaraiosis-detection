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
    python phase3_train.py \\
        --manifest    data/manifest_full.csv \\
        --out_dir     runs/phase3_$(date +%Y%m%d_%H%M%S) \\
        --epochs      50 \\
        --batch_size  4 \\
        --lr          1e-5 \\
        --feature_size 48 \\
        --fold        0 \\
        --wandb_project leuko-abcd
"""

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from libauc.losses import APLoss
from libauc.optimizers import SOAP
from libauc.sampler import DualSampler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
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
    cache_dir : str or None
        If set, preprocessed tensors are saved as .pt files on first access
        and reloaded from disk on subsequent epochs — avoids re-reading NIfTI
        files from slow NFS storage every epoch.
    """

    def __init__(self, df: pd.DataFrame, transform, cache_dir=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.targets   = self.df["label"].astype(int).tolist()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        if self.cache_dir is not None:
            key        = f"{row['subject_id']}_{row['session']}.pt"
            cache_path = self.cache_dir / key
            if cache_path.exists():
                try:
                    image = torch.load(cache_path, weights_only=False)
                    return image, label, idx
                except Exception:
                    cache_path.unlink(missing_ok=True)
            data  = self.transform({"t1w": row["t1w_path"], "t2w": row["t2w_path"]})
            image = data["image"]
            torch.save(image, cache_path)
            return image, label, idx

        data = self.transform({"t1w": row["t1w_path"], "t2w": row["t2w_path"]})
        return data["image"], label, idx


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_fold_splits(df: pd.DataFrame, n_splits: int, fold: int, seed: int):
    sgkf   = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups = df["subject_id"].values
    labels = df["label"].values

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        if fold_idx == fold:
            return df.iloc[train_idx], df.iloc[val_idx]

    raise ValueError(f"Fold {fold} not found in {n_splits}-fold split.")


# -------------------------------------------------------------------------
# W&B logging helpers
# -------------------------------------------------------------------------

def wandb_pr_curve(labels, probs, split="val"):
    """Log a Precision-Recall curve as a native W&B plot."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    # W&B expects a table: each row is one (recall, precision) point
    data = [[r, p] for r, p in zip(recall, precision)]
    table = wandb.Table(data=data, columns=["recall", "precision"])
    return wandb.plot.line(table, "recall", "precision",
                           title=f"{split} Precision-Recall Curve")


def wandb_roc_curve(labels, probs, split="val"):
    """Log an ROC curve as a native W&B plot."""
    fpr, tpr, _ = roc_curve(labels, probs)
    data = [[f, t] for f, t in zip(fpr, tpr)]
    table = wandb.Table(data=data, columns=["fpr", "tpr"])
    return wandb.plot.line(table, "fpr", "tpr",
                           title=f"{split} ROC Curve")


def wandb_score_hist(labels, probs):
    """Log overlapping histograms of predicted probabilities for pos vs neg."""
    labels_arr = np.array(labels)
    probs_arr  = np.array(probs)
    pos_scores = probs_arr[labels_arr == 1].tolist()
    neg_scores = probs_arr[labels_arr == 0].tolist()

    data = [[s, "WMA (pos)"] for s in pos_scores] + \
           [[s, "Healthy (neg)"] for s in neg_scores]
    table = wandb.Table(data=data, columns=["score", "class"])
    return wandb.plot.histogram(table, "score",
                                title="Val score distribution (pos vs neg)")


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(args) -> None:
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load manifest ──────────────────────────────────────────────────────
    df = pd.read_csv(args.manifest)
    n_pos   = int(df["label"].sum())
    n_total = len(df)
    print(f"Manifest: {n_total} rows, {n_pos} positive ({n_pos/n_total*100:.1f}%)")

    train_df, val_df = get_fold_splits(df, n_splits=5, fold=args.fold, seed=args.seed)
    n_train_pos = int(train_df["label"].sum())
    n_val_pos   = int(val_df["label"].sum())
    print(
        f"Fold {args.fold}: "
        f"train={len(train_df)} ({n_train_pos} pos), "
        f"val={len(val_df)} ({n_val_pos} pos)"
    )

    # ── W&B init ───────────────────────────────────────────────────────────
    run = wandb.init(
        project  = args.wandb_project,
        entity   = args.wandb_entity or None,
        name     = args.run_name or f"fold{args.fold}_lr{args.lr}_fs{args.feature_size}",
        tags     = [f"fold{args.fold}", "swin-unetr", "aploss", "soap"],
        config   = {
            # hyperparameters
            "model":         "LeukoBinaryClassifier",
            "encoder":       "SwinUNETR",
            "feature_size":  args.feature_size,
            "loss":           "APLoss",
            "optimizer":      "SOAP",
            "lr":             args.lr,
            "epoch_decay":    1e-6,
            "weight_decay":   args.weight_decay,
            "aploss_margin":  1.0,
            "aploss_gamma":   args.aploss_gamma,
            "epoch_decay":    args.epoch_decay,
            "dropout":        args.dropout,
            "freeze_epochs":  args.freeze_epochs,
            "batch_size":     args.batch_size,
            "epochs":         args.epochs,
            "grad_clip":      1.0,
            # data
            "manifest":      args.manifest,
            "fold":          args.fold,
            "n_folds":       5,
            "n_train":       len(train_df),
            "n_val":         len(val_df),
            "n_train_pos":   n_train_pos,
            "n_val_pos":     n_val_pos,
            "pos_rate":      n_pos / n_total,
            "seed":          args.seed,
            "cache_dir":     args.cache_dir,
        },
        dir      = str(out_dir),
        resume   = "allow",
    )
    print(f"W&B run: {run.url}")

    # ── Data loaders ───────────────────────────────────────────────────────
    if args.cache_dir:
        print(f"Cache dir: {args.cache_dir}  (epoch 1 will be slow — building cache)")

    train_dataset = LeukoDataset(train_df, get_train_transforms(), cache_dir=args.cache_dir)
    train_sampler = DualSampler(
        train_dataset,
        batch_size   = args.batch_size,
        num_pos      = 1,
        sampling_rate= None,
        random_seed  = args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size        = args.batch_size,
        sampler           = train_sampler,
        shuffle           = False,
        num_workers       = 8,
        pin_memory        = True,
        persistent_workers= True,
    )
    val_loader = DataLoader(
        LeukoDataset(val_df, get_val_transforms(), cache_dir=args.cache_dir),
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model    = LeukoBinaryClassifier(feature_size=args.feature_size,
                                     dropout=args.dropout).to(device)
    n_params       = sum(p.numel() for p in model.parameters()) / 1e6
    n_params_head  = sum(p.numel() for p in model.head.parameters()) / 1e6
    print(f"Model parameters: {n_params:.1f}M  (head: {n_params_head:.2f}M)")
    wandb.config.update({"n_params_M": round(n_params, 2)})

    # ── Pretrained weights (optional) ─────────────────────────────────────
    if args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location="cpu", weights_only=False)
        # Checkpoints may store weights under different keys
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        # Strip common prefixes added by DataParallel or different wrapper names
        state_dict = {
            k.replace("module.", "").replace("swinViT.", "backbone.swinViT."): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"Pretrained weights loaded from {args.pretrained_weights}\n"
            f"  missing={len(missing)}  unexpected={len(unexpected)}"
        )
        wandb.config.update({"pretrained_weights": args.pretrained_weights})

    # Watch model: log gradients + weights every 50 batches
    wandb.watch(model, log="gradients", log_freq=50)

    # ── Progressive unfreezing ─────────────────────────────────────────────
    # Phase A (epochs 1..freeze_epochs): freeze backbone, train head only.
    #   Prevents 62M randomly-initialised encoder params from overfitting
    #   on 3-4k samples before the head has learnt anything useful.
    # Phase B (epochs freeze_epochs+1..end): unfreeze backbone with a 10x
    #   lower LR so the pretrained features are fine-tuned gently.
    def set_backbone_grad(requires_grad: bool):
        for p in model.backbone.parameters():
            p.requires_grad = requires_grad

    set_backbone_grad(False)   # start frozen
    print(f"Backbone frozen for first {args.freeze_epochs} epochs.")

    # ── Loss & optimiser ───────────────────────────────────────────────────
    # Only pass head params to SOAP initially; we reinitialise after unfreezing.
    loss_fn   = APLoss(data_len=len(train_df), margin=1.0, gamma=args.aploss_gamma)
    optimizer = SOAP(
        [p for p in model.parameters() if p.requires_grad],
        lr           = args.lr,
        epoch_decay  = args.epoch_decay,
        weight_decay = args.weight_decay,
    )

    history     = []
    best_auprec = 0.0
    global_step = 0
    scheduler   = None   # created at unfreeze epoch

    # ── Epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):

        # ── Unfreeze backbone at freeze_epochs+1 ───────────────────────────
        if epoch == args.freeze_epochs + 1:
            set_backbone_grad(True)
            optimizer = SOAP(
                model.parameters(),
                lr           = args.lr / 10,
                epoch_decay  = args.epoch_decay,
                weight_decay = args.weight_decay,
            )
            # Cosine LR decay over the remaining fine-tuning epochs
            n_finetune_epochs = args.epochs - args.freeze_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_finetune_epochs, eta_min=args.lr / 1000
            )
            print(f"Epoch {epoch}: backbone unfrozen — LR={args.lr/10:.2e}, "
                  f"cosine decay over {n_finetune_epochs} epochs")
            wandb.log({"event/backbone_unfrozen": epoch}, step=global_step)
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_logits, tr_labels = [], []
        epoch_loss, epoch_grad_norm = 0.0, 0.0
        n_batches = 0

        for images, batch_labels, batch_idx in train_loader:
            images       = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_idx    = batch_idx.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss   = loss_fn(logits, batch_labels, batch_idx)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # accumulate for epoch-level averages
            epoch_loss      += loss.item()
            epoch_grad_norm += grad_norm.item()
            n_batches       += 1
            global_step     += 1

            tr_logits.extend(logits.detach().cpu().float().tolist())
            tr_labels.extend(batch_labels.cpu().tolist())

            # ── per-batch metrics ──────────────────────────────────────────
            wandb.log({
                "batch/loss":      loss.item(),
                "batch/grad_norm": grad_norm.item(),
                "batch/step":      global_step,
            }, step=global_step)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        va_logits, va_labels = [], []

        with torch.no_grad():
            for images, batch_labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images).squeeze(1)
                va_logits.extend(logits.cpu().float().tolist())
                va_labels.extend(batch_labels.tolist())

        # ── Metrics ─────────────────────────────────────────────────────────
        tr_probs  = torch.sigmoid(torch.tensor(tr_logits)).numpy()
        va_probs  = torch.sigmoid(torch.tensor(va_logits)).numpy()
        va_labels_arr = np.array(va_labels)

        tr_auprec = average_precision_score(tr_labels, tr_probs)
        va_auprec = average_precision_score(va_labels, va_probs)
        va_auroc  = roc_auc_score(va_labels, va_probs)

        elapsed     = time.time() - t0
        avg_loss    = epoch_loss / n_batches
        avg_gnorm   = epoch_grad_norm / n_batches
        current_lr  = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  "
            f"Train AUPREC={tr_auprec:.4f}  "
            f"Val AUPREC={va_auprec:.4f}  "
            f"Val AUROC={va_auroc:.4f}  "
            f"{elapsed:.0f}s"
        )

        # ── W&B epoch log ───────────────────────────────────────────────────
        backbone_frozen = epoch <= args.freeze_epochs
        log_dict = {
            "epoch":                   epoch,
            # losses & optimiser
            "train/loss":              avg_loss,
            "train/grad_norm":         avg_gnorm,
            "train/lr":                current_lr,
            "train/backbone_frozen":   int(backbone_frozen),
            # AUPREC
            "train/auprec":       tr_auprec,
            "val/auprec":         va_auprec,
            # AUROC
            "val/auroc":          va_auroc,
            # timing
            "perf/epoch_time_s":  elapsed,
        }

        # PR curve, ROC curve, score histogram — logged every 5 epochs + last
        if epoch % 5 == 0 or epoch == args.epochs:
            log_dict["val/pr_curve"]      = wandb_pr_curve(va_labels, va_probs)
            log_dict["val/roc_curve"]     = wandb_roc_curve(va_labels, va_probs)
            log_dict["val/score_dist"]    = wandb_score_hist(va_labels, va_probs)

        wandb.log(log_dict, step=global_step)

        # ── History & checkpoint ────────────────────────────────────────────
        history.append({
            "epoch":        epoch,
            "train_auprec": tr_auprec,
            "val_auprec":   va_auprec,
            "val_auroc":    va_auroc,
            "loss":         avg_loss,
        })

        if va_auprec > best_auprec:
            best_auprec = va_auprec
            ckpt_path   = out_dir / "best_model.pt"
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "val_auprec":          va_auprec,
                "val_auroc":           va_auroc,
                "args":                vars(args),
            }, ckpt_path)
            print(f"  New best AUPREC={best_auprec:.4f}, checkpoint saved.")

            # Log best checkpoint as W&B artifact
            artifact = wandb.Artifact(
                name=f"best-model-fold{args.fold}",
                type="model",
                metadata={"epoch": epoch, "val_auprec": va_auprec, "val_auroc": va_auroc},
            )
            artifact.add_file(str(ckpt_path))
            wandb.log_artifact(artifact)

            # Update run summary so the best value is always visible in the dashboard
            wandb.run.summary["best_val_auprec"] = best_auprec
            wandb.run.summary["best_val_auroc"]  = va_auroc
            wandb.run.summary["best_epoch"]       = epoch

    # ── End of training ─────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    _plot_history(hist_df, out_dir, best_auprec)

    # Log final training curves image to W&B
    wandb.log({"charts/training_curves": wandb.Image(str(out_dir / "training_curves.png"))})

    # Final PR / ROC on val set with best model weights
    ckpt = torch.load(out_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    final_logits, final_labels = [], []
    with torch.no_grad():
        for images, batch_labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            final_logits.extend(logits.cpu().float().tolist())
            final_labels.extend(batch_labels.tolist())
    final_probs = torch.sigmoid(torch.tensor(final_logits)).numpy()

    wandb.log({
        "final/pr_curve":   wandb_pr_curve(final_labels, final_probs, split="final"),
        "final/roc_curve":  wandb_roc_curve(final_labels, final_probs, split="final"),
        "final/score_dist": wandb_score_hist(final_labels, final_probs),
    })

    wandb.finish()

    print(f"\nTraining complete. Best Val AUPREC: {best_auprec:.4f}")
    print(f"Checkpoint: {out_dir / 'best_model.pt'}")
    print(f"Plots:      {out_dir / 'training_curves.png'}")


def _plot_history(hist: pd.DataFrame, out_dir: Path, best_auprec: float) -> None:
    epochs = hist["epoch"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves  (best Val AUPREC={best_auprec:.4f})", fontsize=12)

    ax = axes[0]
    ax.plot(epochs, hist["train_auprec"], label="Train AUPREC", color="steelblue")
    ax.plot(epochs, hist["val_auprec"],   label="Val AUPREC",   color="tomato")
    ax.axhline(y=best_auprec, color="tomato", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUPREC")
    ax.set_title("Average Precision (AUPREC)")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    # Data
    parser.add_argument("--manifest",     default="manifest.csv")
    parser.add_argument("--out_dir",      default="runs/phase3")
    parser.add_argument("--cache_dir",    default=None,
                        help="Directory to cache preprocessed tensors on scratch.")
    # Training
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--feature_size", type=int,   default=48)
    parser.add_argument("--fold",         type=int,   default=0,
                        help="Fold index for StratifiedGroupKFold (0-4).")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dropout",       type=float, default=0.5,
                        help="Dropout in MLP head. Default 0.5 (was 0.3).")
    parser.add_argument("--weight_decay",  type=float, default=1e-3,
                        help="Weight decay for SOAP. Default 1e-3 (was 1e-5).")
    parser.add_argument("--freeze_epochs", type=int,   default=10,
                        help="Freeze backbone for this many epochs, then unfreeze with LR/10.")
    parser.add_argument("--aploss_gamma",  type=float, default=0.1,
                        help="APLoss gamma (moving average coefficient). 0.1 recommended for classification.")
    parser.add_argument("--epoch_decay",   type=float, default=1e-3,
                        help="SOAP epoch_decay (L2 regularisation strength per epoch).")
    # Pretrained weights
    parser.add_argument("--pretrained_weights", default=None,
                        help="Path to SSL/BraTS pretrained SwinUNETR checkpoint (.pth). "
                             "Loaded with strict=False — backbone weights transfer, "
                             "head is always trained from scratch.")
    # W&B
    parser.add_argument("--wandb_project", default="leuko-abcd",
                        help="W&B project name.")
    parser.add_argument("--wandb_entity",  default=None,
                        help="W&B entity (team or username). Defaults to your personal account.")
    parser.add_argument("--run_name",      default=None,
                        help="W&B run name. Auto-generated if not set.")

    args = parser.parse_args()
    train(args)
