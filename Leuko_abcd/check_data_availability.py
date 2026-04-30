#!/usr/bin/env python3
"""
check_data_availability.py
--------------------------
Audit what MRI data is actually available on disk vs what is in labels.csv.

Reports:
  - Total subjects in labels.csv
  - How many have T1w + T2w on disk (WMA vs healthy)
  - How many are missing one or both files
  - Breakdown by session

Usage
-----
    python check_data_availability.py \
        --labels    labels.csv \
        --data_root /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc

    # Or via environment variable (set by activate_env.sh):
    python check_data_availability.py --labels labels.csv
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def check(labels_csv: str, data_root: str) -> None:
    data_root = Path(data_root)
    df = pd.read_csv(labels_csv)

    # Normalise IDs
    def norm_subj(s): return s if str(s).startswith("sub-") else f"sub-{s}"
    def norm_ses(s):  return s if str(s).startswith("ses-") else f"ses-{s}"
    df["subject_id"] = df["subject_id"].apply(norm_subj)
    df["session"]    = df["session"].apply(norm_ses)
    df["label"]      = df["label"].astype(int)

    print(f"\n{'='*55}")
    print(f"  DATA AVAILABILITY REPORT")
    print(f"  data_root : {data_root}")
    print(f"{'='*55}")
    print(f"\n  Labels CSV   : {len(df):>6} subjects total")
    print(f"  WMA  (1)     : {df['label'].sum():>6}  ({df['label'].mean()*100:.1f}%)")
    print(f"  Healthy (0)  : {(df['label']==0).sum():>6}  ({(1-df['label'].mean())*100:.1f}%)")

    # Check file existence
    has_t1  = []
    has_t2  = []
    for _, row in df.iterrows():
        subj = row["subject_id"]
        ses  = row["session"]
        anat = data_root / subj / ses / "anat"
        t1   = anat / f"{subj}_{ses}_run-01_T1w.nii.gz"
        t2   = anat / f"{subj}_{ses}_run-01_T2w.nii.gz"
        has_t1.append(t1.exists())
        has_t2.append(t2.exists())

    df["has_t1"] = has_t1
    df["has_t2"] = has_t2
    df["both"]   = df["has_t1"] & df["has_t2"]
    df["t1_only"]= df["has_t1"] & ~df["has_t2"]
    df["t2_only"]= ~df["has_t1"] & df["has_t2"]
    df["neither"]= ~df["has_t1"] & ~df["has_t2"]

    print(f"\n{'─'*55}")
    print(f"  FILE AVAILABILITY")
    print(f"{'─'*55}")
    print(f"  Both T1w+T2w available  : {df['both'].sum():>6}")
    print(f"    → WMA                 : {df[df['both']]['label'].sum():>6}")
    print(f"    → Healthy             : {(df[df['both']]['label']==0).sum():>6}")
    print(f"  T1w only (missing T2w)  : {df['t1_only'].sum():>6}")
    print(f"  T2w only (missing T1w)  : {df['t2_only'].sum():>6}")
    print(f"  Neither file found      : {df['neither'].sum():>6}")

    print(f"\n{'─'*55}")
    print(f"  BREAKDOWN BY SESSION (subjects with both files)")
    print(f"{'─'*55}")
    available = df[df["both"]]
    for ses, grp in available.groupby("session"):
        n_wma     = grp["label"].sum()
        n_healthy = (grp["label"] == 0).sum()
        print(f"  {ses:<12}  total={len(grp):>5}  WMA={n_wma:>4}  healthy={n_healthy:>5}")

    print(f"\n{'─'*55}")
    print(f"  USABLE FOR TRAINING (both files present)")
    print(f"{'─'*55}")
    n_avail = df["both"].sum()
    n_wma   = df[df["both"]]["label"].sum()
    n_neg   = (df[df["both"]]["label"] == 0).sum()
    print(f"  Total usable : {n_avail}")
    print(f"  WMA          : {n_wma}  ({n_wma/n_avail*100:.1f}%)")
    print(f"  Healthy      : {n_neg}  ({n_neg/n_avail*100:.1f}%)")
    print(f"\n  → Current manifest uses {n_wma} WMA + ~371 healthy")
    print(f"  → Could add {n_neg - 371} more healthy subjects\n")

    # Save full availability table
    out = Path(labels_csv).parent / "data_availability.csv"
    df[["subject_id","session","label","has_t1","has_t2","both"]].to_csv(out, index=False)
    print(f"  Full table saved to: {out}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",    default="labels.csv")
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ.get("ABCD_IMAGING")
        if not args.data_root:
            parser.error("Provide --data_root or set $ABCD_IMAGING (via activate_env.sh)")

    check(args.labels, args.data_root)
