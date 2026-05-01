#!/usr/bin/env python3
"""
build_manifest_from_disk.py
---------------------------
Disk-first manifest builder.

Instead of starting from a labels CSV and looking for files (which misses subjects
on disk that aren't in any label file), this script:

  1. Walks data_root and finds every sub-*/ses-*/anat/ directory that has BOTH a
     T1w and a T2w NIfTI file.
  2. For each (subject, session) pair found on disk, looks up its label across ALL
     label sources:
       - labels.csv          (manually curated)
       - all_labels_merged.csv (mrif_score → binary)
       - Baseline_healthy.csv  (all label=0)
  3. Only keeps rows that have a label from at least one source.
  4. Writes the result to data/manifest_full.csv.

This catches subjects that are on disk but were missing from the merged labels_all.csv
due to ID format differences or coverage gaps.

Usage
-----
    python scripts/build_manifest_from_disk.py \
        --data_root /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc \
        --labels_dir data/ \
        --out_csv    data/manifest_full.csv

    # Or with $ABCD_IMAGING set by activate_env.sh:
    python scripts/build_manifest_from_disk.py
"""

import argparse
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.parent  # Leuko_abcd/
DATA = HERE / "data"

EVENT_MAP = {
    "baseline_year_1_arm_1":    "ses-00A",
    "1_year_follow_up_y_arm_1": "ses-01A",
    "2_year_follow_up_y_arm_1": "ses-02A",
    "3_year_follow_up_y_arm_1": "ses-03A",
    "4_year_follow_up_y_arm_1": "ses-04A",
    "6_year_follow_up_y_arm_1": "ses-06A",
    "8_year_follow_up_y_arm_1": "ses-08A",
}


def load_all_labels(labels_dir: Path) -> dict:
    """
    Load all label sources and return a dict mapping (subject_id, session) → label.

    subject_id is stored WITHOUT the 'sub-' prefix (bare 8-char ID).
    Priority: labels.csv > all_labels_merged.csv > Baseline_healthy.csv
    """
    label_map = {}  # (bare_id, ses) → label

    # ── 3. Baseline_healthy.csv (lowest priority) ────────────────────────────
    bh_path = labels_dir / "Baseline_healthy.csv"
    if bh_path.exists():
        bh = pd.read_csv(bh_path)
        bh["subject_id"] = bh["id_redcap"].str.replace("NDAR_INV", "")
        bh["session"]    = bh["redcap_event_name"].map(EVENT_MAP).fillna("ses-00A")
        for _, row in bh.iterrows():
            label_map[(row["subject_id"], row["session"])] = 0
        print(f"Baseline_healthy.csv  : {len(bh)} rows loaded")
    else:
        print(f"WARNING: {bh_path} not found")

    # ── 2. all_labels_merged.csv (medium priority) ───────────────────────────
    alm_path = labels_dir / "all_labels_merged.csv"
    if alm_path.exists():
        alm = pd.read_csv(alm_path)
        alm["subject_id"] = alm["subject_id"].str.replace("NDAR_INV", "")
        alm["session"]    = alm["event_name"].map(EVENT_MAP)
        alm["label"]      = (alm["mrif_score"] >= 3).astype(int)
        alm = alm.dropna(subset=["session"])
        for _, row in alm.iterrows():
            label_map[(row["subject_id"], row["session"])] = int(row["label"])
        print(f"all_labels_merged.csv : {len(alm)} rows loaded")
    else:
        print(f"WARNING: {alm_path} not found")

    # ── 1. labels.csv (highest priority) ─────────────────────────────────────
    lbl_path = labels_dir / "labels.csv"
    if lbl_path.exists():
        lbl = pd.read_csv(lbl_path)[["subject_id", "session", "label"]]
        for _, row in lbl.iterrows():
            label_map[(row["subject_id"], row["session"])] = int(row["label"])
        print(f"labels.csv            : {len(lbl)} rows loaded")
    else:
        print(f"WARNING: {lbl_path} not found")

    print(f"\nLabel map total: {len(label_map)} (subject, session) pairs\n")
    return label_map


def scan_disk(data_root: Path, label_map: dict) -> pd.DataFrame:
    """
    Walk data_root, find all sub-*/ses-*/anat/ directories with both T1w and T2w,
    and cross-reference with label_map.
    """
    rows = []
    n_scanned  = 0
    n_both     = 0
    n_labeled  = 0
    n_unlabeled = 0

    for sub_dir in sorted(data_root.glob("sub-*")):
        bare_id = sub_dir.name[4:]  # strip 'sub-'
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name  # e.g. ses-00A
            anat = ses_dir / "anat"
            if not anat.exists():
                continue
            n_scanned += 1
            t1_files = sorted(anat.glob("*T1w.nii.gz"))
            t2_files = sorted(anat.glob("*T2w.nii.gz"))
            if not t1_files or not t2_files:
                continue
            n_both += 1

            # Look up label (try bare_id first, then with sub- prefix in case CSV uses it)
            label = label_map.get((bare_id, ses))
            if label is None:
                label = label_map.get((f"sub-{bare_id}", ses))
            if label is None:
                n_unlabeled += 1
                continue
            n_labeled += 1
            rows.append({
                "subject_id": sub_dir.name,  # sub-XXXXXXXX
                "session":    ses,
                "t1w_path":   str(t1_files[0]),
                "t2w_path":   str(t2_files[0]),
                "label":      int(label),
            })

    print(f"Disk scan results:")
    print(f"  Anat dirs scanned            : {n_scanned}")
    print(f"  Both T1w+T2w present         : {n_both}")
    print(f"  With label → kept            : {n_labeled}")
    print(f"  No label found → skipped     : {n_unlabeled}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build manifest by scanning disk first, then looking up labels."
    )
    parser.add_argument("--data_root",  default=None)
    parser.add_argument("--labels_dir", default=str(DATA))
    parser.add_argument("--out_csv",    default=str(DATA / "manifest_full.csv"))
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = os.environ.get("ABCD_IMAGING")
        if not args.data_root:
            parser.error("Provide --data_root or set $ABCD_IMAGING (via activate_env.sh)")

    data_root  = Path(args.data_root)
    labels_dir = Path(args.labels_dir)
    out_csv    = Path(args.out_csv)

    print(f"data_root  : {data_root}")
    print(f"labels_dir : {labels_dir}")
    print(f"out_csv    : {out_csv}\n")

    label_map = load_all_labels(labels_dir)
    df = scan_disk(data_root, label_map)

    if df.empty:
        print("ERROR: no rows found. Check data_root and labels_dir paths.")
        return

    df = df.sort_values(["subject_id", "session"]).reset_index(drop=True)

    print(f"\nManifest summary:")
    print(f"  Total rows       : {len(df)}")
    print(f"  WMA  (label=1)   : {df['label'].sum()}  ({df['label'].mean()*100:.1f}%)")
    print(f"  Healthy (label=0): {(df['label']==0).sum()}  ({(1-df['label'].mean())*100:.1f}%)")
    print(f"  Unique subjects  : {df['subject_id'].nunique()}")

    print(f"\nBreakdown by session:")
    for ses, grp in df.groupby("session"):
        print(f"  {ses:<12}  total={len(grp):>5}  WMA={grp['label'].sum():>4}  healthy={(grp['label']==0).sum():>5}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
