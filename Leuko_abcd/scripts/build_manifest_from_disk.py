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


def load_all_labels(labels_dir: Path) -> tuple:
    """
    Load all label sources.

    Returns
    -------
    label_map : dict  (bare_id, session) → label
        For sources with per-session labels (labels.csv, all_labels_merged.csv).
        Priority: labels.csv > all_labels_merged.csv

    healthy_subjects : set of bare_id
        Subjects in Baseline_healthy.csv — treated as label=0 for ANY session
        on disk unless overridden by label_map.
    """
    label_map = {}        # (bare_id, ses) → label  (session-specific)
    healthy_subjects = set()  # bare_id only — covers all sessions

    # ── 3. Baseline_healthy.csv: subject-level label=0 for ALL sessions ───────
    bh_path = labels_dir / "Baseline_healthy.csv"
    if bh_path.exists():
        bh = pd.read_csv(bh_path)
        bh["bare_id"] = bh["id_redcap"].str.replace("NDAR_INV", "")
        healthy_subjects = set(bh["bare_id"])
        print(f"Baseline_healthy.csv  : {len(healthy_subjects)} subjects (label=0 for all sessions)")
    else:
        print(f"WARNING: {bh_path} not found")

    # ── 2. all_labels_merged.csv (session-specific, medium priority) ─────────
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

    # ── 1. labels.csv / labels_corrected.csv (highest priority) ─────────────
    lbl_path = labels_dir / "labels_corrected.csv"
    if not lbl_path.exists():
        lbl_path = labels_dir / "labels.csv"
    if lbl_path.exists():
        lbl = pd.read_csv(lbl_path)[["subject_id", "session", "label"]]
        for _, row in lbl.iterrows():
            label_map[(row["subject_id"], row["session"])] = int(row["label"])
        print(f"{lbl_path.name:<30}: {len(lbl)} rows loaded")
    else:
        print(f"WARNING: labels_corrected.csv and labels.csv not found")

    print(f"\nSession-specific label map : {len(label_map)} pairs")
    print(f"Healthy subject pool       : {len(healthy_subjects)} subjects\n")
    return label_map, healthy_subjects


def scan_disk(data_root: Path, label_map: dict, healthy_subjects: set) -> pd.DataFrame:
    """
    Walk data_root, find all sub-*/ses-*/anat/ with both T1w and T2w, assign labels.

    Label lookup order for each (subject, session):
      1. label_map[(bare_id, ses)]  — session-specific (labels.csv / all_labels_merged)
      2. 0 if bare_id in healthy_subjects  — Baseline_healthy covers all sessions
      3. skip (no label available)
    """
    rows = []
    n_scanned   = 0
    n_both      = 0
    n_labeled   = 0
    n_unlabeled = 0

    for sub_dir in sorted(data_root.glob("sub-*")):
        bare_id = sub_dir.name[4:]  # strip 'sub-'
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            ses = ses_dir.name
            anat = ses_dir / "anat"
            if not anat.exists():
                continue
            n_scanned += 1
            t1_files = sorted(anat.glob("*T1w.nii.gz"))
            t2_files = sorted(anat.glob("*T2w.nii.gz"))
            if not t1_files or not t2_files:
                continue
            n_both += 1

            # 1. Session-specific label (labels.csv or all_labels_merged)
            label = label_map.get((bare_id, ses))
            # 2. Subject-level healthy fallback (Baseline_healthy covers all sessions)
            if label is None and bare_id in healthy_subjects:
                label = 0
            if label is None:
                n_unlabeled += 1
                continue
            n_labeled += 1
            rows.append({
                "subject_id": sub_dir.name,
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

    label_map, healthy_subjects = load_all_labels(labels_dir)
    df = scan_disk(data_root, label_map, healthy_subjects)

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
