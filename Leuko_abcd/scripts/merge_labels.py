#!/usr/bin/env python3
"""
merge_labels.py
---------------
Merge all label sources into a single labels_all.csv ready for build_manifest.

Sources:
  1. labels.csv           — main labels (subject_id, session, label)
  2. all_labels_merged.csv — additional entries with mrif_score → converted to label
  3. Baseline_healthy.csv  — extra healthy baseline subjects (label=0)

Priority rule: if a (subject, session) pair exists in multiple sources,
the label from labels.csv takes precedence (it was manually curated).

Output: labels_all.csv  (columns: subject_id, session, label)

Usage
-----
    python scripts/merge_labels.py
"""

import pandas as pd
from pathlib import Path

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

# ── 1. labels.csv (primary source) ───────────────────────────────────────────
labels = pd.read_csv(DATA / "labels.csv")[["subject_id", "session", "label"]]
labels["label"] = labels["label"].astype(int)
print(f"labels.csv          : {len(labels):>6} rows | WMA={labels['label'].sum()} healthy={(labels['label']==0).sum()}")

existing_keys = set(zip(labels["subject_id"], labels["session"]))

# ── 2. all_labels_merged.csv (mrif_score → binary label) ─────────────────────
merged = pd.read_csv(DATA / "all_labels_merged.csv")
merged["subject_id"] = merged["subject_id"].str.replace("NDAR_INV", "")
merged["session"]    = merged["event_name"].map(EVENT_MAP)
merged["label"]      = (merged["mrif_score"] >= 3).astype(int)
merged = merged[["subject_id", "session", "label"]].dropna(subset=["session"])
# Keep only pairs NOT already in labels.csv
merged_new = merged[~merged.apply(lambda r: (r["subject_id"], r["session"]) in existing_keys, axis=1)]
existing_keys |= set(zip(merged_new["subject_id"], merged_new["session"]))
print(f"all_labels_merged   : {len(merged_new):>6} new rows | WMA={merged_new['label'].sum()} healthy={(merged_new['label']==0).sum()}")

# ── 3. Baseline_healthy.csv (all label=0) ─────────────────────────────────────
healthy = pd.read_csv(DATA / "Baseline_healthy.csv")
healthy["subject_id"] = healthy["id_redcap"].str.replace("NDAR_INV", "")
healthy["session"]    = healthy["redcap_event_name"].map(EVENT_MAP).fillna("ses-00A")
healthy["label"]      = 0
healthy = healthy[["subject_id", "session", "label"]]
healthy_new = healthy[~healthy.apply(lambda r: (r["subject_id"], r["session"]) in existing_keys, axis=1)]
print(f"Baseline_healthy    : {len(healthy_new):>6} new rows | all label=0")

# ── 4. Merge & save ───────────────────────────────────────────────────────────
out = pd.concat([labels, merged_new, healthy_new], ignore_index=True)
out = out.sort_values(["subject_id", "session"]).reset_index(drop=True)

print(f"\nMerged total        : {len(out):>6} rows")
print(f"  WMA  (label=1)    : {out['label'].sum():>6}  ({out['label'].mean()*100:.1f}%)")
print(f"  Healthy (label=0) : {(out['label']==0).sum():>6}  ({(1-out['label'].mean())*100:.1f}%)")
print(f"  Unique subjects   : {out['subject_id'].nunique():>6}")

out_path = DATA / "labels_all.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print("\nNext step:")
print("  python phase1_2_build_manifest.py --labels data/labels_all.csv --data_root /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc --out_csv data/manifest_full.csv")
