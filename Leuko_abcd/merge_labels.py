#!/usr/bin/env python3
"""
merge_labels.py
---------------
Merge all label sources into a single labels_all.csv ready for build_manifest.

Sources:
  1. labels.csv          — main labels (15451 subjects, WMA + healthy)
  2. Baseline_healthy.csv — additional healthy baseline subjects (label=0)
  3. all_labels_merged.csv — already fully covered by labels.csv, skipped

Output: labels_all.csv  (columns: subject_id, session, label)

Usage
-----
    python merge_labels.py
"""

import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

EVENT_MAP = {
    "baseline_year_1_arm_1":    "ses-00A",
    "2_year_follow_up_y_arm_1": "ses-02A",
    "4_year_follow_up_y_arm_1": "ses-04A",
    "6_year_follow_up_y_arm_1": "ses-06A",
    "8_year_follow_up_y_arm_1": "ses-08A",
}

# ── 1. Load labels.csv ────────────────────────────────────────────────────────
labels = pd.read_csv(HERE / "labels.csv")[["subject_id", "session", "label"]]
labels["label"] = labels["label"].astype(int)
print(f"labels.csv         : {len(labels):>6} rows  "
      f"| WMA={labels['label'].sum()}  healthy={(labels['label']==0).sum()}")

# ── 2. Load Baseline_healthy.csv ──────────────────────────────────────────────
healthy = pd.read_csv(HERE / "Baseline_healthy.csv")
healthy["subject_id"] = healthy["id_redcap"].str.replace("NDAR_INV", "")
healthy["session"]    = healthy["redcap_event_name"].map(EVENT_MAP).fillna("ses-00A")
healthy["label"]      = 0
healthy = healthy[["subject_id", "session", "label"]]

# Keep only subjects not already in labels.csv (avoid duplicates)
existing_keys = set(zip(labels["subject_id"], labels["session"]))
healthy = healthy[
    ~healthy.apply(lambda r: (r["subject_id"], r["session"]) in existing_keys, axis=1)
]
print(f"Baseline_healthy   : {len(healthy):>6} new rows  (all label=0)")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
out = pd.concat([labels, healthy], ignore_index=True)
out = out.sort_values(["subject_id", "session"]).reset_index(drop=True)

print(f"\nMerged total       : {len(out):>6} rows")
print(f"  WMA  (label=1)   : {out['label'].sum():>6}  ({out['label'].mean()*100:.1f}%)")
print(f"  Healthy (label=0): {(out['label']==0).sum():>6}  ({(1-out['label'].mean())*100:.1f}%)")
print(f"  Unique subjects  : {out['subject_id'].nunique():>6}")

out_path = HERE / "labels_all.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print("\nNext step:")
print("  python phase1_2_build_manifest.py \\")
print("      --labels    labels_all.csv \\")
print("      --data_root /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc \\")
print("      --out_csv   manifest_full.csv")
