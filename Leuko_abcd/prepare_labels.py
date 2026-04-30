#!/usr/bin/env python3
"""
prepare_labels.py
-----------------
Converts the raw ABCD clinical CSV (abcd_combined_from_*.csv) into the
labels.csv format expected by phase1_2_build_manifest.py.

Input columns required:
    id_redcap         : NDAR subject ID, e.g. NDAR_INV0A4P0LWM
    redcap_event_name : REDCap session name, e.g. baseline_year_1_arm_1
    mrif_score        : radiological finding score 1–3 (3 = abnormal)

Output columns:
    subject_id        : BIDS sub- ID, e.g. 0A4P0LWM  (sub- added by manifest builder)
    session           : BIDS session, e.g. ses-00A
    label             : 1 if mrif_score >= 3, else 0

Usage
-----
    python prepare_labels.py \
        --in_csv  abcd_combined_from_2025-10-15.csv \
        --out_csv labels.csv
"""

import argparse
import pandas as pd

# Maps REDCap event names to ABCD BIDS session identifiers (mproc convention)
SESSION_MAP = {
    "baseline_year_1_arm_1":       "ses-00A",
    "1_year_follow_up_y_arm_1":    "ses-02A",
    "2_year_follow_up_y_arm_1":    "ses-04A",
    "3_year_follow_up_y_arm_1":    "ses-06A",
    "4_year_follow_up_y_arm_1":    "ses-08A",
}


def ndar_to_bids(ndar_id: str) -> str:
    """
    NDAR_INV0A4P0LWM  ->  0A4P0LWM
    The manifest builder will prepend 'sub-' automatically.
    """
    s = str(ndar_id).strip()
    if "INV" in s:
        return s.split("INV")[1]
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Convert ABCD clinical CSV to labels.csv for the leukoaraiosis pipeline."
    )
    parser.add_argument("--in_csv",  required=True, help="Path to the raw ABCD CSV")
    parser.add_argument("--out_csv", default="labels.csv", help="Output path (default: labels.csv)")
    parser.add_argument(
        "--threshold", type=int, default=3,
        help="mrif_score >= threshold is labelled 1 (default: 3)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)

    required = {"id_redcap", "redcap_event_name", "mrif_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}\nFound: {list(df.columns)}")

    # Drop rows with no mrif_score
    n_before = len(df)
    df = df.dropna(subset=["mrif_score"])
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing mrif_score")

    # Convert IDs
    df["subject_id"] = df["id_redcap"].apply(ndar_to_bids)

    # Convert sessions
    unknown_sessions = set(df["redcap_event_name"].unique()) - set(SESSION_MAP.keys())
    if unknown_sessions:
        print(f"WARNING: unknown session names (will be dropped): {unknown_sessions}")
    df["session"] = df["redcap_event_name"].map(SESSION_MAP)
    df = df.dropna(subset=["session"])

    # Binarize label
    df["label"] = (df["mrif_score"] >= args.threshold).astype(int)

    # Keep only needed columns
    out = df[["subject_id", "session", "label"]].copy()

    # Summary
    print(f"\nSummary:")
    print(f"  Total rows        : {len(out)}")
    print(f"  Positive (label=1): {out['label'].sum()}  ({out['label'].mean()*100:.1f}%)")
    print(f"  Negative (label=0): {(out['label'] == 0).sum()}")
    print(f"  Unique subjects   : {out['subject_id'].nunique()}")
    print(f"  Sessions          : {sorted(out['session'].unique())}")
    print(f"\nSaved to {args.out_csv}")

    out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
