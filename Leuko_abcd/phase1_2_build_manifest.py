#!/usr/bin/env python3
"""
phase1_2_build_manifest.py
--------------------------
Builds the manifest CSV that drives all subsequent training and inference steps.

The manifest is a plain table where each row represents one subject-session pair
that is ready to be fed to the model. It links:
    - the binary leukoaraiosis label (from a user-provided CSV)
    - the absolute paths to the T1w and T2w NIfTI files on disk

Only rows for which BOTH the T1w and T2w files actually exist on disk are kept.
This ensures the training pipeline never encounters a missing file at runtime.

Input
-----
labels CSV
    Columns required: subject_id, session, label
    subject_id : folder name under the mproc root, with or without the "sub-" prefix
                 Examples: "sub-0DBRJXKG" and "0DBRJXKG" are both accepted
    session    : ABCD session identifier, with or without the "ses-" prefix
                 Examples: "ses-00A" and "00A" are both accepted
    label      : integer, 0 (healthy) or 1 (leukoaraiosis present)
                 Convention used in the ABCD study: clfind_score >= 3 => label 1

data_root
    Root of the mproc imaging derivatives directory. Expected sub-structure:
        {data_root}/sub-{ID}/ses-{SESSION}/anat/sub-{ID}_ses-{SESSION}_run-01_T1w.nii.gz
        {data_root}/sub-{ID}/ses-{SESSION}/anat/sub-{ID}_ses-{SESSION}_run-01_T2w.nii.gz
    On Wynton: /wynton/group/abcd/6.0/imaging/derivatives/mproc

Output
------
manifest CSV
    Columns: subject_id, session, t1w_path, t2w_path, label
    One row per valid subject-session pair.

Usage
-----
    python phase1_2_build_manifest.py \\
        --labels    labels.csv \\
        --data_root /wynton/group/abcd/6.0/imaging/derivatives/mproc \\
        --out_csv   manifest.csv

    # Or let the script read the data root from the environment variable
    export ABCD_IMAGING=/wynton/group/abcd/6.0/imaging/derivatives/mproc
    python phase1_2_build_manifest.py --labels labels.csv
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def normalise_subject(raw: str) -> str:
    """
    Ensure the subject ID has the BIDS 'sub-' prefix.

    The labels CSV may contain IDs with or without the prefix.
    Downstream code always expects the prefix because that is how the
    directory names are structured in the mproc tree.

    Examples
    --------
    "0DBRJXKG"     -> "sub-0DBRJXKG"
    "sub-0DBRJXKG" -> "sub-0DBRJXKG"  (no change)
    """
    s = str(raw).strip()
    return s if s.startswith("sub-") else f"sub-{s}"


def normalise_session(raw: str) -> str:
    """
    Ensure the session ID has the BIDS 'ses-' prefix.

    Examples
    --------
    "00A"     -> "ses-00A"
    "ses-00A" -> "ses-00A"  (no change)
    """
    s = str(raw).strip()
    return s if s.startswith("ses-") else f"ses-{s}"


def resolve_nifti_paths(row: pd.Series, data_root: Path) -> tuple:
    """
    Construct the expected T1w and T2w file paths for one subject-session row
    and check whether both files exist on disk.

    The ABCD mproc naming convention is:
        sub-{ID}/ses-{SES}/anat/sub-{ID}_ses-{SES}_run-01_T1w.nii.gz
        sub-{ID}/ses-{SES}/anat/sub-{ID}_ses-{SES}_run-01_T2w.nii.gz

    Parameters
    ----------
    row : pd.Series
        Must contain 'subject_id' and 'session' (already normalised with prefixes).
    data_root : Path
        Root of the mproc directory tree.

    Returns
    -------
    (t1w_path, t2w_path) : (str | None, str | None)
        String paths if the files exist, None otherwise.
    """
    subj = row["subject_id"]
    ses  = row["session"]
    anat = data_root / subj / ses / "anat"

    t1 = anat / f"{subj}_{ses}_run-01_T1w.nii.gz"
    t2 = anat / f"{subj}_{ses}_run-01_T2w.nii.gz"

    return (
        str(t1) if t1.exists() else None,
        str(t2) if t2.exists() else None,
    )


def build_manifest(labels_csv: str, data_root: str, out_csv: str) -> None:
    """
    Load the labels CSV, resolve NIfTI paths, drop incomplete rows, and
    write the resulting manifest to disk.

    Parameters
    ----------
    labels_csv : str
        Path to the user-provided CSV (columns: subject_id, session, label).
    data_root : str
        Root of the mproc imaging derivatives directory.
    out_csv : str
        Output path for the manifest CSV.
    """
    data_root = Path(data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    log.info("Loading labels from %s", labels_csv)
    df = pd.read_csv(labels_csv)

    required_columns = {"subject_id", "session", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"labels CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Normalise IDs so they match the directory names on disk
    df["subject_id"] = df["subject_id"].apply(normalise_subject)
    df["session"]    = df["session"].apply(normalise_session)
    df["label"]      = df["label"].astype(int)

    log.info("Resolving NIfTI file paths for %d rows. This may take a few minutes.", len(df))
    paths = df.apply(resolve_nifti_paths, axis=1, data_root=data_root, result_type="expand")
    df["t1w_path"] = paths[0]
    df["t2w_path"] = paths[1]

    n_before  = len(df)
    df = df.dropna(subset=["t1w_path", "t2w_path"])
    n_dropped = n_before - len(df)

    if n_dropped > 0:
        log.warning("%d rows dropped because T1w or T2w file was not found on disk.", n_dropped)

    log.info("Manifest summary:")
    log.info("  Total rows kept  : %d", len(df))
    log.info("  Positive (1)     : %d  (%.1f%%)", df["label"].sum(), df["label"].mean() * 100)
    log.info("  Negative (0)     : %d", (df["label"] == 0).sum())
    log.info("  Unique subjects  : %d", df["subject_id"].nunique())

    output_columns = ["subject_id", "session", "t1w_path", "t2w_path", "label"]
    df[output_columns].to_csv(out_csv, index=False)
    log.info("Saved manifest to %s", out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build manifest.csv from a labels CSV and a NIfTI data root."
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="CSV with columns: subject_id, session, label",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help=(
            "Root of the mproc imaging derivatives "
            "(e.g. /wynton/group/abcd/6.0/imaging/derivatives/mproc). "
            "Falls back to the $ABCD_IMAGING environment variable if not set."
        ),
    )
    parser.add_argument(
        "--out_csv",
        default="manifest.csv",
        help="Output path for manifest.csv (default: manifest.csv)",
    )
    args = parser.parse_args()

    # Allow data_root to be provided via the environment variable set in activate_env.sh
    if args.data_root is None:
        args.data_root = os.environ.get("ABCD_IMAGING")
        if not args.data_root:
            parser.error(
                "Provide --data_root or set the $ABCD_IMAGING environment variable "
                "(done automatically by activate_env.sh)."
            )

    build_manifest(args.labels, args.data_root, args.out_csv)
