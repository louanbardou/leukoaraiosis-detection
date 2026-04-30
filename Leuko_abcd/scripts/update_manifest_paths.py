#!/usr/bin/env python3
"""
update_manifest_paths.py
------------------------
Replaces the data_root prefix in t1w_path and t2w_path columns of the manifest.
Run this after copying data to scratch.

Usage
-----
    python update_manifest_paths.py \
        --manifest manifest_balanced.csv \
        --old_root /mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc \
        --new_root /mnt/scratch/user/lbardou/abcd_leuko \
        --out      manifest_balanced.csv
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",  default="manifest_balanced.csv")
    parser.add_argument("--old_root",  default="/mnt/fac/CX500007_DS1/ABCD/6.1/imaging/derivatives/mproc")
    parser.add_argument("--new_root",  default="/mnt/scratch/user/lbardou/abcd_leuko")
    parser.add_argument("--out",       default=None, help="Output path (default: overwrite manifest)")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df["t1w_path"] = df["t1w_path"].str.replace(args.old_root, args.new_root, regex=False)
    df["t2w_path"] = df["t2w_path"].str.replace(args.old_root, args.new_root, regex=False)

    out = args.out or args.manifest
    df.to_csv(out, index=False)
    print(f"Updated paths in {out}")
    print(f"  {args.old_root}")
    print(f"  → {args.new_root}")
    print(f"Example t1w: {df['t1w_path'].iloc[0]}")

if __name__ == "__main__":
    main()
