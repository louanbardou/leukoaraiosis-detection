#!/usr/bin/env python3
"""
subsample_manifest.py
---------------------
Subsample the manifest to reduce class imbalance.
Keeps all negatives and randomly samples N positives.

Usage
-----
    python subsample_manifest.py \
        --manifest manifest.csv \
        --n_pos    50 \
        --out      manifest_balanced.csv \
        --seed     42
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    parser.add_argument("--n_pos",    type=int, default=50,
                        help="Number of positive subjects to keep (default: 50)")
    parser.add_argument("--out",      default="manifest_balanced.csv")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    if args.n_pos > len(pos):
        print(f"WARNING: requested {args.n_pos} positives but only {len(pos)} available. Keeping all.")
        args.n_pos = len(pos)

    pos_sampled = pos.sample(n=args.n_pos, random_state=args.seed)
    out = pd.concat([pos_sampled, neg]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"Total rows  : {len(out)}")
    print(f"Positives   : {out['label'].sum()}  ({out['label'].mean()*100:.1f}%)")
    print(f"Negatives   : {(out['label']==0).sum()}")
    print(f"Saved to    : {args.out}")

    out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
