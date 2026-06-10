"""
Regenerate model-ready training arrays with SNV normalization.

Reads from data/processed/ pickles (already baseline-corrected and smoothed),
re-normalizes each spectrum with Standard Normal Variate (zero mean, unit std),
and saves to data/snv_train/.

SNV output is NOT bounded to [0,1] — the model must be retrained on this data.

Usage:
    python prepare_snv_train.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


LABEL_NAMES = [
    "ester", "carboxylic_acid", "alkane", "alkene", "alcohol",
    "arene", "amine", "ketone", "ether", "imine", "sulfonamide",
    "acyl_halide", "phosphate", "aldehyde", "nitro", "enamine",
    "azo", "sulfonic_acid", "amide", "peroxide",
]


def snv(y):
    std = y.std()
    if std < 1e-10:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - y.mean()) / std).astype(np.float32)


def main():
    processed = Path("data/processed")
    out_dir    = Path("data/snv_train")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pickles...")
    df_nist  = pd.read_pickle(processed / "spectra-nist.pkl")
    df_chem  = pd.read_pickle(processed / "spectra-chemotion.pkl")
    combined = pd.concat([df_nist, df_chem], ignore_index=True)
    print(f"  NIST: {len(df_nist)}  Chemotion: {len(df_chem)}  Total: {len(combined)}")

    print("Applying SNV normalization...")
    X = np.array([snv(row["ydata"]) for _, row in combined.iterrows()],
                 dtype=np.float32)

    y = np.array([
        [int(row.get(label, False)) for label in LABEL_NAMES]
        for row in combined["fgroups"]
    ], dtype=np.int32)

    valid = ~np.isnan(X).any(axis=1)
    X, y  = X[valid], y[valid]

    print(f"  X shape: {X.shape}  min={X.min():.3f}  max={X.max():.3f}  NaNs: {np.isnan(X).sum()}")
    print(f"  y shape: {y.shape}")

    np.save(out_dir / "X_spectra.npy", X)
    np.save(out_dir / "y_labels.npy",  y)

    with open(out_dir / "label_names.txt", "w") as f:
        f.write("\n".join(LABEL_NAMES) + "\n")

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
