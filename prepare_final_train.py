"""
Build the merged training dataset in data/final/.

Merges:
  - data/processed/spectra-nist.pkl       (NIST small molecules)
  - data/processed/spectra-chemotion.pkl  (Chemotion small molecules)
  - data/openspecy/spectra-openspecy.pkl  (Open Specy polymer spectra)

and produces model-ready arrays with SNV normalization, in the same format
as data/snv_train/ (which is what PET_FTIR_CNN_training.ipynb consumes):
  X_spectra.npy    float32, (n, 1650), SNV-normalized
  y_labels.npy     int32,   (n, 20), label order as label_names.txt
  label_names.txt
  sources.csv      per-row source ("nist"/"chemotion"/"openspecy") and
                   polymer identity (openspecy rows only)

Usage (from the repo root):
    .venv/bin/python3 prepare_final_train.py
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
    out_dir = Path("data/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for path, source in [
        ("data/processed/spectra-nist.pkl",      "nist"),
        ("data/processed/spectra-chemotion.pkl", "chemotion"),
        ("data/openspecy/spectra-openspecy.pkl", "openspecy"),
    ]:
        df = pd.read_pickle(path)
        if "source" not in df.columns:
            df["source"] = source
        if "polymer" not in df.columns:
            df["polymer"] = ""
        parts.append(df[["fgroups", "ydata", "source", "polymer"]])
        print(f"  {source:10s}: {len(df)} spectra")

    combined = pd.concat(parts, ignore_index=True)

    X = np.array([snv(np.asarray(row, dtype=np.float64))
                  for row in combined["ydata"]], dtype=np.float32)
    y = np.array([[int(fg.get(label, False)) for label in LABEL_NAMES]
                  for fg in combined["fgroups"]], dtype=np.int32)

    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]
    combined = combined[valid].reset_index(drop=True)
    print(f"\nMerged: {X.shape[0]} samples ({(~valid).sum()} dropped for NaNs)")
    print(f"  X: {X.shape}  y: {y.shape}")

    np.save(out_dir / "X_spectra.npy", X)
    np.save(out_dir / "y_labels.npy", y)
    combined[["source", "polymer"]].to_csv(out_dir / "sources.csv", index=False)
    with open(out_dir / "label_names.txt", "w") as f:
        f.write("\n".join(LABEL_NAMES) + "\n")

    print("\nPer-source counts:")
    print(combined["source"].value_counts().to_string())
    print("\nOpen Specy per-polymer counts:")
    print(combined.loc[combined["source"] == "openspecy", "polymer"]
          .value_counts().to_string())
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
