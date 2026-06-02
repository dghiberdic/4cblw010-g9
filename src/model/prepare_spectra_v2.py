"""
prepare_spectra_v2.py

Converts processed Chemotion and NIST spectrum pickle files into model-ready
NumPy arrays.

Input:
data/processed/spectra-chemotion.pkl
data/processed/spectra-nist.pkl

Output:
data/processed/X_spectra.npy
data/processed/y_labels.npy
data/processed/label_names.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd


LABEL_NAMES = [
    "ester",
    "carboxylic_acid",
    "alkane",
    "alkene",
    "alcohol",
    "arene",
    "amine",
    "ketone",
    "ether",
    "imine",
    "sulfonamide",
    "acyl_halide",
    "phosphate",
    "aldehyde",
    "nitro",
    "enamine",
    "azo",
    "sulfonic_acid",
    "amide",
    "peroxide",
]


def load_pickle(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_pickle(path)

    required_columns = {"fgroups", "xdata", "ydata"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    return df


def dataframe_to_arrays(df):
    X = np.array(df["ydata"].tolist(), dtype=np.float32)

    y = np.array([
        [int(row.get(label, False)) for label in LABEL_NAMES]
        for row in df["fgroups"]
    ], dtype=np.int32)

    return X, y


def main():
    processed_dir = Path("data/processed")

    chemotion_path = processed_dir / "spectra-chemotion.pkl"
    nist_path = processed_dir / "spectra-nist.pkl"

    chemotion = load_pickle(chemotion_path)
    nist = load_pickle(nist_path)

    print("Chemotion:", chemotion.shape)
    print("NIST:", nist.shape)

    combined = pd.concat([chemotion, nist], ignore_index=True)
    print("Combined:", combined.shape)

    X, y = dataframe_to_arrays(combined)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X min:", X.min())
    print("X max:", X.max())
    print("y unique:", np.unique(y))

    np.save(processed_dir / "X_spectra.npy", X)
    np.save(processed_dir / "y_labels.npy", y)

    with open(processed_dir / "label_names.txt", "w", encoding="utf-8") as f:
        for label in LABEL_NAMES:
            f.write(label + "\n")

    print("Saved:")
    print(processed_dir / "X_spectra.npy")
    print(processed_dir / "y_labels.npy")
    print(processed_dir / "label_names.txt")


if __name__ == "__main__":
    main()
