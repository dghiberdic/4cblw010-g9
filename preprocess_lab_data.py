"""
Preprocess lab FTIR plastic samples from FTIR_PLASTIC_c8.csv into
model-ready numpy arrays.

Follows the same pipeline as data_preproccesing_lib.py:
    unit_correct → interpolate → smooth → fit → normalize

Usage:
    python preprocess_lab_data.py FTIR_PLASTIC_c8.csv output_dir/

Outputs:
    X_lab.npy        shape (n_samples, 1650)
    meta_lab.csv     sample ID, polymer type, sample number
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from pybaselines import Baseline


INTERVAL = (600, 3900)
MODEL_GRID = np.arange(INTERVAL[0], INTERVAL[1], 2)   # 1650 points


def unit_correct(y_pct):
    # Lab data is %Transmittance (0-100). Convert to fraction then to absorbance.
    return -np.log10(np.clip(y_pct / 100.0, 1e-6, None))


def interpolate(x_raw, y_raw):
    return np.interp(MODEL_GRID, x_raw, y_raw)


def smooth(y):
    return savgol_filter(y, window_length=5, polyorder=3)


def fit(y):
    fitter = Baseline(x_data=MODEL_GRID)
    y_corr, _ = fitter.asls(y, lam=1e5, p=0.01)
    return y - y_corr


def normalize(y):
    std = y.std()
    if std < 1e-10:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - y.mean()) / std).astype(np.float32)


def preprocess_row(row):
    xy    = row.iloc[6:].values
    x_raw = xy[0::2].astype(float)
    y_raw = xy[1::2].astype(float)

    y = unit_correct(y_raw)
    y = interpolate(x_raw, y)
    y = smooth(y)
    y = fit(y)
    y = normalize(y)
    return y


def main(csv_path, out_dir):
    csv_path = Path(csv_path)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} spectra  |  polymers: {sorted(df['Polymer'].unique())}")

    spectra = []
    for i, (_, row) in enumerate(df.iterrows()):
        spectra.append(preprocess_row(row))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")

    X    = np.array(spectra)
    meta = df[["IDE", "Polymer", "Sample"]].reset_index(drop=True)

    np.save(out_dir / "X_lab.npy", X)
    meta.to_csv(out_dir / "meta_lab.csv", index=False)

    print(f"Done.")
    print(f"  X_lab.npy    {X.shape}  min={X.min():.3f}  max={X.max():.3f}")
    print(f"  meta_lab.csv {len(meta)} rows")
    print(f"  → {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_lab_data.py FTIR_PLASTIC_c8.csv output_dir/")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
