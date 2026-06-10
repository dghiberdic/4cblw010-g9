"""
Preprocess lab FTIR JCAMP-DX measurements (Shimadzu IRSpirit) into
model-ready numpy arrays.

Input: a directory tree of .dx files (e.g. the unzipped "Measurements ..."
folders). The substance is inferred from the file name. Spectra are
%Transmittance, 400-4000 cm-1; atmosphere correction was already applied
on the instrument.

Follows the same pipeline as data_preproccesing_lib.py:
    unit_correct -> interpolate -> smooth -> fit -> normalize (SNV)

Usage:
    python preprocess_lab_jcamp.py ../lab_raw data/lab1/

Outputs:
    X_lab.npy        shape (n_samples, 1650), float32, SNV-normalized
    meta_lab.csv     File, Polymer
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from jcamp import jcamp_read
from scipy.signal import savgol_filter
from pybaselines import Baseline


INTERVAL = (600, 3900)
MODEL_GRID = np.arange(INTERVAL[0], INTERVAL[1], 2)   # 1650 points

# File-name prefix -> substance. The meta column is named "Polymer" for
# compatibility with lab_data_testing.ipynb even though TPA/EG are small
# molecules.
NAME_MAP = [
    (re.compile(r"^PET", re.I),            "PET"),
    (re.compile(r"^TPA", re.I),            "TPA"),
    (re.compile(r"^Ethyleneglycol", re.I), "EG"),
]


def unit_correct(y_pct):
    # %Transmittance (values 10-108, slightly >100 in baseline regions is
    # normal) -> absorbance.
    return -np.log10(np.clip(y_pct / 100.0, 1e-6, None))


def preprocess(x_raw, y_raw):
    order = np.argsort(x_raw)
    x_raw, y_raw = x_raw[order], y_raw[order]

    y = unit_correct(y_raw)
    y = np.interp(MODEL_GRID, x_raw, y)
    y = savgol_filter(y, window_length=31, polyorder=3)
    baseline, _ = Baseline(x_data=MODEL_GRID).airpls(y, lam=1e6)
    y = y - baseline

    std = y.std()
    if std < 1e-10:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - y.mean()) / std).astype(np.float32)


def substance_of(filename):
    for pat, name in NAME_MAP:
        if pat.match(filename):
            return name
    return None


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, spectra = [], []
    for f in sorted(in_dir.rglob("*.dx")):
        substance = substance_of(f.name)
        if substance is None:
            print(f"  skipped (unrecognized name): {f.name}")
            continue

        with open(f) as fh:
            d = jcamp_read(fh)
        x = d["x"]
        y = d["y"]

        coverage = (min(x.max(), INTERVAL[1]) - max(x.min(), INTERVAL[0])) \
                   / (INTERVAL[1] - INTERVAL[0])
        if coverage < 0.9:
            print(f"  skipped (coverage {coverage:.2f}): {f.name}")
            continue

        spectra.append(preprocess(x, y))
        rows.append({"File": f.name, "Polymer": substance})
        print(f"  {substance:4s} <- {f.name}")

    X = np.array(spectra)
    meta = pd.DataFrame(rows)

    np.save(out_dir / "X_lab.npy", X)
    meta.to_csv(out_dir / "meta_lab.csv", index=False)

    print(f"\nDone.")
    print(f"  X_lab.npy    {X.shape}  min={X.min():.3f}  max={X.max():.3f}")
    print(f"  per substance: {meta['Polymer'].value_counts().to_dict()}")
    print(f"  -> {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_lab_jcamp.py <jcamp_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
