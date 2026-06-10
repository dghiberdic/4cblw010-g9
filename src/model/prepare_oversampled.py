"""
prepare_oversampled.py

Reads data/processed/X_spectra.npy and y_labels.npy, then:
  1. Drops NaN rows.
  2. Drops 8 unlearnable label columns (too few positives to train on).
  3. Oversamples carboxylic_acid positives 3x (adds 3 copies → 4x total)
     to lift it from ~5% to ~18% of the dataset.
  4. Shuffles and saves to data/oversampled/.

Output files use the same format as data/processed/ and are ready to upload
directly to the Colab CNN notebook.
"""

from pathlib import Path
import numpy as np

LABEL_NAMES = [
    "ester", "carboxylic_acid", "alkane", "alkene", "alcohol",
    "arene", "amine", "ketone", "ether", "imine", "sulfonamide",
    "acyl_halide", "phosphate", "aldehyde", "nitro", "enamine",
    "azo", "sulfonic_acid", "amide", "peroxide",
]

DROP_LABELS = {
    "imine", "sulfonamide", "acyl_halide", "phosphate",
    "aldehyde", "enamine", "azo", "peroxide",
}

KEEP_IDX   = [i for i, n in enumerate(LABEL_NAMES) if n not in DROP_LABELS]
KEEP_NAMES = [LABEL_NAMES[i] for i in KEEP_IDX]

CARBOXYLIC_COPIES = 3


def main():
    out_dir = Path("data/oversampled")
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load("data/processed/X_spectra.npy")
    y = np.load("data/processed/y_labels.npy")
    print(f"Loaded:  X={X.shape}  y={y.shape}")

    # 1. Drop NaN rows
    nan_mask = np.isnan(X).any(axis=1)
    X, y = X[~nan_mask], y[~nan_mask]
    print(f"After NaN drop ({nan_mask.sum()} rows):  X={X.shape}")

    # 2. Drop 8 unlearnable label columns
    y = y[:, KEEP_IDX]
    print(f"After label drop:  y={y.shape}  labels={KEEP_NAMES}")

    # 3. Oversample carboxylic_acid positives
    ca_col   = KEEP_NAMES.index("carboxylic_acid")
    ca_mask  = y[:, ca_col] == 1
    ca_count = int(ca_mask.sum())
    print(f"\nCarboxylic acid before: {ca_count} / {len(y)}  ({100*ca_count/len(y):.1f}%)")

    X_extra = np.repeat(X[ca_mask], CARBOXYLIC_COPIES, axis=0)
    y_extra = np.repeat(y[ca_mask], CARBOXYLIC_COPIES, axis=0)

    X_out = np.vstack([X, X_extra])
    y_out = np.vstack([y, y_extra])

    # 4. Shuffle so oversampled rows are not all at the end
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X_out))
    X_out, y_out = X_out[perm], y_out[perm]

    ca_final = int(y_out[:, ca_col].sum())
    print(f"Carboxylic acid after:  {ca_final} / {len(y_out)}  ({100*ca_final/len(y_out):.1f}%)")
    print(f"\nFinal dataset:  X={X_out.shape}  y={y_out.shape}")

    np.save(out_dir / "X_spectra.npy", X_out)
    np.save(out_dir / "y_labels.npy",  y_out)
    with open(out_dir / "label_names.txt", "w") as f:
        f.write("\n".join(KEEP_NAMES))

    print(f"\nSaved to {out_dir}/")
    print("  X_spectra.npy   y_labels.npy   label_names.txt")
    print("\nLabel distribution in oversampled set:")
    for i, name in enumerate(KEEP_NAMES):
        cnt = int(y_out[:, i].sum())
        print(f"  {name:20s}: {cnt:5d}  ({100*cnt/len(y_out):.1f}%)")


if __name__ == "__main__":
    main()
