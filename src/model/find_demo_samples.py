"""
find_demo_samples.py

Find useful example samples for the PET Hydrolysis FTIR Screener.

Looks for:
1. Strong hydrolysis-product-like samples
2. Possible PET-like samples
3. Uncertain aromatic samples
"""

from pathlib import Path
import numpy as np
import joblib


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


def main():
    model_path = Path("outputs/models/random_forest_with_augmentation.joblib")
    X_path = Path("data/processed/X_spectra.npy")

    model = joblib.load(model_path)
    X = np.load(X_path)

    predictions = model.predict(X)

    ester_i = LABEL_NAMES.index("ester")
    acid_i = LABEL_NAMES.index("carboxylic_acid")
    alcohol_i = LABEL_NAMES.index("alcohol")
    arene_i = LABEL_NAMES.index("arene")

    strong_hydrolysis = []
    possible_hydrolysis = []
    pet_like = []
    uncertain_aromatic = []

    for idx, pred in enumerate(predictions):
        ester = pred[ester_i]
        acid = pred[acid_i]
        alcohol = pred[alcohol_i]
        arene = pred[arene_i]

        if acid == 1 and alcohol == 1:
            strong_hydrolysis.append(idx)
        elif acid == 1 or alcohol == 1:
            possible_hydrolysis.append(idx)
        elif ester == 1 and arene == 1 and acid == 0 and alcohol == 0:
            pet_like.append(idx)
        elif arene == 1 and ester == 0 and acid == 0 and alcohol == 0:
            uncertain_aromatic.append(idx)

    print("Strong hydrolysis-product-like samples:", strong_hydrolysis[:20])
    print("Possible hydrolysis-product-like samples:", possible_hydrolysis[:20])
    print("PET-like samples:", pet_like[:20])
    print("Uncertain aromatic samples:", uncertain_aromatic[:20])

    print()
    print("Counts:")
    print("Strong hydrolysis-product-like:", len(strong_hydrolysis))
    print("Possible hydrolysis-product-like:", len(possible_hydrolysis))
    print("PET-like:", len(pet_like))
    print("Uncertain aromatic:", len(uncertain_aromatic))


if __name__ == "__main__":
    main()
