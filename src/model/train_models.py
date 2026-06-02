"""
train_models.py

Complete ML module for FTIR functional-group prediction.

This script:
1. Loads processed FTIR spectra and functional-group labels.
2. Trains a Random Forest baseline without augmentation.
3. Trains a Random Forest model with data augmentation.
4. Evaluates both models using F1-scores and per-label reports.
5. Saves models and result files.

Input:
data/processed/X_spectra.npy
data/processed/y_labels.npy

Output:
outputs/models/
outputs/model_results/
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    multilabel_confusion_matrix,
)

from src.data.augmentation import (
    vertical_noise,
    baseline_shift,
    baseline_slope,
    multiplicative_scaling,
    smoothing,
    horizontal_shift,
)

# The order of these labels must match the column order in y_labels.npy.
# Each model output corresponds to one of these functional groups.

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


def load_data():
    x_path = Path("data/processed/X_spectra.npy")
    y_path = Path("data/processed/y_labels.npy")

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Could not find processed data. Expected:\n"
            "data/processed/X_spectra.npy\n"
            "data/processed/y_labels.npy"
        )

    X = np.load(x_path)
    y = np.load(y_path)

    print("Loaded data")
    print("X:", X.shape)
    print("y:", y.shape)

    if y.shape[1] != len(LABEL_NAMES):
        raise ValueError(
            f"y has {y.shape[1]} labels, but LABEL_NAMES has {len(LABEL_NAMES)} names."
        )

    return X, y


def make_output_dirs():
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/model_results").mkdir(parents=True, exist_ok=True)


def label_distribution(y):
    counts = y.sum(axis=0)
    df = pd.DataFrame({
        "label": LABEL_NAMES,
        "positive_count": counts,
        "positive_rate": counts / len(y),
    })
    df.to_csv("outputs/model_results/label_distribution.csv", index=False)
    print("\nLabel distribution:")
    print(df.to_string(index=False))
    return df


def augment_training_data(X_train, y_train, copies_per_sample=1):
    """
    Apply mild augmentation only to the training data.

    Important:
    The test set is NOT augmented.
    """

    methods = [
        lambda x: vertical_noise(x, noise_std=0.003),
        lambda x: baseline_shift(x, shift_range=(-0.01, 0.01)),
        lambda x: baseline_slope(x, slope_range=(-0.01, 0.01)),
        lambda x: multiplicative_scaling(x, scale_range=(0.98, 1.02), offset_range=(-0.01, 0.01)),
        lambda x: smoothing(x, sigma_range=(0.2, 0.5)),
        lambda x: horizontal_shift(x, max_shift=3),
    ]

    X_aug = []
    y_aug = []

    rng = np.random.default_rng(42)

    for x, label in zip(X_train, y_train):
        for _ in range(copies_per_sample):
            method = rng.choice(methods)
            X_aug.append(method(x))
            y_aug.append(label)

    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    X_combined = np.vstack([X_train, X_aug])
    y_combined = np.vstack([y_train, y_aug])

    print("\nAugmentation completed")
    print("Original train X:", X_train.shape)
    print("Augmented copies:", X_aug.shape)
    print("Combined train X:", X_combined.shape)

    return X_combined, y_combined


def train_random_forest(X_train, y_train, model_name):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    model_path = Path(f"outputs/models/{model_name}.joblib")
    joblib.dump(model, model_path)

    print(f"Saved model to {model_path}")
    return model


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    samples_f1 = f1_score(y_test, y_pred, average="samples", zero_division=0)

    summary = {
        "model": model_name,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "samples_f1": float(samples_f1),
    }

    print("Micro F1:", micro_f1)
    print("Macro F1:", macro_f1)
    print("Samples F1:", samples_f1)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(f"outputs/model_results/{model_name}_classification_report.csv")

    cm = multilabel_confusion_matrix(y_test, y_pred)
    cm_rows = []

    for label_name, matrix in zip(LABEL_NAMES, cm):
        tn, fp, fn, tp = matrix.ravel()
        cm_rows.append({
            "label": label_name,
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        })

    cm_df = pd.DataFrame(cm_rows)
    cm_df.to_csv(f"outputs/model_results/{model_name}_confusion_matrices.csv", index=False)

    with open(f"outputs/model_results/{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    make_output_dirs()
    X, y = load_data()

    label_distribution(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTrain/test split")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # 1. Baseline without augmentation
    baseline_model = train_random_forest(
        X_train,
        y_train,
        model_name="random_forest_baseline_no_augmentation",
    )

    baseline_summary = evaluate_model(
        baseline_model,
        X_test,
        y_test,
        model_name="random_forest_baseline_no_augmentation",
    )

    # 2. Random Forest with augmentation
    X_train_aug, y_train_aug = augment_training_data(
        X_train,
        y_train,
        copies_per_sample=1,
    )

    augmented_model = train_random_forest(
        X_train_aug,
        y_train_aug,
        model_name="random_forest_with_augmentation",
    )

    augmented_summary = evaluate_model(
        augmented_model,
        X_test,
        y_test,
        model_name="random_forest_with_augmentation",
    )

    comparison = pd.DataFrame([baseline_summary, augmented_summary])
    comparison.to_csv("outputs/model_results/model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nDone. Results saved in outputs/model_results/")


if __name__ == "__main__":
    main()
