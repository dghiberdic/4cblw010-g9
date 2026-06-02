"""
screen_sample.py

PET Hydrolysis FTIR Screener

This script loads a trained Random Forest model and applies it to one FTIR
spectrum from the processed dataset. It predicts functional-group presence,
shows confidence scores where available, and translates the result into a
PET hydrolysis screening interpretation.

This is a proof-of-concept screening tool, not a final industrial analyser.
"""

from pathlib import Path
import argparse
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


CORE_HYDROLYSIS_GROUPS = {
    "ester": "PET starting-material marker",
    "carboxylic_acid": "TPA-related hydrolysis-product marker",
    "alcohol": "ethylene-glycol-related hydrolysis-product marker",
    "arene": "aromatic PET/TPA structural marker",
}


def load_model(model_path: str):
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Run first: python -m src.model.train_models"
        )

    return joblib.load(path)


def load_sample(sample_index: int):
    x_path = Path("data/processed/X_spectra.npy")

    if not x_path.exists():
        raise FileNotFoundError("Could not find data/processed/X_spectra.npy")

    X = np.load(x_path)

    if sample_index < 0 or sample_index >= len(X):
        raise IndexError(f"sample_index must be between 0 and {len(X) - 1}")

    return X[sample_index].reshape(1, -1), len(X)


def get_prediction_and_confidence(model, sample):
    """
    Returns:
    - binary prediction for each functional group
    - confidence/probability for each functional group if model supports predict_proba
    """

    prediction = model.predict(sample)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)

        confidence_values = []

        for label_prob in probabilities:
            # For RandomForest multi-output, each label has its own probability array.
            # If both classes are present, class 1 probability is at index 1.
            if label_prob.shape[1] == 2:
                confidence_values.append(float(label_prob[0][1]))
            else:
                # If only one class was seen during training for a label,
                # use the predicted binary output as fallback confidence.
                confidence_values.append(float(prediction[len(confidence_values)]))

        confidence = np.array(confidence_values)

    return prediction, confidence


def interpret_pet_hydrolysis(prediction, confidence=None):
    pred_dict = dict(zip(LABEL_NAMES, prediction))

    ester = pred_dict.get("ester", 0)
    carboxylic = pred_dict.get("carboxylic_acid", 0)
    alcohol = pred_dict.get("alcohol", 0)
    arene = pred_dict.get("arene", 0)

    if confidence is not None:
        conf_dict = dict(zip(LABEL_NAMES, confidence))
        ester_c = conf_dict.get("ester", 0)
        carboxylic_c = conf_dict.get("carboxylic_acid", 0)
        alcohol_c = conf_dict.get("alcohol", 0)
        arene_c = conf_dict.get("arene", 0)
    else:
        ester_c = carboxylic_c = alcohol_c = arene_c = None

    reasons = []

    if ester == 1:
        reasons.append("ester signal detected, which is linked to PET starting material")
    if carboxylic == 1:
        reasons.append("carboxylic acid signal detected, which is linked to TPA-like hydrolysis products")
    if alcohol == 1:
        reasons.append("alcohol/hydroxyl signal detected, which is linked to ethylene glycol-like products")
    if arene == 1:
        reasons.append("arene signal detected, which is consistent with PET/TPA aromatic structure")

    if carboxylic == 1 or alcohol == 1:
        result = "Hydrolysis-product-like"
    elif ester == 1 and carboxylic == 0 and alcohol == 0:
        result = "PET-like"
    elif arene == 1 and (ester == 1 or carboxylic == 1):
        result = "PET/TPA-related but uncertain"
    else:
        result = "Uncertain"

    if not reasons:
        reasons.append("no clear hydrolysis-related functional-group pattern was detected")

    confidence_note = ""
    if confidence is not None:
        confidence_note = (
            f"Core confidence scores: ester={ester_c:.2f}, "
            f"carboxylic_acid={carboxylic_c:.2f}, "
            f"alcohol={alcohol_c:.2f}, arene={arene_c:.2f}"
        )

    return result, reasons, confidence_note


def format_report(sample_index, total_samples, prediction, confidence, result, reasons, confidence_note):
    lines = []

    lines.append("=" * 72)
    lines.append("PET Hydrolysis FTIR Screener")
    lines.append("=" * 72)
    lines.append(f"Sample index: {sample_index} of {total_samples - 1}")
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "Qualitative screening of FTIR spectra for functional-group patterns "
        "related to PET hydrolysis."
    )
    lines.append("")
    lines.append("Core hydrolysis-related functional groups:")

    for group, meaning in CORE_HYDROLYSIS_GROUPS.items():
        idx = LABEL_NAMES.index(group)
        status = "present" if prediction[idx] == 1 else "absent"

        if confidence is not None:
            lines.append(f"- {group}: {status} | confidence={confidence[idx]:.2f} | {meaning}")
        else:
            lines.append(f"- {group}: {status} | {meaning}")

    lines.append("")
    lines.append("All predicted functional groups:")

    for idx, label in enumerate(LABEL_NAMES):
        status = "present" if prediction[idx] == 1 else "absent"

        if confidence is not None:
            lines.append(f"- {label}: {status} | confidence={confidence[idx]:.2f}")
        else:
            lines.append(f"- {label}: {status}")

    lines.append("")
    lines.append("Screening result:")
    lines.append(result)

    lines.append("")
    lines.append("Reasoning:")
    for reason in reasons:
        lines.append(f"- {reason}")

    if confidence_note:
        lines.append("")
        lines.append(confidence_note)

    lines.append("")
    lines.append("Important limitation:")
    lines.append(
        "This is a qualitative screening result, not a final yield, purity, or "
        "concentration measurement. It must be validated with measured PET, TPA, "
        "ethylene glycol, and PET hydrolysis-mixture spectra."
    )

    lines.append("")
    lines.append("Sustainability relevance:")
    lines.append(
        "The tool supports PET chemical recycling by giving a fast first-pass "
        "screening of whether hydrolysis-related functional-group changes may be "
        "visible in an FTIR spectrum."
    )

    return "\n".join(lines)


def save_report(report_text, sample_index):
    output_dir = Path("outputs/screening_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"sample_{sample_index}_screening_report.txt"
    output_path.write_text(report_text, encoding="utf-8")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Screen one FTIR sample for PET hydrolysis-related functional groups.")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to screen.")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/models/random_forest_baseline_no_augmentation.joblib",
        help="Path to trained model."
    )

    args = parser.parse_args()

    model = load_model(args.model)
    sample, total_samples = load_sample(args.sample)

    prediction, confidence = get_prediction_and_confidence(model, sample)
    result, reasons, confidence_note = interpret_pet_hydrolysis(prediction, confidence)

    report_text = format_report(
        sample_index=args.sample,
        total_samples=total_samples,
        prediction=prediction,
        confidence=confidence,
        result=result,
        reasons=reasons,
        confidence_note=confidence_note,
    )

    print(report_text)

    output_path = save_report(report_text, args.sample)
    print()
    print(f"Saved screening report to: {output_path}")


if __name__ == "__main__":
    main()
