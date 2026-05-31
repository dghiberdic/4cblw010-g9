"""Plot examples of augmentation methods."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.augmentation import *


np.random.seed(0)

df = pd.read_pickle("data/processed/spectra-chemotion.pkl")

wavenumbers = df["xdata"].iloc[0]
spectrum = df["ydata"].iloc[0]
X_train = np.stack(df["ydata"].values)

emsa_model = fit_emsa_model(X_train)

noise_aug = vertical_noise(spectrum, noise_std=0.02)
smooth_aug = smoothing(spectrum, sigma_range=(1.2, 2.0))
shift_aug = horizontal_shift(spectrum, max_shift= 200)
emsa_aug = emsa(spectrum, emsa_model)
offset_aug = baseline_shift(spectrum, shift_range=(-0.08, 0.08))
slope_aug = baseline_slope(spectrum, slope_range=(-2, 2))
scale_aug = multiplicative_scaling( spectrum,scale_range=(0.85, 1.15), offset_range=(-0.05, 0.05),)

fig, axes = plt.subplots(4, 2, figsize=(11, 14))
axes = axes.ravel()

plots = [
    ("Vertical Noise", noise_aug, "Noised Spectrum"),
    ("Smoothing", smooth_aug, "Smoothed Spectrum"),
    ("Horizontal Shift", shift_aug, "Shifted Spectrum"),
    ("EMSA", emsa_aug, "EMSA Augmented"),
    ("Baseline Shift", offset_aug, "Offset Spectrum"),
    ("Baseline Slope", slope_aug, "Sloped Baseline Spectrum"),
    ("Multiplicative Scaling", scale_aug, "Multiplicative Augmentation"),
]

for i, (title, augmented, label) in enumerate(plots):
    axes[i].plot(wavenumbers, spectrum, label="Original Spectrum", linewidth=1.0, alpha=0.9)
    axes[i].plot(wavenumbers, augmented, label=label, linewidth=1.0, alpha=0.75)
    axes[i].set_title(title)
    axes[i].set_xlabel("Wavenumbers")
    axes[i].set_ylabel("Absorbance")
    axes[i].legend(fontsize=8)

axes[-1].axis("off")
plt.suptitle("Augmentation Techniques", fontsize=16)
plt.tight_layout(rect=(0, 0, 1, 0.98))
plt.savefig("outputs/augmentation_effects.png", dpi=200)
plt.show()
