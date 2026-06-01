"""Augmentation methods for IR spectra. 

Most methods take one spectrum as input and return one spectrum as output.

To generate several augmented copies of one spectrum and repeat its label,
use augment_with_label().

oversample() takes selected spectra and their labels, then returns only the 
repeated copies.

For EMSA, first fit the EMSA model using the training set:

    emsa_model = fit_emsa_model(X_train)
    x_aug = emsa(x, emsa_model)

"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.data.emsa import EMSA


def oversample(
    X: np.ndarray,
    y: np.ndarray,
    copies: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat all given samples.

    Args:
        X: Spectra selected for oversampling, shape (num_samples, 1800).
        y: Labels for X, (num_samples, num_classes)..
        copies: Number of times to repeat each sample.

    Returns:
        X_repeated: Repeated spectra, shape (num_samples * copies, 1800).
        y_repeated: Repeated label, shape (num_samples * copies, num_classes).
    """

    X_repeated = np.repeat(X, copies, axis=0)
    y_repeated = np.repeat(y, copies, axis=0)
    return X_repeated, y_repeated


def augment_with_label(
    x: np.ndarray,
    y: np.ndarray,
    method,
    copies: int = 1,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate multiple augmented copies of one spectrum and its label.

    Args:
        x: One spectrum, shape (1800,).
        y: Label for this spectrum, shape (num_classes,).
        method: Augmentation function to apply, such as vertical_noise.
        copies: Number of augmented spectra to generate.
        **kwargs: Extra parameters passed to the augmentation method.

    Returns:
        X_augmented: Augmented spectra, shape (copies, 1800).
        y_augmented: Repeated labels, shape (copies, num_classes).
    """
    X_augmented = np.array([method(x, **kwargs) for _ in range(copies)])
    y_augmented = np.repeat(y[None, :], copies, axis=0)
    return X_augmented, y_augmented


def horizontal_shift(
    x: np.ndarray,
    max_shift: int = 3,
) -> np.ndarray:
    """Shift one spectrum left or right along the wavenumber axis.

    Args:
        x: One spectrum, shape (1800,).
        max_shift: Maximum number of points to shift.

    Returns:
        Shifted spectrum with shape (1800,).
    """
    spectrum = x.copy()
    shift = np.random.choice( [i for i in range(-max_shift, max_shift + 1) if i != 0])
    grid = np.arange(spectrum.size)
    return np.interp(grid, grid - shift, spectrum, left=spectrum[0], right=spectrum[-1])


def vertical_noise(
    x: np.ndarray,
    noise_std: float = 0.005,
) -> np.ndarray:
    """Add random noise to one spectrum's intensity values.

    Args:
        x: One spectrum, shape (1800,).
        noise_std: Strength of the added Gaussian noise.

    Returns:
        Noisy spectrum clipped to the range [0, 1].
    """
    spectrum = x.copy()
    augmented = spectrum + np.random.normal(0.0, noise_std, size=spectrum.shape)
    return np.clip(augmented, 0.0, 1.0)


def baseline_shift(
    x: np.ndarray,
    shift_range: tuple[float, float] = (-0.02, 0.02),
) -> np.ndarray:
    """Move one whole spectrum slightly up or down.

    Args:
        x: One spectrum, shape (1800,).
        shift_range: Minimum and maximum baseline offset.

    Returns:
        Baseline-shifted spectrum clipped to the range [0, 1].
    """
    spectrum = x.copy()
    offset = np.random.uniform(shift_range[0], shift_range[1])
    return np.clip(spectrum + offset, 0.0, 1.0)


def baseline_slope(
    x: np.ndarray,
    slope_range: tuple[float, float] = (-0.03, 0.03),
) -> np.ndarray:
    """Add a small sloped baseline to one spectrum.

    Args:
        x: One spectrum, shape (1800,).
        slope_range: Minimum and maximum slope strength.

    Returns:
        Spectrum with a linear baseline trend, clipped to [0, 1].
    """
    spectrum = x.copy()
    axis = np.linspace(-0.5, 0.5, spectrum.size)
    slope = np.random.uniform(slope_range[0], slope_range[1])
    return np.clip(spectrum + slope * axis, 0.0, 1.0)


def multiplicative_scaling(
    x: np.ndarray,
    scale_range: tuple[float, float] = (0.95, 1.05),
    offset_range: tuple[float, float] = (-0.02, 0.02),
) -> np.ndarray:
    """Scale one spectrum and add a small offset: M * spectrum + C.

    Args:
        x: One spectrum, shape (1800,).
        scale_range: Range for the multiplier M.
        offset_range: Range for the offset C.

    Returns:
        Scaled spectrum clipped to the range [0, 1].
    """
    spectrum = x.copy()
    scale = np.random.uniform(scale_range[0], scale_range[1])
    offset = np.random.uniform(offset_range[0], offset_range[1])
    return np.clip(scale * spectrum + offset, 0.0, 1.0)

def smoothing(
    x: np.ndarray,
    sigma_range: tuple[float, float] = (0.3, 0.8),
) -> np.ndarray:
    """Smooth one spectrum with a small Gaussian filter.

    Args:
        x: One spectrum, shape (1800,).
        sigma_range: Range for the smoothing strength.

    Returns:
        Smoothed spectrum with shape (1800,).
    """
    spectrum = x.copy()
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    smoothed = gaussian_filter1d(spectrum, sigma=sigma, mode="nearest")
    return np.clip(smoothed, 0.0, 1.0)



def emsa(
    x: np.ndarray,
    emsa_model,
) -> np.ndarray:
    """Apply the EMSC-based EMSA model to one spectrum.

    Args:
        x: One spectrum, shape (1800,).
        emsa_model: A fitted EMSA object created by fit_emsa_model().

    Returns:
        One EMSA-augmented spectrum with shape (1800,).
    """
    augmented = emsa_model._EMSA__batch_transform(np.array([x]))[0]
    return np.clip(augmented, 0.0, 1.0)


def fit_emsa_model(
    X_train: np.ndarray,
    order: int = 2,
    strength: float = 0.01,
) -> EMSA:
    """Fit an EMSC-based EMSA model using the whole training set.

    Args:
        X_train: Training spectra, shape (num_train_samples, 1800).
        wavenumbers: Common x-axis, shape (1800,). If not provided, the
            project default np.arange(400, 4000, 2) is used.
        order: Polynomial order used by the EMSC model.

    Returns:
        An EMSA model with reference spectrum and std_of_params fixed from
        X_train.
    """
    wavenumbers = np.arange(400, 4000, 2)

    reference = X_train.mean(axis=0)
    emsa_model = EMSA(
        std_of_params=None,
        wavenumbers=wavenumbers,
        reference=reference,
        order=order,
    )
    coefs = np.dot(emsa_model.A, X_train.T)
    emsa_model.std_of_params = coefs.std(axis=1) * strength
    return emsa_model


__all__ = [
    "augment_with_label",
    "baseline_shift",
    "baseline_slope",
    "emsa",
    "fit_emsa_model",
    "horizontal_shift",
    "multiplicative_scaling",
    "oversample",
    "smoothing",
    "vertical_noise",
]
