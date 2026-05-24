from jcamp import jcamp_read
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import csv
import os

"""
preprocessing pipeline
1. Parsing
2. Formatting
3. Range
4. Interpolation
5. Normalization
6. Labelling

Relevant CSV headers


References:
- DOI: 10.1021/acsomega.6c01193
"""


def parse():
    """
    Goals:
    - filter out gas phase
    """


def format(spectrum_data):
    """
    Goal:
    - Ensure x is strictly increasing
    """
    x, y = spectrum_data['x'], spectrum_data['y']
    sort_ids = np.argsort(x)
    print(x)
    return x[sort_ids], y[sort_ids]


def interpolate(spectrum_data):
    """
    Goal:
    - Cubic spine interpolation to 2cm(-1) intervals
    """
    cs = CubicSpline(spectrum_data['x'], spectrum_data['y'])
    xs = np.arange(400, 4000, 2)
    ys = cs(xs)
    return xs, ys


def normalize(spectrum_data):
    """
    Goal:
    - Normalize to [0, 1] using Min-Max normalization
    """
    y = spectrum_data['y']
    y_norm = (y - data['miny'])/(data['maxy'] - data['miny'])
    return y_norm


def label():
    """
    Goal:
    - Assign functional groups using SMARTS patterns
    - Automatically assign said functional groups to compounds using RDKit
    """
    pass


def plot_spectrum(title: str, x: np.ndarray, y: np.ndarray):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Wavenumber (1/cm)")
    plt.ylabel("Transmittance")
    plt.gca().invert_xaxis()  # IR spectra conventionally go right-to-left
    plt.show()


if __name__ == "__main__":
    dir = r"data-preprocessing-pipeline\IR_data\exp"
    csvfile = open(r"data-preprocessing-pipeline\spectra-chemotion.csv", "w")
    for entry in os.scandir(dir):
        jcampfile = open(entry, "r")
        print(entry.path)
        data = jcamp_read(jcampfile)
        print(data)
        print("\n")

        data['x'], data['y'] = format(data)
        data['x'], data['y'] = interpolate(data)
        data['y'] = normalize(data)
        jcampfile.close()

    
    csvfile.close()

    plot_spectrum(data['title'],data['x'],data['y'])
    

