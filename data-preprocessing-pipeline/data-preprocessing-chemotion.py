from jcamp import jcamp_read
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import csv
import os
import json
from pybaselines import Baseline, utils

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
    pass


def format(raw_data: dict, mdata: dict, identifier: str) -> dict:
    """
    Goal:
    - Extract spectrum data from JCAMP-DX version 4.24 and 5.0 files
    - Ensure x is strictly increasing
    """
    if raw_data['jcamp-dx'] == 5.0:
        raw_data = raw_data['children'][0]
    elif raw_data['jcamp-dx'] != 4.24:
        return 1
    
    xinterval = (raw_data['firstx'], raw_data['lastx'])
    spectrum_data = {
        "id": identifier,
        "smiles": mdata[id],
        "xdata": np.linspace(min(xinterval),max(xinterval),raw_data['y'].size),
        "ydata": raw_data['y'] if xinterval[1]-xinterval[0] > 0 else raw_data['y'][::-1]
    }
    return spectrum_data


def interpolate(spectrum_data: dict) -> list[np.ndarray, np.ndarray]:
    """
    Goal:
    - Cubic spine interpolation to 2cm(-1) intervals
    """
    cs = CubicSpline(spectrum_data['xdata'], spectrum_data['ydata'])
    xs = np.arange(400, 4000, 2)
    ys = cs(xs)
    return xs, ys

def fit(spectrum_data: dict) -> np.ndarray:
    """
    Goal:
    - Apply a baseline correction to the spectra
    """
    y = spectrum_data['ydata']
    fitter = Baseline(x_data=spectrum_data['xdata'])
    y_corr, params = fitter.modpoly(y, poly_order=3)
    return y - y_corr

def normalize(spectrum_data: dict) -> list[np.ndarray, np.ndarray]:
    """
    Goal:
    - Normalize to [0, 1] using Min-Max normalization
    """
    y = spectrum_data['ydata']
    y_norm = (y - y.min())/(y.max() - y.min())
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
    # Metadata import
    jsonfile = open(r"data-preprocessing-pipeline\IR_data-chemotion\meta_data.json", 'r', encoding="utf-8")
    jsondata = json.load(jsonfile)
    metadata = {}
    for record in jsondata:
        for dataset in record["datasets"]:
            for attachment in dataset["attacments"]:
                id = attachment["identifier"].split('/')[1]
                smiles = record["cano_smiles"]
                metadata[id] = smiles
    print(metadata)
    jsonfile.close()
    
    skipped_files = []
    dir = r"data-preprocessing-pipeline\IR_data-chemotion\exp"
    csvfile = open(r"data-preprocessing-pipeline\spectra-chemotion.csv", "w")
    for entry in os.scandir(dir):
        if str(entry.name)[0] == '1':
            break

        # File reading
        print(entry.path)
        jcampfile = open(entry, "r")
        try:
            raw_data = jcamp_read(jcampfile)
        except Exception as exception: # Skips a single file that uses commas instead of dots
            skipped_files.append((entry.path, exception)) 
        jcampfile.close()

        # Data preprocessing
        data = format(raw_data, metadata, entry.name)
        if data == 1:
            skipped_files.append((entry.path, f"version not a float, \"{raw_data['jcamp-dx']}\""))
            continue
            # raise Exception(f"JCAMP-DX version not supported, version {raw_data['jcamp-dx']} found")
        data['xdata'], data['ydata'] = interpolate(data)
        #data['ydata'] = fit(data)
        data['ydata'] = normalize(data)

    csvfile.close()
    for i, file in enumerate(skipped_files):
        print(f"{i+1} file:{file[0]}\nreason: {file[1]}\n")

    plot_spectrum(data['smiles'],data['xdata'],data['ydata'])
    print(data)