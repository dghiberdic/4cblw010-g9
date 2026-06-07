import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pybaselines import Baseline
from rdkit import Chem
import pandas as pd

SMARTS_fgroups = {
    "ester":           Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "alkane":          Chem.MolFromSmarts("[CX4;H3,H2]"),
    "alkene":          Chem.MolFromSmarts("[CX3]=[CX3]"),
    "alcohol":         Chem.MolFromSmarts("[#6][OX2H]"),
    "arene":           Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1"),
    "amine":           Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]"),
    "ketone":          Chem.MolFromSmarts("[#6][CX3](=O)[#6]"),
    "ether":           Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "imine":           Chem.MolFromSmarts("[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]"),
    "sulfonamide":     Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]"),
    "acyl_halide":     Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]"),
    "phosphate":       Chem.MolFromSmarts("[#15X4](=[OX1])([OX2])[OX2]"),
    "aldehyde":        Chem.MolFromSmarts("[CX3H1](=O)[#6,H]"),
    "nitro":           Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]"),
    "enamine":         Chem.MolFromSmarts("[NX3][CX3]=[CX3]"),
    "azo":             Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]"),
    "sulfonic_acid":   Chem.MolFromSmarts("[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]"),
    "amide":           Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]"),
    "peroxide":        Chem.MolFromSmarts("[OX2,OX1-][OX2,OX1-]"),
}

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))

def format(raw_data: dict, mdata: dict, identifier: str, INTERVAL: tuple) -> dict:
    """
    Goal:
    - Extract spectrum data from JCAMP-DX files
    - Ensure x is strictly increasing
    """    
    if raw_data["xunits"] == "MICROMETERS":
        raw_data["firstx"] = 1/(raw_data["firstx"]*1e-4)
        raw_data["lastx"] = 1/(raw_data["lastx"]*1e-4)

    xinterval = (raw_data['firstx'], raw_data['lastx'])
    if xinterval[1]-xinterval[0] < 0:
            xinterval = xinterval[::-1]
            raw_data['y'] = raw_data['y'][::-1]
    
    spectrum_overlap = overlap(INTERVAL[0],INTERVAL[1],xinterval[0],xinterval[1])/(INTERVAL[1]-INTERVAL[0])
    if spectrum_overlap < 0.9:
        raise AssertionError("Spectrum doesn't overlap enough")

    spectrum_data = {
        "id": identifier,
        "molecule": mdata[identifier],
        "fgroups": {},
        "xdata": np.linspace(xinterval[0],xinterval[1],raw_data['y'].size),
        "ydata": raw_data['y']
    }
    return spectrum_data

def unit_correct(raw_data: dict, spectrum_data: dict):
    match raw_data["yunits"]:
        case "TRANSMITTANCE":
            return -np.log10(np.clip(spectrum_data["ydata"], 1e-6, None))
        case "ABSORBANCE":
            return spectrum_data["ydata"]
        case "(micromol/mol)-1m-1 (base 10)":
            raise Exception("Micromol x-unit found")
        case "Reflectance" if "diffuse" in raw_data["ylabel"].lower():
            spectrum_data["ydata"] = spectrum_data["ydata"] * raw_data["yfactor"]
            spectrum_data["ydata"] = np.clip(spectrum_data["ydata"], 1e-6, 1.0)
            spectrum_data["ydata"] = (1 - spectrum_data["ydata"])**2 / (2 * spectrum_data["ydata"])
        case "Reflectance" if "hemispherical" in raw_data["ylabel"].lower():
            spectrum_data["ydata"] = spectrum_data["ydata"] * raw_data["yfactor"]
            spectrum_data['ydata'] = -np.log10(np.clip(1 - spectrum_data["ydata"], 1e-6, None))
        case "dispersion index" | "absorption index":
            return None
        case _:
            raise Exception("No match found")

def interpolate(spectrum_data: dict, INTERVAL: tuple) -> list[np.ndarray, np.ndarray]:
    """
    Goal:
    - Cubic spine interpolation to 2cm(-1) intervals
    """
    xs = np.arange(INTERVAL[0],INTERVAL[1], 2)
    return xs, np.interp(xs, spectrum_data['xdata'], spectrum_data['ydata'])

def fit(spectrum_data: dict) -> np.ndarray:
    """
    Goal:
    - Apply a baseline correction to the spectra
    """
    y = spectrum_data['ydata']
    fitter = Baseline(x_data=spectrum_data['xdata'])
    y_corr, params = fitter.asls(y, lam=1e5, p=0.01)
    return y - y_corr

def smooth(spectrum_data: dict) -> np.ndarray:
    return savgol_filter(spectrum_data['ydata'], window_length=5, polyorder=3)

def normalize(spectrum_data: dict) -> list[np.ndarray, np.ndarray]:
    """
    Goal:
    - Normalize to [0, 1] using Min-Max normalization
    """
    y = spectrum_data['ydata']
    y_norm = (y - y.min())/(y.max() - y.min())
    return y_norm

def label(spectrum_data: dict, smiles=False, inchi=False) -> list[np.ndarray]:
    """
    Goal:
    - Assign functional groups using SMARTS patterns
    - Automatically assign said functional groups to compounds using RDKit
    """
    if smiles:
        mol = Chem.MolFromSmiles(spectrum_data["molecule"])
    elif inchi:
        mol = Chem.MolFromInchi(spectrum_data["molecule"])
    else:
        raise Exception("Either smiles or inchi must equal True")
    if mol is None:
        return 1
    fgroups = dict([(key, mol.HasSubstructMatch(val)) for key, val  in SMARTS_fgroups.items()])
    return fgroups

def plot_spectrum(smiles: str, x: np.ndarray, y: np.ndarray):
    plt.plot(x, y)
    plt.title(smiles)
    plt.xlabel("Wavenumber (1/cm)")
    plt.ylabel("Normalized Absorbance")
    plt.gca().invert_xaxis()  # IR spectra conventionally go right-to-left
    plt.show()
