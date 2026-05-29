from jcamp import jcamp_read
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
import json
from pybaselines import Baseline, utils
from rdkit import Chem
import pandas as pd
from data_preprocessing_chemotion import plot_spectrum, normalize, interpolate

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

def format(raw_data: dict, mdata: dict, identifier: str) -> dict:
    """
    Goal:
    - Extract spectrum data from JCAMP-DX version 4.24 and 5.0 files
    - Ensure x is strictly increasing
    """    
    xinterval = (raw_data['firstx'], raw_data['lastx'])
    spectrum_data = {
        "id": identifier,
        "molecule": mdata[identifier],
        "fgroups": {},
        "xdata": np.linspace(min(xinterval),max(xinterval),raw_data['y'].size),
        "ydata": raw_data['y'] if xinterval[1]-xinterval[0] > 0 else raw_data['y'][::-1]
    }
    return spectrum_data

def label(spectrum_data: dict) -> list[np.ndarray]:
    """
    Goal:
    - Assign functional groups using SMARTS patterns
    - Automatically assign said functional groups to compounds using RDKit
    """
    mol = Chem.MolFromInchi(spectrum_data["molecule"])
    if mol is None:
        return 1
    fgroups = dict([(key, mol.HasSubstructMatch(val)) for key, val  in SMARTS_fgroups.items()])
    return fgroups

if __name__ == "__main__":
    # Metadata import
    metadata = pd.read_csv(r"data\raw\nist_ir_info.csv", dtype=str)[["cID", "inchi", "state"]]
    metadata_state = metadata.set_index("cID")["state"].to_dict()
    metadata_inchi = metadata.set_index("cID")["inchi"].to_dict()
    
    i = 0
    unknown_states = 0
    accumulated_data = []
    skipped_files = []
    dir = r"data\IR"
    entries = len(os.listdir(dir))
    for entry in os.scandir(dir):
        i+=1
        # File reading
        print(f"{i}/{entries}\t{entry.path}")
        jcampfile = open(entry, "r")
        raw_data = jcamp_read(jcampfile)
        jcampfile.close()
        
        id_ = entry.name.split('_')[0]

        state = metadata_state[id_]
        if state is str:
            if "gas" in metadata_state[id_]:
                skipped_files.append((entry.path, f"gas state found, \"{data["id"]}\""))
                continue
        else:
            unknown_states += 1
        
        try:
            data = format(raw_data, metadata_inchi, id_)
        except KeyError as e:
            skipped_files.append((entry.path, e))
            continue

        data['xdata'], data['ydata'] = interpolate(data)
        data['ydata'] = -np.log10(np.clip(data["ydata"], 1e-6, None)) # Transmission -> Absorbance
        data['ydata'] = normalize(data)
        
        if data["molecule"] is np.nan:
            skipped_files.append((entry.path, f"inchi could not be found, \"{data["id"]}\""))
            continue

        data['fgroups'] = label(data)
        if data['fgroups'] == 1:
            skipped_files.append((entry.path, f"molecule could not be found, \"{data['molecule']}\""))
            continue

        accumulated_data.append(data)
        
    # Writing to csv
    df = pd.DataFrame(accumulated_data)[["fgroups", "xdata", "ydata"]]
    df.to_pickle(r"data-preprocessing-pipeline\spectra-nist.pkl")

    for i, file in enumerate(skipped_files):
        print(f"{i+1} file:{file[0]}\nreason: {file[1]}\n")
    print(f"{len(skipped_files)} files skipped")
    print(f"{unknown_states} unknown states")
