import os
from jcamp import jcamp_read
import json
import pandas as pd
from data_preproccesing_lib import *

"""
preprocessing pipeline
1. Parsing
2. Formatting
3. Unit correction
4. Interpolation
5. Smoothing
6. Baseline Fitting
7. Normalizing
8. Labelling
"""

INTERVAL = (600, 3900)

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
    jsonfile.close()

    i = 0
    accumulated_data = []
    skipped_files = []
    dir = r"data-preprocessing-pipeline\IR_data-chemotion\exp"
    entries = len(os.listdir(dir))
    for entry in os.scandir(dir):
        i+=1
        print(f"{i}/{entries}\t{entry.path}")

        # PARSING
        jcampfile = open(entry, "r")
        try:
            raw_data = jcamp_read(jcampfile)
        except Exception as exception: # Skips a single file that uses commas instead of dots
            skipped_files.append((entry.path, exception)) 
        jcampfile.close()

        # FORMATTING
        if raw_data['jcamp-dx'] == 5.0:
            raw_data = raw_data['children'][0]
        elif raw_data['jcamp-dx'] != 4.24:
            skipped_files.append((entry.path, f"version not a float, \"{raw_data['jcamp-dx']}\""))
            continue
        try:
            data = format(raw_data, metadata, entry.name, INTERVAL)
        except AssertionError as e:
            skipped_files.append((entry.path, e))
            continue

        
        # CORRECTING UNITS
        data['ydata'] = unit_correct(raw_data, data)
        if data['ydata'] is None:
            skipped_files.append((entry.path, f"y unit is dispersion/absorption index, \"{data["id"]}\""))
            continue

        # INTERPOLATE, SMOOTH, FIT & NORMALIZE
        data['xdata'], data['ydata'] = interpolate(data, INTERVAL)
        data['ydata'] = smooth(data)
        data['ydata'] = fit(data)
        data['ydata'] = normalize(data)
        
        # LABELLING
        data['fgroups'] = label(data, smiles=True)
        if data['fgroups'] == 1:
            skipped_files.append((entry.path, f"molecule could not be found, \"{data['molecule']}\""))
            continue

        accumulated_data.append(data)

    
    plot_spectrum("", data['xdata'], data['ydata'])

    # Writing to csv
    df = pd.DataFrame(accumulated_data)[["fgroups", "xdata", "ydata"]]
    df.to_pickle(r"data-preprocessing-pipeline\spectra-chemotion.pkl")

    for i, file in enumerate(skipped_files):
        print(f"{i+1} file:{file[0].name}\nreason: {file[1]}")
