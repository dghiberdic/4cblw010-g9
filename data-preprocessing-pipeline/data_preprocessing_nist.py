from jcamp import jcamp_read
import numpy as np
import os
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
    metadata = pd.read_csv(r"data\raw\nist_ir_info.csv", dtype=str)[["cID", "inchi", "state"]]
    state_metadata = metadata.set_index("cID")["state"].to_dict()
    metadata = metadata.set_index("cID")["inchi"].to_dict()
    
    i = 0
    accumulated_data = []
    skipped_files = []
    dir = r"data\raw\IR"
    entries = len(os.listdir(dir))
    for entry in os.scandir(dir):
        i+=1
        print(f"{i}/{entries}\t{entry.path}")

        # PARSING
        jcampfile = open(entry, "r")
        raw_data = jcamp_read(jcampfile)
        jcampfile.close()

        id_ = entry.name.split('_')[0]
        if "gas" == state_metadata[id_]:
            continue   

        # FORMATTING
        try:
            data = format(raw_data, metadata, id_, INTERVAL)
        except KeyError as e:
            skipped_files.append((entry.path, e))
            continue
        except AssertionError as e:
            skipped_files.append((entry.path, e))
            continue
        if data["molecule"] is np.nan:
            skipped_files.append((entry.path, f"inchi could not be found, \"{data["id"]}\""))
            continue

        # CORRECTING UNITS
        data['ydata'] = unit_correct(raw_data, data)
        if data['ydata'] is None:
            skipped_files.append((entry.path, f"y unit is dispersion/absorption index, \"{data["id"]}\""))
            continue

        # INTERPOLATE, SMOOTH, FIT & NORMALIZE
        data['xdata'], data['ydata'] = interpolate(data, INTERVAL)
        data['ydata'] = smooth(data)
        #data['ydata'] = fit(data)
        data['ydata'] = normalize(data)
        
        # LABELLING
        data['fgroups'] = label(data, inchi=True)
        if data['fgroups'] == 1:
            skipped_files.append((entry.path, f"molecule could not be found, \"{data['molecule']}\""))
            continue

        
        accumulated_data.append(data)

        
    # Writing to csv
    df = pd.DataFrame(accumulated_data)[["fgroups", "xdata", "ydata"]]
    df.to_pickle(r"data-preprocessing-pipeline\spectra-nist.pkl")

    for i, file in enumerate(skipped_files):
        print(f"{i+1} file:{file[0]}\treason: {file[1]}")
    print(f"{len(skipped_files)} files skipped")
