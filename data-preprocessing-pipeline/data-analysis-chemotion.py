import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

df = pd.read_pickle(r"data-preprocessing-pipeline\spectra-chemotion.pkl")
print(df)
fgroups = pd.json_normalize(df.loc[:, "fgroups"])
fgroup_counts = fgroups.sum().sort_values(ascending=False)
print(fgroups.sum())

fgroup_counts.plot(kind='bar', figsize=(12, 6))
plt.ylabel("Counts")
plt.xlabel("Functional Group")
plt.title("Functional Group Counts - Chemotion Database")
plt.tight_layout()
plt.show()
