import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IR_CSV = os.path.join(ROOT, "data/raw/nist/NistChemData/data/spectra/nist_ir_info.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CONDENSED = {"solid", "liquid", "solution", "film", "semi-solid", "paste"}

SMARTS = {
    "ester":           Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]"),
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "arene":           Chem.MolFromSmarts("c1ccccc1"),
    "primary_alcohol": Chem.MolFromSmarts("[OX2H][CX4H2]"),
}

LABELS = {
    "ester":           "Ester",
    "carboxylic_acid": "Carb. acid",
    "arene":           "Arene",
    "primary_alcohol": "Prim. OH",
}

print("Loading dataset...")
ir = pd.read_csv(IR_CSV)
condensed = ir[ir["state"].isin(CONDENSED)].copy()
print(f"{len(condensed):,} non gas phase entries.")

print("SMART matching")
rows = []

for inchi in condensed["inchi"]:
    if not isinstance(inchi, str) or not inchi.startswith("InChI="):
        rows.append({k: False for k in SMARTS})
        continue
    
    mol = MolFromInchi(inchi)
    if mol is None:
        rows.append({k: False for k in SMARTS})
        continue
    
    rows.append({k: mol.HasSubstructMatch(pat) for k, pat in SMARTS.items()})

ldf = pd.DataFrame(rows, index=condensed.index)
condensed = pd.concat([condensed, ldf], axis=1)

valid = condensed[condensed["inchi"].str.startswith("InChI=", na=False)]
vl = ldf.loc[valid.index]

print(f"\n{len(valid):,} valid / {len(condensed):,} total.")
print(f"at least one target group: {vl.any(axis=1).sum():,} ({100*vl.any(axis=1).mean():.1f}%)\n")

for k, label in LABELS.items():
    n = vl[k].sum()
    print(f" {label:<15} {n:>5,}  ({100*n/len(valid):.1f}%)")