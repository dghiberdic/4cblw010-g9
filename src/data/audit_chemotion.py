import os
import json
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META_JSON = os.path.join(ROOT, "data/raw/chemotion/data/dataset/meta_data.json")

SMARTS = {
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

LABELS = {
    "ester":           "Ester",
    "carboxylic_acid": "Carb. acid",
    "alkane":          "Alkane",
    "alkene":          "Alkene",
    "alcohol":         "Alcohol",
    "arene":           "Arene",
    "amine":           "Amine",
    "ketone":          "Ketone",
    "ether":           "Ether",
    "imine":           "Imine",
    "sulfonamide":     "Sulfonamide",
    "acyl_halide":     "Acyl halide",
    "phosphate":       "Phosphate",
    "aldehyde":        "Aldehyde",
    "nitro":           "Nitro",
    "enamine":         "Enamine",
    "azo":             "Azo",
    "sulfonic_acid":   "Sulfonic acid",
    "amide":           "Amide",
    "peroxide":        "Peroxide",
}

print("Loading dataset...")
records = json.load(open(META_JSON))
df = pd.DataFrame(records)
print(f"{len(df):,} total entries.")

print("SMARTS matching...")
rows = []
for smi in df["cano_smiles"]:
    if not isinstance(smi, str) or not smi.strip():
        rows.append({k: False for k in SMARTS})
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        rows.append({k: False for k in SMARTS})
        continue
    rows.append({k: mol.HasSubstructMatch(pat) for k, pat in SMARTS.items()})

ldf = pd.DataFrame(rows, index=df.index)

valid_mask = df["cano_smiles"].apply(lambda s: isinstance(s, str) and bool(s.strip()))
valid = df[valid_mask]
vl = ldf.loc[valid.index]

print(f"\n{len(valid):,} valid / {len(df):,} total.")
print(f"at least one target group: {vl.any(axis=1).sum():,} ({100*vl.any(axis=1).mean():.1f}%)\n")

for k, label in LABELS.items():
    n = vl[k].sum()
    print(f" {label:<15} {n:>5,}  ({100*n/len(valid):.1f}%)")
