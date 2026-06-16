from rdkit import Chem
from rdkit.Chem import Draw

# Each entry: (display_smiles, smarts_query)
FGROUPS = {
    "Ester":           ("CC(=O)OCC",              Chem.MolFromSmarts("[#6][CX3](=O)[OX2H0][#6]")),
    "Carboxylic Acid": ("CC(=O)O",                Chem.MolFromSmarts("[CX3](=O)[OX2H]")),
    "Alkane":          ("CCC",                    Chem.MolFromSmarts("[CX4;H3,H2]")),
    "Alkene":          ("C=CC",                   Chem.MolFromSmarts("[CX3]=[CX3]")),
    "Alcohol":         ("CCO",                    Chem.MolFromSmarts("[#6][OX2H]")),
    "Arene":           ("c1ccccc1",               Chem.MolFromSmarts("[cX3]1[cX3][cX3][cX3][cX3][cX3]1")),
    "Amine":           ("CCN",                    Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")),
    "Ketone":          ("CC(=O)C",                Chem.MolFromSmarts("[#6][CX3](=O)[#6]")),
    "Ether":           ("CCOCC",                  Chem.MolFromSmarts("[OD2]([#6])[#6]")),
    #"imine":           ("CC=NC",                  Chem.MolFromSmarts("[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]")),
    #"sulfonamide":     ("CS(=O)(=O)N",            Chem.MolFromSmarts("[#16X4]([NX3])(=[OX1])(=[OX1])[#6]")),
    #"acyl_halide":     ("CC(=O)Cl",               Chem.MolFromSmarts("[CX3](=[OX1])[F,Cl,Br,I]")),
    #"phosphate":       ("COP(=O)(OC)OC",          Chem.MolFromSmarts("[#15X4](=[OX1])([OX2])[OX2]")),
    #"aldehyde":        ("CC=O",                   Chem.MolFromSmarts("[CX3H1](=O)[#6,H]")),
    "Nitro":           ("C[N+](=O)[O-]",          Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]")),
    #"enamine":         ("CCNC=C",                 Chem.MolFromSmarts("[NX3][CX3]=[CX3]")),
    #"azo":             ("CN=NC",                  Chem.MolFromSmarts("[#6][NX2]=[NX2][#6]")),
    "Sulfonic Acid":   ("CS(=O)(=O)O",            Chem.MolFromSmarts("[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]")),
    "Amide":           ("CC(=O)N",                Chem.MolFromSmarts("[NX3][CX3](=[OX1])[#6]")),
    #"peroxide":        ("COO",                    Chem.MolFromSmarts("[OX2,OX1-][OX2,OX1-]")),
}

mols, legends, highlight_atoms, highlight_bonds = [], [], [], []

for name, (smi, query) in FGROUPS.items():
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"Bad SMILES for {name}: {smi}")
        continue

    mols.append(mol)
    legends.append(name)

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=3,
    subImgSize=(250, 250),
    legends=legends,
)
img.save("functional_groups.png")
img

mol = Chem.MolFromSmiles("O=C(O)c1ccc(C(O)=O)cc1")
img = Chem.Draw.MolToImage(mol, size=(250, 250))
img.save("TPA.png")
img