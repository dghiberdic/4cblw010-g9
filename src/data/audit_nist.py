import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IR_CSV  = os.path.join(ROOT, "data/raw/nist_ir_info.csv")
CMP_CSV = os.path.join(ROOT, "data/raw/nist_compounds.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CONDENSED = {"solid", "liquid", "solution", "film", "semi-solid", "paste"}

COMPOUNDS = {
    "Terephthalic acid (TPA)": "100-21-0",
    "Ethylene glycol (EG)":    "107-21-1",
}

ir  = pd.read_csv(IR_CSV)
cmp = pd.read_csv(CMP_CSV, low_memory=False)
cas2id = cmp.dropna(subset=["cas_rn"]).set_index("cas_rn")["ID"].astype(str).to_dict()

state_counts = ir["state"].value_counts()
print(f"total {len(ir):,}\n")
print(state_counts.to_string())

condensed = ir[ir["state"].isin(CONDENSED)]
print(f"\nnon gas total: {len(condensed):,}")

print(f"\n{'Compound':<30} {'Condensed':>10} {'Gas':>5}")
print("-" * 48)
for name, cas in COMPOUNDS.items():
    cid = cas2id.get(cas)
    if cid is None:
        print(f"{name:<30} {'N/A':>10} {'N/A':>5}")
        continue
    rows = ir[ir["cID"] == cid]
    c = rows[rows["state"].isin(CONDENSED)].shape[0]
    g = rows[rows["state"] == "gas"].shape[0]
    print(f"{name:<30} {c:>10} {g:>5}")

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#d55e00" if s == "gas" else "#009e73" for s in state_counts.index]
bars = ax.bar(state_counts.index, state_counts.values, color=colors)
ax.bar_label(bars, fmt="%d", padding=2, fontsize=8)
ax.set_xlabel("State")
ax.set_ylabel("no spectra")
ax.set_title("NIST IR spectra by state  (green = condensed, orange = gas)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "nist_state_distribution.png"), dpi=150)
plt.close(fig)
print(f"\nsaved outputs/nist_state_distribution.png")
