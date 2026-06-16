import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


df = pd.read_pickle(r"data-preprocessing-pipeline\spectra-nist.pkl")

print(df)
print(df.size)

fgroups = pd.json_normalize(df.loc[:, "fgroups"])
fgroup_counts = fgroups.sum().sort_values(ascending=False)

no_groups = (fgroups.sum(axis=1) == 0).sum()
print(fgroup_counts, no_groups)

fgroup_counts.plot(kind='bar', figsize=(12, 6))
plt.ylabel("Counts")
plt.xlabel("Functional Group")
plt.title("Functional Group Counts - Chemotion Database")
plt.tight_layout()
plt.show()

"""
df = pd.read_pickle(r"data-preprocessing-pipeline\spectra-nist.pkl")
label_df = df["fgroups"].apply(pd.Series).astype(int)

y = label_df.values
X = np.arange(len(y))

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_idx, test_idx in msss.split(X, y):
    y_train = label_df.iloc[train_idx]
    y_test  = label_df.iloc[test_idx]

freq = pd.DataFrame({
    "overall":  label_df.mean(),
    "train":    y_train.mean(),
    "test":     y_test.mean(),
})
freq["diff"] = (freq["train"] - freq["test"]).abs()
print(freq.sort_values("diff", ascending=False).to_string())


train_counts = y_train.sum()
test_counts  = y_test.sum()

total = train_counts + test_counts
order = total.sort_values(ascending=False).index

train_counts = train_counts[order]
test_counts  = test_counts[order]
train_pct    = train_counts / total[order] * 100

x = np.arange(len(order))

fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.bar(x, train_counts, label="Train", color="steelblue", alpha=0.85)
ax1.bar(x, test_counts,  label="Test",  color="tomato",    alpha=0.85, bottom=train_counts)
ax1.set_ylabel("Sample count")
ax1.set_xticks(x)
ax1.set_xticklabels(order, rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.plot(x, train_pct, color="black", linewidth=1.5, linestyle="--", marker="o", markersize=4, label="Train %")
ax2.axhline(70, color="gray", linewidth=0.8, linestyle=":")
ax2.set_ylabel("Train split %")
ax2.set_ylim(0, 100)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2)

plt.title("Train/test split per label")
plt.tight_layout()
plt.show()
"""