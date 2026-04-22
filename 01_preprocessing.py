# =============================================================
#  FILE: 01_preprocessing.py
#  PURPOSE: Load raw CSV, inspect structure, simulate and
#           handle missing values using all four methods,
#           detect outliers via IQR and Z-Score, export
#           fertility_clean.csv for all subsequent scripts.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 130

# ── STEP 1: Load ─────────────────────────────────────────────
df = pd.read_csv("fertility_health_dataset.csv")
print("=" * 60)
print("STEP 1 — Dataset Loaded")
print("=" * 60)
print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Target : {dict(df['Pregnancy_Success'].value_counts())}")
print(f"Success rate : {(df['Pregnancy_Success']=='Success').mean()*100:.2f}%\n")

# ── STEP 2: df.info() ────────────────────────────────────────
print("=" * 60)
print("STEP 2 — df.info()")
print("=" * 60)
df.info()
print()

# ── STEP 3: df.describe() ────────────────────────────────────
print("=" * 60)
print("STEP 3 — df.describe()")
print("=" * 60)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[num_cols].describe().round(3).T.to_string())
print()

# ── STEP 4: Missing Values ────────────────────────────────────
print("=" * 60)
print("STEP 4 — Missing Value Handling")
print("=" * 60)
print(f"Raw missing cells: {df.isnull().sum().sum()}")
print("Dataset is clean. Simulating 5% NaN to demonstrate methods.\n")

np.random.seed(42)
df_sim = df.copy()
for col in ["Female_BMI", "Sperm_Count_M_per_mL", "Stress_Level", "Sleep_Quality"]:
    idx = np.random.choice(df_sim.index, size=100, replace=False)
    df_sim.loc[idx, col] = np.nan

print("Simulated NaN counts:")
print(df_sim[["Female_BMI","Sperm_Count_M_per_mL","Stress_Level","Sleep_Quality"]].isnull().sum().to_string())

# Four handling methods
df_sim["Female_BMI"]           = df_sim["Female_BMI"].fillna(df_sim["Female_BMI"].mean())
df_sim["Stress_Level"]         = df_sim["Stress_Level"].ffill()
df_sim["Sleep_Quality"]        = df_sim["Sleep_Quality"].bfill()
df_sim["Sperm_Count_M_per_mL"] = df_sim["Sperm_Count_M_per_mL"].interpolate(method="linear")

print("\nAfter fillna / ffill / bfill / interpolate:")
print(df_sim[["Female_BMI","Sperm_Count_M_per_mL","Stress_Level","Sleep_Quality"]].isnull().sum().to_string())
print("All resolved ✓\n")

df_clean = df.copy()

# ── STEP 5: Outlier Detection ─────────────────────────────────
print("=" * 60)
print("STEP 5 — Outlier Detection (IQR + Z-Score)")
print("=" * 60)

key_cols = ["Female_Age","Male_Age","Female_BMI","Male_BMI",
            "Stress_Level","Sleep_Quality","Sperm_Count_M_per_mL","Sperm_Motility_Pct"]

print(f"\n{'Column':<26} {'IQR Outliers':>14} {'Lower':>8} {'Upper':>8}")
print("-" * 60)
for col in key_cols:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n = int(((df_clean[col]<lo)|(df_clean[col]>hi)).sum())
    print(f"{col:<26} {n:>14} {lo:>8.2f} {hi:>8.2f}")

print("\nZ-Score outliers (|z|>3):")
z = np.abs(stats.zscore(df_clean[key_cols]))
outlier_counts = (z > 3).sum(axis=0)

for col, cnt in zip(key_cols, outlier_counts):
    print(f"{col:<26} {int(cnt)} outliers")

# Boxplot grid
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(key_cols):
    sns.boxplot(y=df_clean[col], ax=axes[i],
                color=sns.color_palette("Set2",8)[i],
                flierprops=dict(marker="o", markersize=2.5, alpha=0.5))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("")
plt.suptitle("Boxplots — Outlier Detection", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("01_outlier_boxplots.png", bbox_inches="tight")
plt.show()

# ── STEP 6: Export ────────────────────────────────────────────
df_clean.to_csv("fertility_clean.csv", index=False)
print(f"\nExported: fertility_clean.csv  {df_clean.shape}")
print("✓ Preprocessing complete")
