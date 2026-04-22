import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

df = pd.read_csv("fertility_clean.csv")
df["Success"] = (df["Pregnancy_Success"] == "Success").astype(int)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

NUM = ["Female_Age","Male_Age","Female_BMI","Male_BMI",
       "Stress_Level","Sleep_Quality","Past_Pregnancies",
       "Sperm_Count_M_per_mL","Sperm_Motility_Pct"]

# ── A: Summary Statistics ────────────────────────────────
print("=" * 60)
print("A — Summary Statistics")
print("=" * 60)
print(df[NUM].describe().round(2).T.to_string())
print()

# ── B: Correlation Heatmap ─────────────────────────────────
print("Plotting Correlation Heatmap …")
corr = df[NUM + ["Success"]].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap=cmap, linewidths=0.5, vmin=-1, vmax=1,
            ax=ax, annot_kws={"size": 9})
ax.set_title("Correlation Heatmap — Fertility Health Dataset", pad=14)
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", bbox_inches="tight")
plt.show()

print("\nTop correlations:")
pairs = (corr.where(np.tril(np.ones_like(corr),k=-1).astype(bool))
             .stack().reset_index())
pairs.columns = ["A","B","r"]
pairs["abs"] = pairs["r"].abs()
print(pairs.sort_values("abs", ascending=False).head(8)[["A","B","r"]].to_string(index=False))
print()

# ── C: Histograms with Skewness ────────────────────────────
print("Plotting Histograms …")
from scipy import stats as sc_stats

fig, axes = plt.subplots(3, 3, figsize=(15, 11))
axes = axes.flatten()
colors = sns.color_palette("Set2", 9)
for i, col in enumerate(NUM):
    skv = df[col].skew()
    axes[i].hist(df[col].dropna(), bins=35, color=colors[i],
                 edgecolor="white", alpha=0.85)
    mu, sig = df[col].mean(), df[col].std()
    x = np.linspace(df[col].min(), df[col].max(), 200)
    ax2 = axes[i].twinx()
    ax2.plot(x, sc_stats.norm.pdf(x, mu, sig), "r--", lw=1.5, alpha=0.7)
    ax2.set_yticks([])
    axes[i].set_title(f"{col}\nskewness = {skv:.3f}", fontsize=10)
    axes[i].set_xlabel(col, fontsize=9)
plt.suptitle("Histograms — Skewness Analysis", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("02_histograms.png", bbox_inches="tight")
plt.show()

print("Skewness table:")
sk = df[NUM].skew().round(4).reset_index()
sk.columns = ["Column","Skewness"]
sk["Interpretation"] = sk["Skewness"].apply(
    lambda s: "Highly Right-Skewed" if s>1 else
              "Mod. Right-Skewed"   if s>0.5 else
              "Approx Normal"       if -0.5<=s<=0.5 else
              "Mod. Left-Skewed"    if s>-1 else
              "Highly Left-Skewed")
print(sk.to_string(index=False))
print()

# ── D: Comparative Boxplots by Outcome ────────────────────
print("Plotting Boxplots by Outcome …")
pal = {"Success":"#2ecc71","Failure":"#e74c3c"}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
pairs_box = [
    ("Sperm_Count_M_per_mL","Sperm Count (M/mL)"),
    ("Female_Age","Female Age"),
    ("Sleep_Quality","Sleep Quality"),
    ("Female_BMI","Female BMI"),
    ("Sperm_Motility_Pct","Sperm Motility %"),
    ("Stress_Level","Stress Level"),
]
for ax, (col, lbl) in zip(axes.flatten(), pairs_box):
    sns.boxplot(x="Pregnancy_Success", y=col, data=df,
                ax=ax, palette=pal)
    ax.set_title(f"{lbl} by Outcome")
    ax.set_xlabel("")
plt.suptitle("Boxplots — Key Features vs Pregnancy Outcome",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("02_boxplots.png", bbox_inches="tight")
plt.show()

# ── E: Success Rate by Lifestyle Factor ────────────────────
print("Plotting Lifestyle Success Rates …")
lifestyle_cols = ["PCOS","Period_Regularity","Male_Smokes",
                  "Female_Exercise","Female_Alcohol","Male_Alcohol"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), lifestyle_cols):
    grp = df.groupby(col)["Success"].mean().sort_values() * 100
    colors_b = sns.color_palette("Set2", len(grp))
    ax.barh(grp.index, grp.values, color=colors_b)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title(f"Success Rate by {col}")
    for i, v in enumerate(grp.values):
        ax.text(v+0.2, i, f"{v:.1f}%", va="center", fontsize=9)
plt.suptitle("Pregnancy Success Rate by Lifestyle Factors",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("02_lifestyle.png", bbox_inches="tight")
plt.show()

# ── F: Age group success ────────────────────────────────────
df["AgeGroup"] = pd.cut(df["Female_Age"],
    bins=[19,25,30,35,40,45],
    labels=["20-25","26-30","31-35","36-40","41-45"])
age_sr = df.groupby("AgeGroup", observed=True)["Success"].mean()*100

fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(age_sr.index, age_sr.values,
              color=sns.color_palette("RdYlGn_r", len(age_sr)),
              edgecolor="white")
ax.set_xlabel("Female Age Group")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Pregnancy Success Rate Declines with Female Age")
for bar, val in zip(bars, age_sr.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("02_age_success.png", bbox_inches="tight")
plt.show()

print("✓ EDA complete → run 03_hypothesis_testing.py next")
