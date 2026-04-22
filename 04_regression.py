# =============================================================
#  FILE: 04_regression.py
#  PURPOSE: Objectives 2 & 3 — Two Simple Linear Regressions
#
#  Regression 1: Female_Age → Male_Age
#    Justification: Pearson r = 0.9593 (strongest in heatmap)
#    Insight: Couples share similar age — social pairing
#
#  Regression 2: Sperm_Motility_Pct → Sperm_Count_M_per_mL
#    Justification: Clinical pair — demonstrates SLR limits
#    Key lesson: Low R² → motivates Multiple Regression
#
#  Each model: Train/Test split, R², MAE, MSE, RMSE,
#              Scatter, Actual vs Predicted, Residual Plot
#  Run AFTER 03_hypothesis_testing.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130

df = pd.read_csv("fertility_clean.csv")
print(f"Loaded: {df.shape[0]:,} rows\n")


def run_regression(df, x_col, y_col, label, num,
                   c_point, c_line, fname):
    """Fit SLR, print full stats, produce 3-panel plot."""
    print("=" * 65)
    print(f"  REGRESSION {num}: {x_col}  →  {y_col}")
    print("=" * 65)

    X = df[[x_col]].values
    y = df[y_col].values

    # Pearson r
    r, r_p = stats.pearsonr(X.flatten(), y)
    print(f"\n  Pearson r     : {r:.6f}")
    print(f"  r² (formula)  : {r**2:.6f}")
    print(f"  Corr p-value  : {r_p:.4e}")

    # Train / Test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42)
    print(f"\n  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

    # Fit
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    slope     = model.coef_[0]
    intercept = model.intercept_
    y_pred    = model.predict(X_te)

    # Metrics
    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    mse  = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n  Equation : {y_col} = {slope:.4f} × {x_col} + {intercept:.4f}")
    print(f"\n  Metrics (Test Set):")
    print(f"    R²   = {r2:.6f}  ({r2*100:.2f}% variance explained)")
    print(f"    MAE  = {mae:.4f}")
    print(f"    MSE  = {mse:.4f}")
    print(f"    RMSE = {rmse:.4f}")

    # Sample predictions
    sample = np.linspace(X.min(), X.max(), 6).reshape(-1,1)
    print(f"\n  Sample Predictions:")
    for sv, pv in zip(sample.flatten(), model.predict(sample)):
        print(f"    {x_col} = {sv:.1f}  →  {y_col} ≈ {pv:.2f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter + line
    x_rng = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
    axes[0].scatter(X_tr, y_tr, alpha=0.3, s=14, color=c_point, label="Train")
    axes[0].scatter(X_te, y_te, alpha=0.5, s=14, color="gray", label="Test")
    axes[0].plot(x_rng, model.predict(x_rng), color=c_line,
                 lw=2.5, label=f"ŷ={slope:.3f}x+{intercept:.2f}")
    axes[0].set_xlabel(x_col); axes[0].set_ylabel(y_col)
    axes[0].set_title(f"Scatter + Regression Line\nr={r:.4f}  R²={r2:.4f}")
    axes[0].legend(fontsize=9)

    # Actual vs Predicted
    mn = min(y_te.min(), y_pred.min())
    mx = max(y_te.max(), y_pred.max())
    axes[1].scatter(y_te, y_pred, alpha=0.4, s=14, color=c_point)
    axes[1].plot([mn,mx],[mn,mx],"r--",lw=2,label="Perfect")
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title("Actual vs Predicted")
    axes[1].legend()

    # Residuals
    res = y_te - y_pred
    axes[2].scatter(y_pred, res, alpha=0.4, s=14, color=c_point)
    axes[2].axhline(0, color="red", lw=2, ls="--")
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Residuals")
    axes[2].set_title("Residual Plot")

    plt.suptitle(f"Regression {num}: {label}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {fname}\n")
    return model, dict(slope=slope, intercept=intercept,
                       r=r, r2=r2, mae=mae, rmse=rmse)


# ── Regression 1: Female_Age → Male_Age ──────────────────────
m1, s1 = run_regression(
    df, "Female_Age", "Male_Age",
    "Female Age predicts Partner Age  [r=0.9593]",
    1, "#3498db", "#e74c3c", "04_regression_1.png")

# ── Regression 2: Sperm_Motility_Pct → Sperm_Count_M_per_mL ─
m2, s2 = run_regression(
    df, "Sperm_Motility_Pct", "Sperm_Count_M_per_mL",
    "Sperm Motility predicts Sperm Count  [clinical pair]",
    2, "#9b59b6", "#e67e22", "04_regression_2.png")

# ── Comparison Summary ────────────────────────────────────────
print("=" * 65)
print("  COMPARISON SUMMARY")
print("=" * 65)
print(f"""
  Regression 1 (Female Age → Male Age)
    R² = {s1['r2']:.4f}  →  Very strong. Partners are matched in age.
    Slope means: for every 1yr increase in female age,
    male age increases by ~{s1['slope']:.2f} years.

  Regression 2 (Sperm Motility → Sperm Count)
    R² = {s2['r2']:.4f}  →  Near-zero. Biologically independent.
    These parameters are regulated separately — SLR fails here.
    This motivates Multiple Linear Regression as future scope.

  VIVA KEY POINTS:
  1. R² alone doesn't define a good study — it defines fit.
  2. Low R² teaches limitations, which IS a finding.
  3. Residual plots: flat/random = good. Pattern = violation.
  4. For skewed variables (wage, value) → log-transform first.
""")
print("✓ Regression complete")
