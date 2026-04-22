from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Optional, List
import math, os

# ── App setup ────────────────────────────────────────────
app = FastAPI(
    title="Fertility Health Dashboard API",
    description="Dynamic backend for fertility dataset analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the HTML dashboard) from current directory
app.mount("/static", StaticFiles(directory="."), name="static")


# ── Load dataset once at startup ───────────────────────────
def load_df():
    for fname in ["fertility_clean.csv", "fertility_health_dataset.csv"]:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            df["Success"] = (df["Pregnancy_Success"] == "Success").astype(int)
            df["Female_AgeGrp"] = pd.cut(
                df["Female_Age"],
                bins=[19, 25, 30, 35, 40, 45],
                labels=["20-25", "26-30", "31-35", "36-40", "41-45"]
            ).astype(str)
            return df
    raise FileNotFoundError("fertility_clean.csv not found. Run 01_preprocessing.py first.")

DF_FULL = load_df()
print(f"✓ Dataset loaded: {DF_FULL.shape[0]:,} rows × {DF_FULL.shape[1]} columns")


# ── Filter helper ───────────────────────────────────────────
def filter_df(
    age_min: int = 20,
    age_max: int = 45,
    pcos: Optional[str] = None,
    period: Optional[str] = None,
    f_smoke: Optional[str] = None,
    m_smoke: Optional[str] = None,
    f_exercise: Optional[str] = None,
    m_exercise: Optional[str] = None,
    f_alcohol: Optional[str] = None,
    m_alcohol: Optional[str] = None,
    outcome: Optional[str] = None,
) -> pd.DataFrame:
    df = DF_FULL.copy()
    df = df[(df["Female_Age"] >= age_min) & (df["Female_Age"] <= age_max)]
    if pcos:       df = df[df["PCOS"] == pcos]
    if period:     df = df[df["Period_Regularity"] == period]
    if f_smoke:    df = df[df["Female_Smokes"] == f_smoke]
    if m_smoke:    df = df[df["Male_Smokes"] == m_smoke]
    if f_exercise: df = df[df["Female_Exercise"] == f_exercise]
    if m_exercise: df = df[df["Male_Exercise"] == m_exercise]
    if f_alcohol:  df = df[df["Female_Alcohol"] == f_alcohol]
    if m_alcohol:  df = df[df["Male_Alcohol"] == m_alcohol]
    if outcome:    df = df[df["Pregnancy_Success"] == outcome]
    return df


def clean(obj):
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ═════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/")
def serve_dashboard():
    """Serve the HTML dashboard."""
    return FileResponse("dashboard 1.html")


# ── 1. Overview Metrics ────────────────────────────────────
@app.get("/api/overview")
def overview(
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
):
    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)
    n = len(df)
    if n == 0:
        return JSONResponse({"error": "No data matches filters"}, status_code=404)

    num_cols = ["Female_Age","Male_Age","Female_BMI","Male_BMI",
                "Stress_Level","Sleep_Quality","Past_Pregnancies",
                "Sperm_Count_M_per_mL","Sperm_Motility_Pct"]

    # Skewness
    skewness = {col: round(float(df[col].skew()), 4) for col in num_cols}

    # Correlation matrix
    corr_cols = num_cols + ["Success"]
    corr = df[corr_cols].corr().round(4)
    corr_data = {col: {c: round(float(corr.loc[col, c]), 4)
                       for c in corr_cols}
                 for col in corr_cols}

    # Describe
    desc = df[num_cols].describe().round(3).to_dict()

    # Age-group success rates
    df2 = df.copy()
    df2["AgeGrp"] = pd.cut(df2["Female_Age"],
        bins=[19,25,30,35,40,45],
        labels=["20-25","26-30","31-35","36-40","41-45"])
    age_sr = df2.groupby("AgeGrp", observed=True)["Success"].mean().mul(100).round(2)

    # Success / Failure counts
    oc = df["Pregnancy_Success"].value_counts().to_dict()

    result = {
        "n":               n,
        "n_total":         len(DF_FULL),
        "n_success":       int(df["Success"].sum()),
        "success_rate":    round(float(df["Success"].mean()) * 100, 2),
        "avg_female_age":  round(float(df["Female_Age"].mean()), 2),
        "avg_male_age":    round(float(df["Male_Age"].mean()), 2),
        "avg_sperm_count": round(float(df["Sperm_Count_M_per_mL"].mean()), 2),
        "avg_sperm_motility": round(float(df["Sperm_Motility_Pct"].mean()), 2),
        "avg_sleep":       round(float(df["Sleep_Quality"].mean()), 2),
        "avg_stress":      round(float(df["Stress_Level"].mean()), 2),
        "avg_female_bmi":  round(float(df["Female_BMI"].mean()), 2),
        "pcos_pct":        round(float((df["PCOS"]=="Yes").mean())*100, 2),
        "outcome_counts":  oc,
        "age_success_rate": age_sr.to_dict(),
        "skewness":        skewness,
        "correlation":     corr_data,
        "describe":        desc,
        # Clinical averages by outcome
        "clinical_avgs": {
            grp: {
                col: round(float(sub[col].mean()), 3)
                for col in ["Sperm_Count_M_per_mL","Sperm_Motility_Pct",
                             "Sleep_Quality","Stress_Level","Female_Age","Female_BMI"]
            }
            for grp, sub in df.groupby("Pregnancy_Success")
        },
    }
    return JSONResponse(clean(result))


# ── 2. Lifestyle Success Rates ──────────────────────────────
@app.get("/api/lifestyle")
def lifestyle(
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
):
    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)

    overall_sr = float(df["Success"].mean()) * 100

    def group_rates(col):
        g = (df.groupby(col)["Success"]
               .agg(["mean","count"])
               .reset_index())
        g.columns = ["category","rate","count"]
        g["rate"]   = (g["rate"] * 100).round(2)
        g["diff"]   = (g["rate"] - overall_sr).round(2)
        return g.to_dict(orient="records")

    lifestyle_cols = [
        "PCOS","Period_Regularity","Female_Smokes","Male_Smokes",
        "Female_Exercise","Male_Exercise","Female_Alcohol","Male_Alcohol",
        "Past_Pregnancies"
    ]

    result = {
        "overall_success_rate": round(overall_sr, 2),
        "factors": {col: group_rates(col) for col in lifestyle_cols if col in df.columns}
    }
    return JSONResponse(clean(result))


# ── 3. Histograms ───────────────────────────────────────────
@app.get("/api/histogram/{column}")
def histogram(
    column: str,
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    bins: int = Query(30),
):
    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)

    if column not in df.columns:
        return JSONResponse({"error": f"Column '{column}' not found"}, status_code=400)

    vals = df[column].dropna()
    counts, edges = np.histogram(vals, bins=bins)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]

    result = {
        "column":   column,
        "n":        int(len(vals)),
        "mean":     round(float(vals.mean()), 4),
        "std":      round(float(vals.std()),  4),
        "skewness": round(float(vals.skew()), 4),
        "min":      round(float(vals.min()),  4),
        "max":      round(float(vals.max()),  4),
        "bins":     [round(float(c), 2) for c in centers],
        "counts":   counts.tolist(),
        # By outcome
        "by_outcome": {
            grp: {
                "mean":   round(float(sub[column].mean()), 4),
                "counts": np.histogram(sub[column].dropna(), bins=edges)[0].tolist()
            }
            for grp, sub in df.groupby("Pregnancy_Success")
        }
    }
    return JSONResponse(clean(result))


# ── 4. Scatter data ───────────────────────────────────────────
@app.get("/api/scatter")
def scatter(
    x: str = Query("Sperm_Count_M_per_mL"),
    y: str = Query("Sperm_Motility_Pct"),
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    max_points: int = Query(500),
):
    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)

    sub = df[[x, y, "Pregnancy_Success", "PCOS", "Female_Age"]].dropna()

    # Sample for performance
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)

    result = {
        "n": len(sub),
        "points": [
            {
                "x": round(float(row[x]), 2),
                "y": round(float(row[y]), 2),
                "outcome": row["Pregnancy_Success"],
                "pcos": row["PCOS"],
                "age": int(row["Female_Age"]),
            }
            for _, row in sub.iterrows()
        ]
    }
    return JSONResponse(clean(result))


# ── 5. Hypothesis Test ────────────────────────────────────────
@app.get("/api/hypothesis")
def hypothesis(
    variable: str = Query("Sperm_Count_M_per_mL"),
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    alpha: float = Query(0.05),
):
    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)

    grp_s = df[df["Pregnancy_Success"] == "Success"][variable].dropna()
    grp_f = df[df["Pregnancy_Success"] == "Failure"][variable].dropna()

    if len(grp_s) < 2 or len(grp_f) < 2:
        return JSONResponse({"error": "Insufficient data in one group"}, status_code=422)

    # Welch T-Test
    t, p = stats.ttest_ind(grp_s, grp_f, equal_var=False)

    # Z-Test
    n1, n2 = len(grp_s), len(grp_f)
    m1, m2 = float(grp_s.mean()), float(grp_f.mean())
    s1, s2 = float(grp_s.std(ddof=1)), float(grp_f.std(ddof=1))
    se = math.sqrt(s1**2/n1 + s2**2/n2)
    z  = (m1 - m2) / se if se > 0 else 0
    zp = 2 * (1 - float(stats.norm.cdf(abs(z))))

    # Cohen's d
    pool = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    d    = (m1 - m2) / pool if pool > 0 else 0
    eff  = ("Negligible" if abs(d)<0.2 else "Small" if abs(d)<0.5
            else "Medium" if abs(d)<0.8 else "Large")

    # 95% CI
    ci_s = float(stats.t.ppf(0.975, n1-1)) * s1 / math.sqrt(n1)
    ci_f = float(stats.t.ppf(0.975, n2-1)) * s2 / math.sqrt(n2)

    # Histogram bins for distribution chart
    all_vals = pd.concat([grp_s, grp_f])
    bins_arr = np.linspace(all_vals.min(), all_vals.max(), 26)
    hist_s, _ = np.histogram(grp_s, bins=bins_arr)
    hist_f, _ = np.histogram(grp_f, bins=bins_arr)
    bin_centers = [(bins_arr[i]+bins_arr[i+1])/2 for i in range(len(bins_arr)-1)]

    result = {
        "variable": variable,
        "alpha":   alpha,
        "reject":  bool(p < alpha),
        "decision": "REJECT H0" if p < alpha else "FAIL TO REJECT H0",
        # T-Test
        "t_stat":  round(float(t), 4),
        "p_value": round(float(p), 6),
        # Z-Test
        "z_stat":  round(z, 4),
        "z_p":     round(zp, 6),
        # Effect
        "cohens_d": round(d, 4),
        "effect":   eff,
        # Group stats
        "success": {"n": n1, "mean": round(m1,4), "std": round(s1,4), "ci95": round(ci_s,4)},
        "failure": {"n": n2, "mean": round(m2,4), "std": round(s2,4), "ci95": round(ci_f,4)},
        # Chart data
        "dist_bins":    [round(float(b),2) for b in bin_centers],
        "dist_success": hist_s.tolist(),
        "dist_failure": hist_f.tolist(),
    }
    return JSONResponse(clean(result))


# ── 6. Regression ───────────────────────────────────────────
@app.get("/api/regression/{reg_num}")
def regression(
    reg_num: int,
    age_min: int = Query(20), age_max: int = Query(45),
    pcos: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    f_smoke: Optional[str] = Query(None),
    m_smoke: Optional[str] = Query(None),
    f_exercise: Optional[str] = Query(None),
    m_exercise: Optional[str] = Query(None),
    f_alcohol: Optional[str] = Query(None),
    m_alcohol: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    predict_x: Optional[float] = Query(None),
):
    # Decide columns based on which regression
    if reg_num == 1:
        x_col, y_col = "Female_Age", "Male_Age"
        label = "Female Age → Male Age"
    elif reg_num == 2:
        x_col, y_col = "Sperm_Motility_Pct", "Sperm_Count_M_per_mL"
        label = "Sperm Motility → Sperm Count"
    else:
        return JSONResponse({"error": "reg_num must be 1 or 2"}, status_code=400)

    df = filter_df(age_min, age_max, pcos, period, f_smoke, m_smoke,
                   f_exercise, m_exercise, f_alcohol, m_alcohol, outcome)
    sub = df[[x_col, y_col]].dropna()

    if len(sub) < 10:
        return JSONResponse({"error": "Not enough data for regression"}, status_code=422)

    X = sub[[x_col]].values
    y = sub[y_col].values

    r_val, _ = stats.pearsonr(X.flatten(), y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    slope     = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2        = float(r2_score(y_te, y_pred))
    mae       = float(mean_absolute_error(y_te, y_pred))
    rmse      = float(math.sqrt(mean_squared_error(y_te, y_pred)))

    # Regression line points (for chart)
    x_min, x_max = float(X.min()), float(X.max())
    line_x = [round(x_min + i*(x_max-x_min)/99, 2) for i in range(100)]
    line_y = [round(slope*xv + intercept, 3) for xv in line_x]

    # Scatter sample (300 pts max)
    sample_n = min(300, len(sub))
    smp = sub.sample(sample_n, random_state=1)
    scatter_pts = [
        {"x": round(float(r[x_col]),2), "y": round(float(r[y_col]),2)}
        for _, r in smp.iterrows()
    ]

    # Residuals sample
    res_pts = [
        {"x": round(float(yp),2), "y": round(float(yt-yp),2)}
        for yp, yt in zip(y_pred[:100], y_te[:100])
    ]

    # Actual vs Predicted sample
    avp_pts = [
        {"actual": round(float(yt),2), "pred": round(float(yp),2)}
        for yt, yp in zip(y_te[:100], y_pred[:100])
    ]

    # Prediction for given x
    prediction = None
    if predict_x is not None:
        prediction = round(slope * predict_x + intercept, 3)

    result = {
        "label":       label,
        "x_col":       x_col,
        "y_col":       y_col,
        "n_train":     int(len(X_tr)),
        "n_test":      int(len(X_te)),
        "pearson_r":   round(r_val, 4),
        "r2":          round(r2, 4),
        "slope":       round(slope, 4),
        "intercept":   round(intercept, 4),
        "mae":         round(mae, 4),
        "rmse":        round(rmse, 4),
        "equation":    f"{y_col} = {slope:.4f} × {x_col} + {intercept:.4f}",
        "line_x":      line_x,
        "line_y":      line_y,
        "scatter":     scatter_pts,
        "residuals":   res_pts,
        "actual_vs_pred": avp_pts,
        "prediction":  prediction,
    }
    return JSONResponse(clean(result))


# ── 7. Filter options (for dropdowns) ────────────────────────
@app.get("/api/options")
def options():
    return JSONResponse({
        "age_range":   [int(DF_FULL["Female_Age"].min()), int(DF_FULL["Female_Age"].max())],
        "pcos":        sorted(DF_FULL["PCOS"].unique().tolist()),
        "period":      sorted(DF_FULL["Period_Regularity"].unique().tolist()),
        "smoking":     sorted(DF_FULL["Female_Smokes"].unique().tolist()),
        "exercise":    sorted(DF_FULL["Female_Exercise"].unique().tolist()),
        "alcohol":     sorted(DF_FULL["Female_Alcohol"].unique().tolist()),
        "outcome":     sorted(DF_FULL["Pregnancy_Success"].unique().tolist()),
    })


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
