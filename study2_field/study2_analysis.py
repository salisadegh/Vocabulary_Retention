"""
Study 2: Self-reported vocabulary learning strategies vs measured retention
============================================================================
Full reproducible analysis on the field sample (n=109, Iranian EFL learners).

Analysis structure (pre-specified):
  PRIMARY   : Total strategy use vs retention (capped ratio), single test.
  SECONDARY : Residualized-change model: Followup ~ Initial + Total strategy.
  EXPLORATORY: 9 subscales (CONDITION excluded a priori, alpha<0), BH-FDR.
  SENSITIVITY: A) exclude retention>100%  B) capped ratio, full n
               C) exclude retention>100% AND baseline ceiling (Initial=30)
               D) residualized change (ANCOVA-style), full n
  PREDICTIVE: 5-fold CV logistic/RF AUC for 90/80/70 thresholds
              + label-permutation test (empirical null) for the AUC.
  COMPENSATORY: strategy use vs baseline proficiency.
Outputs: CSV tables, figures (PNG+PDF), summary markdown.
"""
import numpy as np
import pandas as pd
from scipy import stats
import json, warnings
warnings.filterwarnings("ignore")

RNG = np.random.default_rng(42)
OUT = "study2_field/outputs"
FIG = "study2_field/figures"

# ---------------------------------------------------------------- load/clean
df = pd.read_csv("study2_field/data/study2_data.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

SUBSCALES = {
    "DET":  [f"DET_{i}" for i in range(1, 6)],
    "COG":  [f"COG_{i}" for i in range(6, 12)],
    "MEM":  [f"MEM_{i}" for i in range(12, 18)],
    "SOC":  [f"SOC_{i}" for i in range(18, 21)],
    "META": [f"META_{i}" for i in range(21, 26)],
    "SKILLS": ["RSKILL_26", "WSKILL_27", "SSKILL_28", "LSKILL_29"],
    "TOOLS": ["TOOLS_30", "TOOLS_31"],
    "CONDITION": [f"CONDITION_{i}" for i in range(32, 37)],
    "ENVIRONMENT": [f"ENVIRONMENT_{i}" for i in range(37, 43)],
    "TIME": [f"TIME_{i}" for i in range(43, 46)],
}
ITEMS = [c for cols in SUBSCALES.values() for c in cols]
EXPLORATORY = [k for k in SUBSCALES if k != "CONDITION"]  # CONDITION: alpha<0

df["ret_pct"] = df["Followup_score"] / df["Initial_score"] * 100
df["ret_cap"] = df["ret_pct"].clip(upper=100)
df["TOTAL"] = df[ITEMS].mean(axis=1)
for k, cols in SUBSCALES.items():
    df[k] = df[cols].mean(axis=1)

clean_notes = {
    "n": int(len(df)),
    "missing_gender": int(df["Gender"].isna().sum()),
    "missing_age": int(df["Age"].isna().sum()),
    "retention_gt_100_n": int((df["ret_pct"] > 100).sum()),
    "retention_max_pct": round(float(df["ret_pct"].max()), 1),
    "baseline_ceiling_n_initial_eq_30": int((df["Initial_score"] == 30).sum()),
    "sav_csv_discrepancy": "row index 85 (0-based), Retention_70: CSV=1 correct "
                           "(ratio 105.9%), SAV=0 incorrect -> fix SAV",
}

# ---------------------------------------------------------------- reliability
def cronbach_alpha(d: pd.DataFrame) -> float:
    d = d.dropna()
    k = d.shape[1]
    return k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))

def alpha_ci(d: pd.DataFrame, n_boot=800):
    d = d.dropna().values
    n = len(d)
    boots = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        s = pd.DataFrame(d[idx])
        boots.append(cronbach_alpha(s))
    return np.percentile(boots, [2.5, 97.5])

rel_rows = []
for k, cols in SUBSCALES.items():
    a = cronbach_alpha(df[cols])
    lo, hi = alpha_ci(df[cols])
    rel_rows.append({"scale": k, "k_items": len(cols), "alpha": round(a, 3),
                     "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
                     "inferential_use": "excluded" if k == "CONDITION" else "exploratory"})
a = cronbach_alpha(df[ITEMS]); lo, hi = alpha_ci(df[ITEMS])
rel_rows.append({"scale": "TOTAL(45)", "k_items": 45, "alpha": round(a, 3),
                 "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
                 "inferential_use": "primary"})
rel = pd.DataFrame(rel_rows)
rel.to_csv(f"{OUT}/table1_reliability.csv", index=False)

# ---------------------------------------------------------------- helpers
def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out

def ols_beta_p(y, X):
    """OLS with intercept; returns beta, se, p for each column of X."""
    X = np.column_stack([np.ones(len(y)), X])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta[1:], se[1:], p[1:]

# ------------------------------------------------ specifications (A,B,C,D)
specs = {}
specs["B_capped_full"] = (df, "ret_cap")
sA = df[df["ret_pct"] <= 100]
specs["A_exclude_gt100"] = (sA, "ret_pct")
sC = df[(df["ret_pct"] <= 100) & (df["Initial_score"] < 30)]
specs["C_exclude_gt100_and_ceiling"] = (sC, "ret_pct")

corr_rows = []
for spec, (d, ycol) in specs.items():
    # primary: TOTAL
    r, p = stats.pearsonr(d["TOTAL"], d[ycol])
    rs, ps = stats.spearmanr(d["TOTAL"], d[ycol])
    corr_rows.append({"spec": spec, "n": len(d), "predictor": "TOTAL",
                      "family": "primary", "pearson_r": round(r, 3),
                      "p": round(p, 4), "spearman_rho": round(rs, 3),
                      "p_spearman": round(ps, 4), "q_fdr": np.nan})
    # exploratory subscales with BH-FDR within spec
    ps_list, tmp = [], []
    for k in EXPLORATORY:
        r, p = stats.pearsonr(d[k], d[ycol])
        rs, psp = stats.spearmanr(d[k], d[ycol])
        tmp.append({"spec": spec, "n": len(d), "predictor": k,
                    "family": "exploratory", "pearson_r": round(r, 3),
                    "p": round(p, 4), "spearman_rho": round(rs, 3),
                    "p_spearman": round(psp, 4)})
        ps_list.append(p)
    qs = bh_fdr(ps_list)
    for row, q in zip(tmp, qs):
        row["q_fdr"] = round(q, 3)
        corr_rows.append(row)

# spec D: residualized change (Followup ~ Initial + predictor), full n
d = df
ps_list, tmp = [], []
for name in ["TOTAL"] + EXPLORATORY:
    X = d[["Initial_score", name]].values
    beta, se, p = ols_beta_p(d["Followup_score"].values.astype(float), X)
    row = {"spec": "D_residualized_change", "n": len(d), "predictor": name,
           "family": "primary" if name == "TOTAL" else "exploratory",
           "beta_strategy": round(beta[1], 3), "se": round(se[1], 3),
           "p": round(p[1], 4)}
    if name == "TOTAL":
        row["q_fdr"] = np.nan
        corr_rows.append(row)
    else:
        tmp.append(row); ps_list.append(p[1])
qs = bh_fdr(ps_list)
for row, q in zip(tmp, qs):
    row["q_fdr"] = round(q, 3); corr_rows.append(row)

corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{OUT}/table2_associations_all_specs.csv", index=False)

# ---------------------------------------------------- predictive null + perm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = df[ITEMS].values
cv = StratifiedKFold(5, shuffle=True, random_state=42)
ml_rows = []
for target in ["Retention_90", "Retention_80", "Retention_70"]:
    y = df[target].values
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1))
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    auc_lr = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc").mean()
    auc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc").mean()
    # permutation null (logistic pipeline; fast and standard)
    obs = max(auc_lr, auc_rf)
    null = []
    for b in range(200):
        yp = RNG.permutation(y)
        null.append(cross_val_score(lr, X, yp, cv=cv, scoring="roc_auc").mean())
    p_perm = (1 + np.sum(np.array(null) >= obs)) / (1 + len(null))
    ml_rows.append({"target": target, "auc_logreg": round(auc_lr, 3),
                    "auc_rf": round(auc_rf, 3), "best_auc": round(obs, 3),
                    "perm_null_mean": round(float(np.mean(null)), 3),
                    "perm_null_p95": round(float(np.percentile(null, 95)), 3),
                    "p_permutation": round(p_perm, 3)})
ml = pd.DataFrame(ml_rows)
ml.to_csv(f"{OUT}/table3_predictive_permutation.csv", index=False)

# ---------------------------------------------------------- compensatory
comp_rows = []
for name in ["TOTAL", "MEM", "META", "COG"]:
    r, p = stats.pearsonr(df[name], df["Initial_score"])
    comp_rows.append({"predictor": name, "outcome": "Initial_score",
                      "pearson_r": round(r, 3), "p": round(p, 4)})
comp = pd.DataFrame(comp_rows)
comp.to_csv(f"{OUT}/table4_compensatory.csv", index=False)

# ---------------------------------------------------------- power
from scipy.stats import norm
za, zb = norm.ppf(0.975), norm.ppf(0.80)
power = {f"min_detectable_r_n{n}": round(float(np.tanh((za + zb) / np.sqrt(n - 3))), 3)
         for n in [109, 88, 72]}

# ---------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

# Fig 1: retention distribution with artifact region
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(df["ret_pct"], bins=28, color="#4878a8", edgecolor="white")
ax.axvline(100, color="#c0392b", ls="--", lw=1.5)
ax.axvspan(100, df["ret_pct"].max() + 3, color="#c0392b", alpha=0.12)
ax.text(101, ax.get_ylim()[1] * 0.9,
        f">100% retention\n(n={clean_notes['retention_gt_100_n']}, "
        "measurement artifact)", color="#c0392b", fontsize=9, va="top")
ax.set_xlabel("Retention ratio (%) = Follow-up / Initial × 100")
ax.set_ylabel("Participants")
ax.set_title("Distribution of vocabulary retention (n=109)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{FIG}/fig1_retention_distribution.{ext}")
plt.close(fig)

# Fig 2: forest plot of TOTAL + subscales across specs A,B,C
fig, ax = plt.subplots(figsize=(7, 6.5))
plot_specs = ["B_capped_full", "A_exclude_gt100", "C_exclude_gt100_and_ceiling"]
labels = {"B_capped_full": "B: capped ratio, full n",
          "A_exclude_gt100": "A: exclude >100%",
          "C_exclude_gt100_and_ceiling": "C: exclude >100% & ceiling"}
colors = {"B_capped_full": "#4878a8", "A_exclude_gt100": "#e1812c",
          "C_exclude_gt100_and_ceiling": "#3a923a"}
preds = ["TOTAL"] + EXPLORATORY
ypos = np.arange(len(preds))[::-1] * 1.0
for j, spec in enumerate(plot_specs):
    sub = corr[(corr["spec"] == spec) & (corr["predictor"].isin(preds))]
    sub = sub.set_index("predictor").loc[preds]
    rvals = sub["pearson_r"].values
    ns = sub["n"].values
    ci = 1.96 / np.sqrt(ns - 3)  # Fisher-z approx CI half-width on z-scale
    zlo = np.tanh(np.arctanh(rvals) - ci)
    zhi = np.tanh(np.arctanh(rvals) + ci)
    off = (j - 1) * 0.22
    ax.errorbar(rvals, ypos + off, xerr=[rvals - zlo, zhi - rvals], fmt="o",
                ms=4, lw=1, capsize=2, color=colors[spec], label=labels[spec])
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(["TOTAL (primary)"] + EXPLORATORY)
ax.set_xlabel("Pearson r with retention (95% CI)")
ax.set_title("Strategy–retention associations across specifications")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{FIG}/fig2_forest_specifications.{ext}")
plt.close(fig)

# Fig 3: scatter TOTAL vs capped retention + compensatory inset
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
ax = axes[0]
ax.scatter(df["TOTAL"], df["ret_cap"], s=22, alpha=0.7, color="#4878a8")
m, b = np.polyfit(df["TOTAL"], df["ret_cap"], 1)
xs = np.linspace(df["TOTAL"].min(), df["TOTAL"].max(), 50)
ax.plot(xs, m * xs + b, color="#c0392b", lw=1.5)
rr, pp = stats.pearsonr(df["TOTAL"], df["ret_cap"])
ax.set_title(f"Total strategy use vs retention (r={rr:.2f}, p={pp:.2f})")
ax.set_xlabel("Total strategy use (1–5)")
ax.set_ylabel("Retention ratio, capped (%)")
ax = axes[1]
ax.scatter(df["TOTAL"], df["Initial_score"], s=22, alpha=0.7, color="#3a923a")
m, b = np.polyfit(df["TOTAL"], df["Initial_score"], 1)
ax.plot(xs, m * xs + b, color="#c0392b", lw=1.5)
rr, pp = stats.pearsonr(df["TOTAL"], df["Initial_score"])
ax.set_title(f"Compensatory check: strategies vs baseline (r={rr:.2f}, p={pp:.3f})")
ax.set_xlabel("Total strategy use (1–5)")
ax.set_ylabel("Initial vocabulary score (max 30)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"{FIG}/fig3_scatter_compensatory.{ext}")
plt.close(fig)

# ---------------------------------------------------------- summary
with open(f"{OUT}/cleaning_notes.json", "w") as f:
    json.dump({"cleaning": clean_notes, "power": power}, f, indent=2)

primary = corr[(corr["family"] == "primary")]
sig_expl = corr[(corr["family"] == "exploratory") & (corr["q_fdr"] < 0.05)]
lines = [
    "# Study 2 results summary (auto-generated)",
    f"\nN = {clean_notes['n']}; >100% retention artifacts: "
    f"{clean_notes['retention_gt_100_n']}; baseline ceiling: "
    f"{clean_notes['baseline_ceiling_n_initial_eq_30']}.",
    "\n## Primary test (TOTAL strategy use) per specification:",
    primary[["spec", "n", "pearson_r", "beta_strategy", "p"]]
        .fillna("").to_string(index=False),
    f"\n## Exploratory subscales surviving BH-FDR (q<.05): "
    f"{len(sig_expl)} of {4 * len(EXPLORATORY)} tests",
    (sig_expl[["spec", "predictor", "pearson_r", "p", "q_fdr"]]
        .to_string(index=False) if len(sig_expl) else "(none)"),
    "\n## Predictive (5-fold CV + permutation):",
    ml.to_string(index=False),
    "\n## Compensatory pattern:",
    comp.to_string(index=False),
    f"\n## Power: {power}",
]
with open(f"{OUT}/SUMMARY.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
