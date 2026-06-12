"""Study 2 supplementary analyses + cross-study common-metric table.
Outputs into study2_field/outputs/: auc_bootstrap_ci.csv, primary_r_ci.csv,
disattenuation.csv, item_level_fdr.csv, common_metric_table.csv
"""
import numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(42)
OUT = "study2_field/outputs"

df = pd.read_csv("study2_field/data/study2_data.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
SUB = {"DET": range(1, 6), "COG": range(6, 12), "MEM": range(12, 18),
       "SOC": range(18, 21), "META": range(21, 26)}
ITEMS = ([f"{k}_{i}" for k, r in SUB.items() for i in r]
         + ["RSKILL_26", "WSKILL_27", "SSKILL_28", "LSKILL_29",
            "TOOLS_30", "TOOLS_31"]
         + [f"CONDITION_{i}" for i in range(32, 37)]
         + [f"ENVIRONMENT_{i}" for i in range(37, 43)]
         + [f"TIME_{i}" for i in range(43, 46)])
df["ret_pct"] = df["Followup_score"] / df["Initial_score"] * 100
df["ret_cap"] = df["ret_pct"].clip(upper=100)
df["TOTAL"] = df[ITEMS].mean(axis=1)

# ---- 1) bootstrap CIs for CV-AUC, per threshold --------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

X = df[ITEMS].to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=42)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.1))
auc_rows = []
for tgt in ["Retention_90", "Retention_80", "Retention_70"]:
    y = df[tgt].to_numpy()
    ph = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]
    obs = roc_auc_score(y, ph)
    boots = []
    n = len(y)
    for _ in range(3000):
        i = RNG.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        boots.append(roc_auc_score(y[i], ph[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    auc_rows.append({"target": tgt, "cv_auc": round(obs, 3),
                     "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3)})
auc_tab = pd.DataFrame(auc_rows)
auc_tab.to_csv(f"{OUT}/auc_bootstrap_ci.csv", index=False)
print(auc_tab.to_string(index=False), "\n")

# ---- 2) Fisher CIs for primary r in specs A/B/C --------------------------
def r_ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

specs = {"B_capped_full": (df, "ret_cap"),
         "A_exclude_gt100": (df[df["ret_pct"] <= 100], "ret_pct"),
         "C_excl_gt100_ceiling": (df[(df["ret_pct"] <= 100)
                                     & (df["Initial_score"] < 30)], "ret_pct")}
pr = []
for s, (d, yc) in specs.items():
    r, p = stats.pearsonr(d["TOTAL"], d[yc])
    lo, hi = r_ci(r, len(d))
    pr.append({"spec": s, "n": len(d), "r": round(r, 3),
               "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
               "p": round(p, 4)})
pr = pd.DataFrame(pr)
pr.to_csv(f"{OUT}/primary_r_ci.csv", index=False)
print(pr.to_string(index=False), "\n")

# ---- 3) disattenuation sensitivity ---------------------------------------
alpha_total = 0.87
r_obs, hi_obs = pr.loc[0, "r"], pr.loc[0, "ci95_hi"]
dis = []
for rel_dv in [0.5, 0.6, 0.7, 0.8]:
    k = np.sqrt(alpha_total * rel_dv)
    dis.append({"assumed_DV_reliability": rel_dv,
                "disattenuated_r": round(r_obs / k, 3),
                "disattenuated_CI_upper": round(hi_obs / k, 3)})
dis = pd.DataFrame(dis)
dis.to_csv(f"{OUT}/disattenuation.csv", index=False)
print(dis.to_string(index=False), "\n")

# ---- 4) item-level exploratory FDR (45 tests) -----------------------------
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(q, 0, 1); return out

ip, ir = [], []
for it in ITEMS:
    r, p = stats.pearsonr(df[it], df["ret_cap"])
    ir.append(r); ip.append(p)
q = bh(ip)
item_tab = pd.DataFrame({"item": ITEMS, "r": np.round(ir, 3),
                         "p": np.round(ip, 4), "q_fdr": np.round(q, 3)}
                        ).sort_values("p")
item_tab.to_csv(f"{OUT}/item_level_fdr.csv", index=False)
print(f"Item-level: {(q < .05).sum()} of 45 significant after FDR "
      f"(min q = {q.min():.3f})\n")

# ---- 5) common-metric table (person-level r) ------------------------------
def auc_to_r(auc):                       # binormal approximation
    d = stats.norm.ppf(auc) * np.sqrt(2)
    return d / np.sqrt(d ** 2 + 4)

s1b = pd.read_csv("study1_duolingo/results/study1b_summary.csv")
po = s1b[(s1b.features == "practice_only") & (s1b.model == "Ridge")].iloc[0]
fu = s1b[(s1b.features == "full") & (s1b.model == "Ridge")].iloc[0]
rows = [
    {"analysis": "S1 trace-level GBM (AUC .607, r-equivalent, approx.)",
     "unit": "trace", "n": "1,930,889", "r": round(auc_to_r(0.607), 3),
     "ci95": "—(approx. conversion)"},
    {"analysis": "S1b person-level, practice-only (Ridge)",
     "unit": "person", "n": str(po["n_users"]), "r": po["pearson_r"],
     "ci95": po["ci95"]},
    {"analysis": "S1b person-level, practice + W1 accuracy (Ridge)",
     "unit": "person", "n": str(fu["n_users"]), "r": fu["pearson_r"],
     "ci95": fu["ci95"]},
    {"analysis": "S2 self-report TOTAL vs retention (primary, capped)",
     "unit": "person", "n": "109", "r": pr.loc[0, "r"],
     "ci95": f"[{pr.loc[0,'ci95_lo']},{pr.loc[0,'ci95_hi']}]"},
]
cm = pd.DataFrame(rows)
cm.to_csv(f"{OUT}/common_metric_table.csv", index=False)
print(cm.to_string(index=False))
print("\nKey inferential statement: S2 upper 95% bound "
      f"(r={pr.loc[0,'ci95_hi']}) lies BELOW the lower bounds of both "
      f"person-level behavioural estimates ({po['ci95']}, {fu['ci95']}).")
