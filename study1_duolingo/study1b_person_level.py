"""Person-level prospective analysis (Study 1b).
Unit of analysis = the learner (as in Study 2). Behavioural features from
week 1 predict mean recall in week 2, 5-fold CV over users.
Two feature sets:
  practice-only : volume/spacing/coverage (no week-1 accuracy)
  full          : practice-only + week-1 accuracy
Outputs: results/study1b_user_table.csv (features), study1b_summary.csv,
         study1b_fig_scatter.png
"""
import numpy as np, pandas as pd, glob, json
from scipy import stats

RESULTS = "study1_duolingo/results"
SEED = 42

parts = sorted(glob.glob("data/duolingo_sample/duolingo_sample_part*.csv.gz"))
df = pd.concat([pd.read_csv(p, dtype={"user_id": str}) for p in parts],
               ignore_index=True)
t0, t1 = df["timestamp"].min(), df["timestamp"].max()
mid = t0 + (t1 - t0) / 2
df["week"] = np.where(df["timestamp"] < mid, 1, 2)
df["delta_days"] = (df["delta"] / 86400.0).clip(lower=1 / 86400)
df["log_lag"] = np.log10(df["delta_days"])
df["hist_rate"] = df["history_correct"] / df["history_seen"].clip(lower=1)

w1, w2 = df[df["week"] == 1], df[df["week"] == 2]
g1 = w1.groupby("user_id")
feat = pd.DataFrame({
    "n_traces_w1": g1.size(),
    "active_days_w1": g1["timestamp"].apply(
        lambda s: pd.to_datetime(s, unit="s").dt.date.nunique()),
    "mean_log_lag": g1["log_lag"].mean(),
    "sd_log_lag": g1["log_lag"].std().fillna(0),
    "prop_long_lag": g1["delta_days"].apply(lambda s: (s > 1).mean()),
    "distinct_lexemes": g1["lexeme_id"].nunique(),
    "mean_session_seen": g1["session_seen"].mean(),
    "mean_hist_rate_w1": g1["hist_rate"].mean(),
    "acc_w1": g1["p_recall"].mean(),          # week-1 accuracy (full model)
    "lang": g1["learning_language"].agg(lambda s: s.mode().iat[0]),
})
out2 = w2.groupby("user_id").agg(n_traces_w2=("p_recall", "size"),
                                 recall_w2=("p_recall", "mean"))
tab = feat.join(out2, how="inner")
tab = tab[(tab["n_traces_w1"] >= 10) & (tab["n_traces_w2"] >= 5)].copy()
tab = pd.get_dummies(tab, columns=["lang"], dtype=np.float32)
tab.to_csv(f"{RESULTS}/study1b_user_table.csv")
print(f"Users in both weeks meeting thresholds: n={len(tab):,} "
      f"(of {df['user_id'].nunique():,} total)")

PRACTICE = (["n_traces_w1", "active_days_w1", "mean_log_lag", "sd_log_lag",
             "prop_long_lag", "distinct_lexemes", "mean_session_seen"]
            + [c for c in tab.columns if c.startswith("lang_")])
FULL = PRACTICE + ["mean_hist_rate_w1", "acc_w1"]

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

y = tab["recall_w2"].to_numpy()
cv = KFold(5, shuffle=True, random_state=SEED)
rows = []
keep_pred = {}
for fname, feats in [("practice_only", PRACTICE), ("full", FULL)]:
    X = tab[feats].to_numpy(np.float64)
    for mname, model in [
        ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("GBM", HistGradientBoostingRegressor(max_iter=300,
                                              random_state=SEED)),
    ]:
        ph = cross_val_predict(model, X, y, cv=cv)
        r, p = stats.pearsonr(y, ph)
        z = np.arctanh(r); se = 1 / np.sqrt(len(y) - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        rows.append({"features": fname, "model": mname, "n_users": len(y),
                     "pearson_r": round(r, 3), "ci95": f"[{lo:.3f},{hi:.3f}]",
                     "p": f"{p:.2e}",
                     "R2": round(1 - np.var(y - ph) / np.var(y), 3),
                     "mae": round(np.abs(y - ph).mean(), 4)})
        keep_pred[(fname, mname)] = ph

res = pd.DataFrame(rows)
res.to_csv(f"{RESULTS}/study1b_summary.csv", index=False)
print(res.to_string(index=False))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "figure.dpi": 300})
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, key, ttl in [
    (axes[0], ("practice_only", "GBM"), "Practice-only features"),
    (axes[1], ("full", "GBM"), "Practice + week-1 accuracy"),
]:
    ph = keep_pred[key]
    r = stats.pearsonr(y, ph)[0]
    ax.scatter(ph, y, s=4, alpha=0.15, color="#4878a8", rasterized=True)
    ax.plot([y.min(), 1], [y.min(), 1], "--", color="grey", lw=1)
    ax.set_title(f"{ttl}  (r={r:.2f})")
    ax.set_xlabel("Predicted week-2 recall")
axes[0].set_ylabel("Observed week-2 recall")
fig.suptitle(f"Study 1b: person-level prospective prediction "
             f"(n={len(y):,} users, 5-fold CV)", y=1.02)
fig.tight_layout()
fig.savefig(f"{RESULTS}/study1b_fig_scatter.png", bbox_inches="tight")
plt.close(fig)

with open(f"{RESULTS}/study1b_design.json", "w") as f:
    json.dump({"unit": "user (matches Study 2)",
               "design": "prospective: week-1 features -> week-2 mean recall",
               "inclusion": ">=10 W1 traces and >=5 W2 traces",
               "selection_caveat": "users active in both weeks "
                                   "(engagement-survival selection)",
               "cv": "KFold(5) over users"}, f, indent=2)
print("Study 1b complete.")
