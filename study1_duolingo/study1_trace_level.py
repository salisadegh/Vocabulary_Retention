"""Trace-level analysis: predicting session recall from recorded practice
behaviour (Settles & Meeder 2016 dataset; deterministic user-complete
subsample, see make_sample.py and manifest).

Run from the repository root:
    python study1_duolingo/study1_trace_level.py

Pipeline: load sample parts -> feature engineering -> user-level GroupKFold
(no learner appears in both train and test) -> models (global-mean baseline;
half-life regression with nested hyperparameter selection on an inner
user-level holdout; ridge; histogram gradient boosting) -> metrics, lag
sensitivity, permutation importance, calibration. Long runs checkpoint
per fold and can resume.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

RESULTS = "study1_duolingo/results"
SEED = 42
GRID = [(0.001, 3), (0.003, 8), (0.01, 8)]  # HLR (learning rate, epochs)

USECOLS = ["p_recall", "timestamp", "delta", "user_id", "learning_language",
           "ui_language", "lexeme_id", "history_seen", "history_correct",
           "session_seen", "session_correct"]
DTYPES = {"p_recall": "float32", "delta": "float64",
          "history_seen": "int32", "history_correct": "int32",
          "session_seen": "int16", "session_correct": "int16",
          "user_id": "category", "learning_language": "category",
          "ui_language": "category", "lexeme_id": "category"}


def load(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No data found at {pattern}. "
                         "Run download_data.sh and make_sample.py first.")
    df = pd.concat([pd.read_csv(p, usecols=USECOLS, dtype=DTYPES)
                    for p in paths], ignore_index=True)
    print(f"loaded {len(df):,} traces, {df['user_id'].nunique():,} users")
    return df


def featurize(df):
    df = df.copy()
    df["delta_days"] = (df["delta"] / 86400.0).clip(lower=1 / 86400)
    df["wrong"] = (df["history_seen"] - df["history_correct"]).clip(lower=0)
    df["sqrt_seen"] = np.sqrt(df["history_seen"])
    df["sqrt_correct"] = np.sqrt(df["history_correct"])
    df["sqrt_wrong"] = np.sqrt(df["wrong"])
    df["hist_rate"] = df["history_correct"] / df["history_seen"].clip(lower=1)
    df["p_recall"] = df["p_recall"].clip(0.0001, 0.9999)
    return df


class HLR:
    """Half-life regression: h = 2^(theta.x), p = 2^(-delta_days / h),
    trained by minibatch SGD on squared error with L2 regularisation.
    Exponent clipping keeps larger learning rates numerically stable."""
    FEATS = ["sqrt_correct", "sqrt_wrong"]

    def __init__(self, lrate=0.01, l2=0.1, epochs=8, seed=SEED,
                 hmin=15 / 1440, hmax=274):
        self.lrate, self.l2, self.epochs = lrate, l2, epochs
        self.hmin, self.hmax = hmin, hmax
        self.rng = np.random.default_rng(seed)
        self.theta = None

    def _h(self, X):
        return np.clip(2.0 ** np.clip(X @ self.theta, -7, 9),
                       self.hmin, self.hmax)

    def fit(self, F, d, p):
        X = np.column_stack([F.astype(np.float64), np.ones(len(F))])
        self.theta = np.zeros(X.shape[1])
        n = len(X)
        for _ in range(self.epochs):
            idx = self.rng.permutation(n)
            for s in range(0, n, 4096):
                b = idx[s:s + 4096]
                h = self._h(X[b])
                ph = np.clip(2.0 ** (-d[b] / h), 1e-4, 1 - 1e-4)
                dph = ph * np.log(2) * d[b] / h ** 2
                dh = (h * np.log(2))[:, None] * X[b]
                g = ((-2 * (p[b] - ph) * dph)[:, None] * dh).mean(axis=0) \
                    + self.l2 * self.theta / n
                self.theta -= self.lrate * g
        return self

    def predict(self, F, d):
        X = np.column_stack([F.astype(np.float64), np.ones(len(F))])
        return np.clip(2.0 ** (-d / self._h(X)), 0, 1)


def run_fold(fold, X, y, perfect, groups, F, d, cols, n_splits):
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from scipy.stats import spearmanr

    tr, te = list(GroupKFold(n_splits).split(X, y, groups))[fold]
    print(f"[fold {fold}] n_train={len(tr):,} n_test={len(te):,}", flush=True)

    itr_i, ival_i = next(GroupShuffleSplit(
        n_splits=1, test_size=0.10, random_state=SEED + fold
    ).split(tr, groups=groups[tr]))
    itr, ival = tr[itr_i], tr[ival_i]
    best, best_mae = None, np.inf
    for lr, ep in GRID:
        m = HLR(lrate=lr, epochs=ep).fit(F[itr], d[itr].astype(np.float64),
                                         y[itr].astype(np.float64))
        mae = np.abs(m.predict(F[ival], d[ival].astype(np.float64))
                     - y[ival]).mean()
        if mae < best_mae:
            best_mae, best = mae, (lr, ep)
    hlr = HLR(lrate=best[0], epochs=best[1]).fit(
        F[tr], d[tr].astype(np.float64), y[tr].astype(np.float64))

    preds = {"baseline_global_mean": np.full(len(te), float(y[tr].mean())),
             "HLR_tuned": hlr.predict(F[te], d[te].astype(np.float64))}
    ridge = Ridge(alpha=1.0).fit(X[tr], y[tr])
    preds["Ridge_linear"] = np.clip(ridge.predict(X[te]), 0, 1)
    gbm = HistGradientBoostingRegressor(max_iter=200, early_stopping=False,
                                        random_state=SEED).fit(X[tr], y[tr])
    preds["GBM"] = np.clip(gbm.predict(X[te]), 0, 1)

    def rows_for(mask, suffix=""):
        out = []
        for name, ph in preds.items():
            if suffix and not name.startswith(("GBM", "baseline")):
                continue
            yy, pp, pf = y[te][mask], ph[mask], perfect[te][mask]
            auc = (roc_auc_score(pf, pp)
                   if len(np.unique(pf)) > 1 else np.nan)
            sp = spearmanr(yy, pp).statistic if np.std(pp) > 0 else np.nan
            out.append({"fold": fold, "model": name + suffix,
                        "mae": mean_absolute_error(yy, pp), "spearman": sp,
                        "auc_perfect_recall": auc, "n_test": int(mask.sum()),
                        "hlr_config": f"lr={best[0]},ep={best[1]}"})
        return out

    rows = rows_for(np.ones(len(te), bool))
    rows += rows_for(d[te] >= (300 / 86400), suffix="__lag>=5min")
    pd.DataFrame(rows).to_csv(f"{RESULTS}/trace_fold{fold}_metrics.csv",
                              index=False)

    rng = np.random.default_rng(SEED + fold)
    keep = rng.choice(len(te), size=min(50_000, len(te)), replace=False)
    np.savez_compressed(f"{RESULTS}/trace_fold{fold}_preds.npz",
                        obs=y[te][keep].astype(np.float32),
                        gbm=preds["GBM"][keep].astype(np.float32),
                        perfect=perfect[te][keep])
    sub = rng.choice(te, size=min(60_000, len(te)), replace=False)
    pi = permutation_importance(gbm, X[sub], y[sub], n_repeats=5,
                                random_state=SEED,
                                scoring="neg_mean_absolute_error")
    pd.Series(pi.importances_mean, index=cols
              ).to_csv(f"{RESULTS}/trace_fold{fold}_importance.csv")
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)


def finish(n_splits, n_rows, n_users):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 300})

    res = pd.concat([pd.read_csv(f"{RESULTS}/trace_fold{f}_metrics.csv")
                     for f in range(n_splits)], ignore_index=True)
    res.to_csv(f"{RESULTS}/trace_fold_metrics_all.csv", index=False)
    agg = (res.groupby("model")[["mae", "spearman", "auc_perfect_recall"]]
              .agg(["mean", "std"]).round(4))
    agg.to_csv(f"{RESULTS}/trace_summary.csv")

    imps = [pd.read_csv(f"{RESULTS}/trace_fold{f}_importance.csv",
                        index_col=0).iloc[:, 0] for f in range(n_splits)]
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    imp.to_csv(f"{RESULTS}/trace_gbm_importance.csv")

    Z = [np.load(f"{RESULTS}/trace_fold{f}_preds.npz")
         for f in range(n_splits)]
    obs = np.concatenate([z["obs"] for z in Z])
    pred = np.concatenate([z["gbm"] for z in Z])
    bins = np.quantile(pred, np.linspace(0, 1, 11))
    bins[0], bins[-1] = 0, 1
    ix = np.clip(np.digitize(pred, bins) - 1, 0, 9)
    bp = [pred[ix == k].mean() for k in range(10)]
    bo = [obs[ix == k].mean() for k in range(10)]
    ece = float(np.mean(np.abs(np.array(bp) - np.array(bo))))
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0.6, 1.0], [0.6, 1.0], "--", color="grey", lw=1)
    ax.plot(bp, bo, "o-", color="#4878a8")
    ax.set_xlabel("Predicted recall (GBM, decile bins)")
    ax.set_ylabel("Observed recall")
    ax.set_title(f"Calibration, pooled user-level CV (ECE={ece:.3f})")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig_calibration.png")
    plt.close(fig)

    with open(f"{RESULTS}/provenance.json", "w") as f:
        json.dump({"source": "Harvard Dataverse doi:10.7910/DVN/N8XJME",
                   "paper": "Settles & Meeder (2016), ACL, P16-1174",
                   "rows_used": n_rows, "n_users": n_users,
                   "sample": "deterministic user-complete 15% "
                             "(md5(user_id)%100<15); see manifest",
                   "split": "GroupKFold by user_id (no user leakage)",
                   "hlr_tuning": "nested user-level inner holdout",
                   "ece_gbm": ece}, f, indent=2)
    print(agg.to_string())
    print(f"\nECE (GBM): {ece:.4f}")
    print("\nTop predictors:\n", imp.head(8).round(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/duolingo_sample/"
                                      "duolingo_sample_part*.csv.gz")
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--fold", type=int, default=None,
                    help="run a single fold (checkpoint/resume); "
                         "omit to run everything")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    df = featurize(load(args.data))
    gbm_feats = ["delta_days", "history_seen", "history_correct", "wrong",
                 "hist_rate", "sqrt_seen", "sqrt_correct", "sqrt_wrong"]
    Xg = pd.get_dummies(df[gbm_feats + ["learning_language", "ui_language"]],
                        columns=["learning_language", "ui_language"]
                        ).astype(np.float32)
    X = Xg.to_numpy()
    cols = list(Xg.columns)
    y = df["p_recall"].to_numpy(np.float32)
    perfect = (df["session_correct"] == df["session_seen"]
               ).to_numpy(np.int8)
    groups = pd.factorize(df["user_id"])[0].astype(np.int32)
    F = df[HLR.FEATS].to_numpy(np.float32)
    d = df["delta_days"].to_numpy(np.float32)
    n_rows, n_users = int(len(df)), int(df["user_id"].nunique())
    del df, Xg

    folds = [args.fold] if args.fold is not None else range(args.splits)
    for f in folds:
        run_fold(f, X, y, perfect, groups, F, d, cols, args.splits)
    if args.fold is None:
        finish(args.splits, n_rows, n_users)
