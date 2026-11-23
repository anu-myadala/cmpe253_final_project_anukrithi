"""Statistical-rigor and evaluation-design checks added in week 14.

  1. repeated splits   the whole train/val/test protocol re-run with 10 different split seeds;
                       mean and 95% t-interval for AUC, balanced accuracy and the key fairness gaps
                       (the Track B deck asks for "mean +/- CI across multiple seeds, not a single run")
  2. DeLong test       LR vs LightGBM AUC on the main test split (paired, DeLong et al. 1988), plus
                       DeLong 95% CIs for each AUC
  3. multiple testing  every per-group z-test p-value in results/fairness_*.csv (groups n >= 200)
                       adjusted with Holm (family-wise) and Benjamini-Hochberg (false discovery rate)
  4. lender holdout    group-aware split: 20% of lenders (LEI) held out entirely, so the test set has
                       no lender seen in training; AUC and gaps vs. the random split
  5. duplicates        exact duplicate rows inside the cohort and across the train / test split
  6. manifest          SHA-256 of the raw and cleaned data, config hash, package versions

Outputs: results/rigor_repeated_splits.csv, rigor_repeated_summary.csv, rigor_delong.csv,
         rigor_multiple_testing.csv, rigor_lender_holdout.csv, rigor_duplicates.csv,
         results/manifest.json

Run:  python src/rigor.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import glob
import hashlib
import json
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from baselines import CFG, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from fairness import counts, fairness_metrics

SEEDS = list(range(10))
KEY = [("derived_race", "Black or African American", "White", "Black"),
       ("derived_race", "Asian", "White", "Asian"),
       ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino", "Hispanic"),
       ("derived_sex", "Female", "Male", "Female")]


def gaps(test, yhat):
    out = {}
    y = test["y"].values
    for attr, grp, ref, k in KEY:
        a = test[attr].values == grp; b = test[attr].values == ref
        ma, mb = fairness_metrics(*counts(y[a], yhat[a])), fairness_metrics(*counts(y[b], yhat[b]))
        out[f"{k}_sel_gap"] = ma["selection_rate"] - mb["selection_rate"]
        out[f"{k}_tpr_gap"] = ma["tpr"] - mb["tpr"]
    return out


def gap_ci(test, yhat, B=500, seed=0):
    """Bootstrap 95% CIs (resampling rows within each group) for the selection and TPR gaps."""
    rng = np.random.default_rng(seed)
    y = test["y"].values; out = {}
    for attr, grp, ref, k in KEY:
        ia = np.flatnonzero(test[attr].values == grp); ib = np.flatnonzero(test[attr].values == ref)
        bs = {"sel": [], "tpr": []}
        for _ in range(B):
            a = rng.choice(ia, len(ia)); b = rng.choice(ib, len(ib))
            ma, mb = fairness_metrics(*counts(y[a], yhat[a])), fairness_metrics(*counts(y[b], yhat[b]))
            bs["sel"].append(ma["selection_rate"] - mb["selection_rate"]); bs["tpr"].append(ma["tpr"] - mb["tpr"])
        for m in ("sel", "tpr"):
            out[f"{k}_{m}_gap_lo"], out[f"{k}_{m}_gap_hi"] = np.nanpercentile(bs[m], [2.5, 97.5])
    return out


def fit_eval(train, val, test, num, cat, seed):
    X = num + cat
    rows = []
    for name, model in make_models(num, cat, seed).items():
        model.fit(train[X], train["y"])
        thr = choose_threshold(model.predict_proba(val[X])[:, 1], val["y"].values)
        p = model.predict_proba(test[X])[:, 1]; yhat = (p >= thr).astype(int)
        r = {"model": name, "auc": roc_auc_score(test["y"], p),
             "balanced_accuracy": balanced_accuracy_score(test["y"], yhat), "selection_rate": yhat.mean()}
        r.update(gaps(test, yhat)); rows.append(r)
    return rows


# ---- DeLong (fast version, Sun & Xu 2014) ------------------------------------------------
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, preds):
    """AUCs and their covariance for several score vectors on the same rows."""
    y = np.asarray(y); pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[r]) for r in range(k)]); ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.r_[pos[r], neg[r]]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p); order = np.argsort(p)[::-1]; m = len(p); adj = np.empty(m); run = 1.0
    for rank, i in zip(range(m, 0, -1), order):
        run = min(run, p[i] * m / rank); adj[i] = run
    return adj


def sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat

    # 1. repeated splits ------------------------------------------------------------------------
    rows = []
    for s in SEEDS:
        dev, test = train_test_split(df, test_size=CFG["split"]["test"], stratify=df["y"], random_state=s)
        train, val = train_test_split(dev, test_size=CFG["split"]["val"] / (1 - CFG["split"]["test"]),
                                      stratify=dev["y"], random_state=s)
        for r in fit_eval(train, val, test, num, cat, s):
            r["split_seed"] = s; rows.append(r)
        print(f"seed {s} done")
    rep = pd.DataFrame(rows); rep.to_csv(RESULTS / "rigor_repeated_splits.csv", index=False, float_format="%.5f")
    summ = []
    metrics = [c for c in rep.columns if c not in ("model", "split_seed")]
    for m, g in rep.groupby("model"):
        for c in metrics:
            v = g[c].values; half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            summ.append({"model": m, "metric": c, "mean": v.mean(), "sd": v.std(ddof=1), "ci_lo": v.mean() - half,
                         "ci_hi": v.mean() + half, "min": v.min(), "max": v.max(), "n_seeds": len(v)})
    summ = pd.DataFrame(summ); summ.to_csv(RESULTS / "rigor_repeated_summary.csv", index=False, float_format="%.5f")
    print(summ[summ.metric.isin(["auc", "Black_sel_gap", "Hispanic_sel_gap"])].round(4).to_string(index=False))
    # paired difference LGBM - LR across seeds
    piv = rep.pivot(index="split_seed", columns="model", values="auc")
    d = piv["lightgbm"] - piv["logistic_regression"]
    t = stats.ttest_rel(piv["lightgbm"], piv["logistic_regression"])

    # 2. DeLong on the main split ------------------------------------------------------------------
    train, val, test = split(df, CFG["seed"])
    P = []
    for name, model in make_models(num, cat, CFG["seed"]).items():
        model.fit(train[X], train["y"]); P.append(model.predict_proba(test[X])[:, 1])
    aucs, cov = delong(test["y"].values, np.vstack(P))
    se = np.sqrt(np.diag(cov))
    diff = aucs[1] - aucs[0]; se_d = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = diff / se_d; p = 2 * (1 - stats.norm.cdf(abs(z)))
    dl = pd.DataFrame([
        {"comparison": "AUC logistic_regression", "estimate": aucs[0], "ci_lo": aucs[0] - 1.96 * se[0], "ci_hi": aucs[0] + 1.96 * se[0]},
        {"comparison": "AUC lightgbm", "estimate": aucs[1], "ci_lo": aucs[1] - 1.96 * se[1], "ci_hi": aucs[1] + 1.96 * se[1]},
        {"comparison": "AUC lightgbm - logistic_regression (DeLong)", "estimate": diff, "ci_lo": diff - 1.96 * se_d,
         "ci_hi": diff + 1.96 * se_d, "z": z, "p": p},
        {"comparison": "AUC lightgbm - logistic_regression (10 split seeds, paired t)", "estimate": d.mean(),
         "ci_lo": d.mean() - stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)),
         "ci_hi": d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)), "z": t.statistic, "p": t.pvalue},
    ])
    dl.to_csv(RESULTS / "rigor_delong.csv", index=False, float_format="%.6g")
    print(dl.to_string(index=False))

    # 3. multiple testing -------------------------------------------------------------------------
    rows = []
    for f in sorted(glob.glob(str(RESULTS / "fairness_*_*.csv"))):
        name = Path(f).stem
        if name.endswith("_2025"):
            continue
        rep_ = pd.read_csv(f)
        rep_ = rep_[(~rep_["small"].astype(bool)) & rep_["group"].ne(rep_["reference"])]
        for _, r in rep_.iterrows():
            for m in ["selection_rate", "tpr", "fpr", "precision"]:
                if pd.notna(r.get(f"p_{m}")):
                    rows.append({"table": name, "group": r["group"], "metric": m, "gap": r[f"gap_{m}"], "p_raw": r[f"p_{m}"]})
    mt = pd.DataFrame(rows)
    mt["p_holm"] = holm(mt.p_raw.values); mt["p_bh"] = bh(mt.p_raw.values)
    mt["sig_raw"] = mt.p_raw < 0.05; mt["sig_holm"] = mt.p_holm < 0.05; mt["sig_bh"] = mt.p_bh < 0.05
    mt.to_csv(RESULTS / "rigor_multiple_testing.csv", index=False, float_format="%.6g")
    print(f"multiple testing: {len(mt)} tests; significant raw {mt.sig_raw.sum()}, BH {mt.sig_bh.sum()}, Holm {mt.sig_holm.sum()}")
    keep = mt[mt.table.str.contains("lightgbm") & mt.metric.eq("selection_rate")]
    print(keep[["table", "group", "gap", "p_raw", "p_holm", "p_bh"]].round(5).to_string(index=False))

    # 4. lender holdout ------------------------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG["seed"])
    dev_i, test_i = next(gss.split(df, groups=df["lei"]))
    dev, test_g = df.iloc[dev_i], df.iloc[test_i]
    train_g, val_g = train_test_split(dev, test_size=0.25, stratify=dev["y"], random_state=CFG["seed"])
    lh = []
    for r in fit_eval(train_g, val_g, test_g, num, cat, CFG["seed"]):
        r.update({"split": "lender holdout (unseen LEIs)", "n_test": len(test_g), "n_test_lenders": test_g["lei"].nunique()}); lh.append(r)
    for r in fit_eval(train, val, test, num, cat, CFG["seed"]):
        r.update({"split": "random (main)", "n_test": len(test), "n_test_lenders": test["lei"].nunique()}); lh.append(r)
    lh = pd.DataFrame(lh)
    lh["share_test_lenders_seen_in_train"] = np.where(lh.split.str.startswith("random"),
                                                      test["lei"].isin(set(train["lei"])).mean(), 0.0)
    lh.to_csv(RESULTS / "rigor_lender_holdout.csv", index=False, float_format="%.5f")
    print(lh[["split", "model", "auc", "balanced_accuracy", "Black_sel_gap", "Hispanic_sel_gap", "n_test"]].round(4).to_string(index=False))

    # 5. duplicates -----------------------------------------------------------------------------------
    cols = X + ["y"]
    dup_in = int(df.duplicated(subset=cols).sum())
    key = lambda d: pd.util.hash_pandas_object(d[cols].astype(object).astype(str), index=False)
    cross = int(key(test).isin(set(key(train))).sum())
    pd.DataFrame([{"check": "exact duplicate rows (model inputs + label) in the 2024 cohort", "count": dup_in, "share": dup_in / len(df)},
                  {"check": "test rows whose inputs + label also appear in train", "count": cross, "share": cross / len(test)}]
                 ).to_csv(RESULTS / "rigor_duplicates.csv", index=False, float_format="%.6f")
    print(f"duplicates in cohort {dup_in}; test rows also in train {cross}")

    # 6. manifest -----------------------------------------------------------------------------------------
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "*.csv"))) + sorted(glob.glob(str(ROOT / "data" / "clean" / "*.parquet")))
    man = {"data_sha256": {Path(f).name: sha(f) for f in files if "SYNTHETIC" not in f},
           "config_sha256": sha(ROOT / "config.yaml"), "seed": CFG["seed"],
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__, "scipy": scipy.__version__}}
    (RESULTS / "manifest.json").write_text(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
