"""Early drift check: train on the 2024 split, score the full 2025 cohort.

Reports, for each baseline, AUC / selection rate on the 2025 cohort versus the 2024 test split
(threshold frozen from 2024 validation), the population stability index (PSI) and the
Kolmogorov-Smirnov statistic for every numeric input and for the model score, and the 2025
fairness gaps by race and ethnicity. PSI bins are the 2024 deciles; PSI < 0.10 is the usual
"stable" cut-off, 0.10-0.25 "monitor", > 0.25 "shifted".

Run:  python src/drift_check.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from baselines import choose_threshold, feature_lists, make_models, split
from fairness import group_report

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RESULTS = ROOT / "results"


def psi(expected, actual, bins=10):
    """Population stability index with quantile bins taken from the expected (2024) sample."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-6, None), np.clip(pa, 1e-6, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def main():
    seed = CFG["seed"]
    y0, y1 = CFG["years"]["train_test"], CFG["years"]["drift"]
    d0 = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{y0}.parquet")
    d1 = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{y1}.parquet")
    num, cat = feature_lists()
    X = num + cat
    train, val, test = split(d0, seed)

    # feature drift
    rows = []
    for c in num:
        a, b = test[c].dropna(), d1[c].dropna()
        rows.append({"feature": c, "psi": psi(a, b), "ks": float(stats.ks_2samp(a, b).statistic),
                     f"median_{y0}": float(a.median()), f"median_{y1}": float(b.median())})
    # model drift
    perf = []
    for name, model in make_models(num, cat, seed).items():
        model.fit(train[X], train["y"])
        thr = choose_threshold(model.predict_proba(val[X])[:, 1], val["y"].values)
        p0, p1 = model.predict_proba(test[X])[:, 1], model.predict_proba(d1[X])[:, 1]
        rows.append({"feature": f"score_{name}", "psi": psi(p0, p1), "ks": float(stats.ks_2samp(p0, p1).statistic),
                     f"median_{y0}": float(np.median(p0)), f"median_{y1}": float(np.median(p1))})
        yhat0, yhat1 = (p0 >= thr).astype(int), (p1 >= thr).astype(int)
        perf.append({"model": name, "threshold": thr,
                     f"auc_{y0}_test": roc_auc_score(test["y"], p0), f"auc_{y1}": roc_auc_score(d1["y"], p1),
                     f"selection_rate_{y0}_test": yhat0.mean(), f"selection_rate_{y1}": yhat1.mean(),
                     f"base_rate_{y0}_test": test["y"].mean(), f"base_rate_{y1}": d1["y"].mean(),
                     f"accuracy_{y0}_test": (yhat0 == test["y"]).mean(), f"accuracy_{y1}": (yhat1 == d1["y"]).mean()})
        for attr in ["derived_race", "derived_ethnicity"]:
            group_report(d1["y"], yhat1, d1[attr], CFG["reference_groups"][attr], B=CFG["bootstrap_B"], seed=seed,
                         min_n=CFG["min_group_n"]).to_csv(RESULTS / f"fairness_{name}_{attr}_{y1}.csv", index=False, float_format="%.4f")

    drift = pd.DataFrame(rows)
    drift["psi_flag"] = np.where(drift.psi < 0.10, "stable", np.where(drift.psi < 0.25, "monitor", "shifted"))
    drift.to_csv(RESULTS / f"drift_{y0}_to_{y1}.csv", index=False, float_format="%.4f")
    perf = pd.DataFrame(perf)
    perf.to_csv(RESULTS / f"drift_performance_{y0}_to_{y1}.csv", index=False, float_format="%.4f")
    print(drift.to_string(index=False)); print(); print(perf.to_string(index=False))


if __name__ == "__main__":
    main()
