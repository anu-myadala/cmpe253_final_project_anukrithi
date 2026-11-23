"""Two baselines (logistic regression + LightGBM) with subgroup fairness metrics and CIs.

Protocol (config.yaml): stratified 60/20/20 train/val/test split with a fixed seed, one shared
preprocessing pipeline (median impute + standardise numerics, one-hot categoricals), a 0.5
decision threshold, and every metric reported on the untouched test split.

Outputs (results/):
  baseline_overall.csv            AUC, accuracy, balanced accuracy, Brier score per model
  fairness_<model>_<attribute>.csv per-group selection rate / TPR / FPR / precision, gaps vs. the
                                  reference group, bootstrap 95 % CIs, two-proportion z-tests
  intersectional_<model>.csv      race x sex slices (small slices flagged)
  power_note.csv                  smallest gap detectable per group at 80 % power
  roc_baselines.png, tpr_fpr_by_race.png
  run_log.json                    config snapshot, package versions, split sizes, timings

Run:  python src/baselines.py                       (uses years.train_test from config.yaml)
      python src/baselines.py --year 2024 --tag synthetic
"""
import argparse
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness import group_report, min_detectable_gap

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RESULTS = ROOT / "results"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def feature_lists():
    num = list(CFG["features"]["numeric"]) + (CFG["tract_proxies"] if CFG["include_tract_proxies"] else [])
    cat = list(CFG["features"]["categorical"])
    return num, cat


def split(df, seed):
    s = CFG["split"]
    dev, test = train_test_split(df, test_size=s["test"], stratify=df["y"], random_state=seed)
    train, val = train_test_split(dev, test_size=s["val"] / (1 - s["test"]), stratify=dev["y"], random_state=seed)
    return train, val, test


def make_models(num, cat, seed):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])
    lr = Pipeline([("pre", pre), ("clf", LogisticRegression(**CFG["models"]["logistic_regression"], random_state=seed))])
    gbm = Pipeline([("pre", pre), ("clf", lgb.LGBMClassifier(**CFG["models"]["lightgbm"], random_state=seed, verbose=-1))])
    return {"logistic_regression": lr, "lightgbm": gbm}


def choose_threshold(p_val, y_val):
    """Pick the operating point on the validation split (never on the test split)."""
    rule = CFG.get("threshold_rule", "fixed")
    if rule == "match_base_rate":
        return float(np.quantile(p_val, 1 - np.mean(y_val)))
    if rule == "youden":
        fpr, tpr, thr = roc_curve(y_val, p_val)
        return float(thr[np.argmax(tpr - fpr)])
    return float(CFG["threshold"])


def main(year, tag):
    seed = CFG["seed"]
    t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{year}.parquet")
    num, cat = feature_lists()
    X_cols = num + cat
    train, val, test = split(df, seed)
    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    overall, curves, timings, thresholds = [], {}, {}, {}
    for name, model in make_models(num, cat, seed).items():
        t1 = time.time()
        model.fit(train[X_cols], train["y"])
        timings[name] = round(time.time() - t1, 1)
        p_val = model.predict_proba(val[X_cols])[:, 1]
        thr = choose_threshold(p_val, val["y"].values)
        thresholds[name] = thr
        p = model.predict_proba(test[X_cols])[:, 1]
        yhat = (p >= thr).astype(int)
        curves[name] = roc_curve(test["y"], p)
        overall.append({
            "model": name, "n_test": len(test), "threshold": thr, "auc": roc_auc_score(test["y"], p),
            "accuracy": accuracy_score(test["y"], yhat), "balanced_accuracy": balanced_accuracy_score(test["y"], yhat),
            "brier": brier_score_loss(test["y"], p), "selection_rate": yhat.mean(), "base_rate": test["y"].mean(),
            "val_auc": roc_auc_score(val["y"], p_val),
        })
        for attr in CFG["sensitive"]:
            rep = group_report(test["y"], yhat, test[attr], CFG["reference_groups"][attr],
                               B=CFG["bootstrap_B"], seed=seed, min_n=CFG["min_group_n"])
            rep.to_csv(RESULTS / f"fairness_{name}_{attr}{suffix}.csv", index=False, float_format="%.4f")
        inter = test[CFG["intersectional"]].astype(str).agg(" | ".join, axis=1)
        ref = " | ".join(CFG["reference_groups"][a] for a in CFG["intersectional"])
        group_report(test["y"], yhat, inter, ref, B=CFG["bootstrap_B"], seed=seed, min_n=CFG["min_group_n"]) \
            .to_csv(RESULTS / f"intersectional_{name}{suffix}.csv", index=False, float_format="%.4f")

    overall = pd.DataFrame(overall)
    overall.to_csv(RESULTS / f"baseline_overall{suffix}.csv", index=False, float_format="%.4f")

    # power note: smallest selection-rate gap detectable vs. the reference group, given test-set sizes
    rows = []
    for attr in CFG["sensitive"]:
        sizes = test[attr].value_counts()
        n_ref = int(sizes.get(CFG["reference_groups"][attr], 0))
        for grp, n in sizes.items():
            if grp == CFG["reference_groups"][attr] or n_ref == 0:
                continue
            rows.append({"attribute": attr, "group": grp, "n_group": int(n), "n_reference": n_ref,
                         "min_detectable_gap_80pct_power": min_detectable_gap(n_ref, int(n), p=1 - test["y"].mean())})
    pd.DataFrame(rows).to_csv(RESULTS / f"power_note{suffix}.csv", index=False, float_format="%.4f")

    # --- plots -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for (name, (fpr, tpr, _)), color in zip(curves.items(), [BLUE, ORANGE]):
        auc = overall.loc[overall.model == name, "auc"].item()
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name.replace('_', ' ')} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], lw=1, ls="--", color="#9a9a9a", label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC on the {year} test split (n = {len(test):,})", fontsize=10)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / f"roc_baselines{suffix}.png", dpi=200); plt.close(fig)

    attr = "derived_race"
    rep = pd.read_csv(RESULTS / f"fairness_lightgbm_{attr}{suffix}.csv")
    rep = rep[~rep["small"]].sort_values("n", ascending=False)
    x = np.arange(len(rep)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - w / 2, rep["tpr"], w, color=BLUE, label="TPR (approved | originated)")
    ax.bar(x + w / 2, rep["fpr"], w, color=ORANGE, label="FPR (approved | denied)")
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={n:,})" for g, n in zip(rep["group"], rep["n"])], fontsize=7.5)
    ax.set_ylabel("Rate on the test split"); ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"LightGBM: TPR and FPR by {attr} ({year}, threshold {thresholds['lightgbm']:.3f})", fontsize=10)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / f"tpr_fpr_by_race{suffix}.png", dpi=200); plt.close(fig)

    log = {
        "year": year, "tag": tag, "seed": seed, "config": CFG, "features": X_cols, "thresholds": thresholds,
        "n": {"train": len(train), "val": len(val), "test": len(test)}, "fit_seconds": timings,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "lightgbm": lgb.__version__},
        "wall_seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS / f"run_log{suffix}.json").write_text(json.dumps(log, indent=2, default=str))
    print(overall.to_string(index=False))
    print(f"\nrace report (LightGBM):\n{rep[['group', 'n', 'selection_rate', 'tpr', 'fpr', 'precision']].to_string(index=False)}")
    print(f"\ndone in {log['wall_seconds']}s -> {RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=CFG["years"]["train_test"])
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. 'synthetic')")
    a = ap.parse_args()
    main(a.year, a.tag)
