"""Calibration by group and human-in-the-loop routing (selective prediction).

  calibration   expected calibration error (ECE, 10 equal-width bins) overall and per group, for the raw
                models and after isotonic calibration fitted on the validation split; reliability plot
  routing       the model abstains on the applications closest to its decision threshold (smallest
                |logit(p) - logit(threshold)|) and sends them to a human underwriter. For automation
                coverage 100% ... 50%: error rate and balanced accuracy on the automated part, and the
                share of each group that gets routed to a human (a fair process should not route one
                group to manual review much more often than another without a reason).

Outputs: results/human_ai_calibration.csv, results/human_ai_routing.csv,
         results/human_ai_reliability.png, results/human_ai_routing.png

Run:  python src/human_ai.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split

GROUPS = [("all", None, None), ("White", "derived_race", "White"), ("Black", "derived_race", "Black or African American"),
          ("Asian", "derived_race", "Asian"), ("Hispanic", "derived_ethnicity", "Hispanic or Latino"),
          ("Not Hispanic", "derived_ethnicity", "Not Hispanic or Latino"), ("Female", "derived_sex", "Female"),
          ("Male", "derived_sex", "Male")]
COVERAGE = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def mask(test, attr, val):
    return np.ones(len(test), bool) if attr is None else (test[attr].values == val)


def main():
    seed = CFG["seed"]
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat
    train, val, test = split(df, seed)
    y = test["y"].values
    cal_rows, route_rows, curves = [], [], {}
    for name, model in make_models(num, cat, seed).items():
        model.fit(train[X], train["y"])
        pv, pt = model.predict_proba(val[X])[:, 1], model.predict_proba(test[X])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(pv, val["y"].values)
        pt_cal = iso.predict(pt)
        for lab, attr, v in GROUPS:
            m = mask(test, attr, v)
            cal_rows.append({"model": name, "group": lab, "n": int(m.sum()), "mean_predicted": pt[m].mean(),
                             "observed_approval": y[m].mean(), "ece_raw": ece(pt[m], y[m]),
                             "ece_isotonic": ece(pt_cal[m], y[m]),
                             "calibration_in_the_large_raw": pt[m].mean() - y[m].mean()})
        curves[name] = (pt, pt_cal)
        # routing
        thr = choose_threshold(pv, val["y"].values)
        yhat = (pt >= thr).astype(int)
        margin = np.abs(logit(pt) - logit(np.array([thr]))[0])
        order = np.argsort(-margin)                                # most confident first
        for cov in COVERAGE:
            k = int(round(cov * len(test)))
            auto = np.zeros(len(test), bool); auto[order[:k]] = True
            r = {"model": name, "coverage": cov, "error_rate_automated": float((yhat[auto] != y[auto]).mean()),
                 "balanced_acc_automated": balanced_accuracy_score(y[auto], yhat[auto]),
                 "denials_automated": int(((yhat == 0) & auto).sum()), "denials_total": int((yhat == 0).sum())}
            for lab, attr, v in GROUPS[1:]:
                m = mask(test, attr, v)
                r[f"routed_share_{lab}"] = float((~auto[m]).mean())
            route_rows.append(r)
    cal = pd.DataFrame(cal_rows); cal.to_csv(RESULTS / "human_ai_calibration.csv", index=False, float_format="%.4f")
    rt = pd.DataFrame(route_rows); rt.to_csv(RESULTS / "human_ai_routing.csv", index=False, float_format="%.4f")
    print(cal.round(4).to_string(index=False)); print(rt.round(4).to_string(index=False))

    # reliability plot (LightGBM, raw vs isotonic) by race
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4))
    for ax, (name, (pt, pc)) in zip(axes, curves.items()):
        for lab, attr, v, col in [("White", "derived_race", "White", BLUE), ("Black", "derived_race", "Black or African American", ORANGE),
                                  ("Hispanic", "derived_ethnicity", "Hispanic or Latino", "#2e9e5b")]:
            m = mask(test, attr, v)
            q = np.quantile(pt[m], np.linspace(0, 1, 11)); q[0], q[-1] = 0, 1
            b = np.clip(np.digitize(pt[m], q[1:-1]), 0, 9)
            xs = [pt[m][b == i].mean() for i in range(10) if (b == i).any()]
            ys = [y[m][b == i].mean() for i in range(10) if (b == i).any()]
            ax.plot(xs, ys, marker="o", color=col, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", color="#8a8a8a", lw=1)
        ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1)
        ax.set_xlabel("predicted approval probability (decile bins)"); ax.set_ylabel("observed approval rate")
        ax.set_title(f"Reliability by group: {name.replace('_', ' ')}", fontsize=10); ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "human_ai_reliability.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for name, col in [("logistic_regression", BLUE), ("lightgbm", ORANGE)]:
        d = rt[rt.model == name]
        axes[0].plot(d.coverage * 100, d.error_rate_automated * 100, marker="o", color=col, label=name.replace("_", " "))
    axes[0].invert_xaxis(); axes[0].set_xlabel("share of applications decided automatically (%)")
    axes[0].set_ylabel("error rate on automated decisions (%)"); axes[0].set_title("Selective prediction", fontsize=10)
    d = rt[rt.model == "lightgbm"]
    for lab, col in [("White", BLUE), ("Black", ORANGE), ("Hispanic", "#2e9e5b"), ("Asian", "#8a5cc7")]:
        axes[1].plot(d.coverage * 100, d[f"routed_share_{lab}"] * 100, marker="o", color=col, label=lab)
    axes[1].invert_xaxis(); axes[1].set_xlabel("share decided automatically (%)"); axes[1].set_ylabel("routed to a human (%)")
    axes[1].set_title("Who gets routed to manual review (LightGBM)", fontsize=10)
    for ax in axes:
        ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "human_ai_routing.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
