"""Fairness ablations on the two baselines (Track B: >= 2 baselines + ablations, statistical rigor).

Five variants, each fitted on the SAME training split and scored on the SAME untouched test split,
so every comparison is paired:

  baseline        the two models from baselines.py, re-fitted here (identical numbers)
  reweighing      Kamiran & Calders (2012): training weights w(g, y) = P(g) P(y) / P(g, y) make the
                  label independent of the group in the weighted training set (pre-processing)
  eo_postprocess  Hardt et al. (2016): group-specific thresholds fitted on the VALIDATION split
                  (fairlearn ThresholdOptimizer, equalized odds). Needs the group at decision time,
                  so in US lending it is an upper bound on what post-processing can buy, not a
                  deployable option (ECOA / Regulation B disparate-treatment rule)
  reductions      Agarwal et al. (2018): exponentiated-gradient reduction under a demographic-parity
                  constraint (fairlearn ExponentiatedGradient). The group is used during training
                  only; the result is a randomised classifier, scored as the method defines it
  reductions_rethreshold  the same randomised classifier's P(yhat = 1) re-cut at the base-rate-matching
                  threshold, so its gap can be read at the same selection rate as the other variants
  proxy_features  baseline inputs + the four census-tract proxies from config.yaml (how much
                  disparity geography re-introduces)

Groups below min_group_n in the training split are merged into one "Other" bucket for the
interventions (never for the evaluation, which keeps the original labels).
The decision threshold of every score-producing variant is chosen on the validation split with the
rule in config.yaml (match_base_rate), as in baselines.py. Gaps come with bootstrap 95 % CIs and
two-proportion z-tests (fairness.group_report); variants are compared with the baseline by
McNemar's test on paired correct/incorrect test predictions.

Outputs (results/):
  ablation_summary.csv                 one row per (model, variant): AUC, accuracy, balanced accuracy,
                                       selection rate, gaps + CIs for Black, Hispanic, Female, McNemar p
  ablation_<model>_<variant>_<attr>.csv full group reports (also intersectional race x sex)
  ablation_mcnemar.csv                  paired tests, including LR vs LightGBM
  ablation_gaps.png                     Black-White selection-rate and TPR gaps with CIs per variant
Run:  python src/ablations.py                 (group attribute from config: ablations.group_attr)
      python src/ablations.py --group-attr derived_ethnicity

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from fairness import group_report

warnings.filterwarnings("ignore", category=FutureWarning)
AB = CFG.get("ablations", {})
GROUP_ATTR = AB.get("group_attr", "derived_race")
OTHER = AB.get("collapse_small_groups_to", "Other")
RED = AB.get("reductions", {"constraint": "demographic_parity", "eps": 0.01, "max_iter": 50})
EO = AB.get("eo_postprocess", {"constraint": "equalized_odds", "objective": "balanced_accuracy_score"})
VARIANTS = ["baseline", "reweighing", "eo_postprocess", "reductions", "reductions_rethreshold", "proxy_features"]
# the gaps pulled into the summary table: (attribute, group, reference)
HEADLINE = [("derived_race", "Black or African American", "Black"), ("derived_race", "Asian", "Asian"),
            ("derived_ethnicity", "Hispanic or Latino", "Hispanic"), ("derived_sex", "Female", "Female")]


def make_clf(name, seed):
    if name == "logistic_regression":
        return LogisticRegression(**CFG["models"]["logistic_regression"], random_state=seed)
    return lgb.LGBMClassifier(**CFG["models"]["lightgbm"], random_state=seed, verbose=-1)


def dense(M):
    return np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)


def collapse_small(series, train_series, min_n):
    keep = train_series.value_counts()
    keep = keep[keep >= min_n].index
    return series.where(series.isin(keep), OTHER).astype(str)


def reweighing_weights(g, y):
    """Kamiran & Calders: w(g, y) = P(g) P(y) / P(g, y), computed on the training split."""
    df = pd.DataFrame({"g": np.asarray(g), "y": np.asarray(y)})
    p_g = df["g"].value_counts(normalize=True)
    p_y = df["y"].value_counts(normalize=True)
    p_gy = df.groupby(["g", "y"]).size() / len(df)
    table = {(gg, yy): p_g[gg] * p_y[yy] / p_gy[(gg, yy)] for gg, yy in p_gy.index}
    return np.array([table[(gg, yy)] for gg, yy in zip(df["g"], df["y"])])


def mcnemar(y, yhat_a, yhat_b):
    """McNemar's test on paired predictions: b = A right & B wrong, c = A wrong & B right."""
    y, a_ok, b_ok = np.asarray(y), np.asarray(yhat_a) == np.asarray(y), np.asarray(yhat_b) == np.asarray(y)
    b = int((a_ok & ~b_ok).sum()); c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return b, c, float("nan"), 1.0, "none"
    if b + c < 25:
        return b, c, float("nan"), float(stats.binomtest(min(b, c), b + c, 0.5).pvalue), "exact"
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2), float(1 - stats.chi2.cdf(chi2, 1)), "chi2_cc"


def overall(y, p, yhat):
    return {"auc": roc_auc_score(y, p) if p is not None else float("nan"),
            "accuracy": accuracy_score(y, yhat), "balanced_accuracy": balanced_accuracy_score(y, yhat),
            "selection_rate": float(np.mean(yhat)), "base_rate": float(np.mean(y))}


def main(group_attr):
    seed = CFG["seed"]; t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X_cols = num + cat
    num_px = num + CFG["tract_proxies"]
    train, val, test = split(df, seed)
    min_n = CFG["min_group_n"]
    g_tr = collapse_small(train[group_attr].astype(str), train[group_attr].astype(str), min_n)
    g_va = collapse_small(val[group_attr].astype(str), train[group_attr].astype(str), min_n)
    g_te = collapse_small(test[group_attr].astype(str), train[group_attr].astype(str), min_n)
    suffix = "" if group_attr == "derived_race" else f"_{group_attr}"

    # one preprocessing fit per feature set, shared by every variant of a model
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr, Xva, Xte = (dense(M) for M in (pre.fit_transform(train[X_cols]), pre.transform(val[X_cols]), pre.transform(test[X_cols])))
    pre_px = make_models(num_px, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr_px, Xva_px, Xte_px = (dense(M) for M in (pre_px.fit_transform(train[num_px + cat]), pre_px.transform(val[num_px + cat]), pre_px.transform(test[num_px + cat])))
    ytr, yva, yte = train["y"].values, val["y"].values, test["y"].values
    w_rw = reweighing_weights(g_tr, ytr)

    preds, summary, timings = {}, [], {}
    for model_name in ["logistic_regression", "lightgbm"]:
        for variant in VARIANTS:
            t1 = time.time(); p = None; thr = float("nan")
            if variant == "baseline":
                clf = make_clf(model_name, seed).fit(Xtr, ytr)
                base_clf = clf
                thr = choose_threshold(clf.predict_proba(Xva)[:, 1], yva)
                p = clf.predict_proba(Xte)[:, 1]; yhat = (p >= thr).astype(int)
            elif variant == "reweighing":
                clf = make_clf(model_name, seed).fit(Xtr, ytr, sample_weight=w_rw)
                thr = choose_threshold(clf.predict_proba(Xva)[:, 1], yva)
                p = clf.predict_proba(Xte)[:, 1]; yhat = (p >= thr).astype(int)
            elif variant == "eo_postprocess":
                to = ThresholdOptimizer(estimator=base_clf, constraints=EO["constraint"], objective=EO["objective"],
                                        prefit=True, predict_method="predict_proba")
                to.fit(Xva, yva, sensitive_features=g_va)
                yhat = np.asarray(to.predict(Xte, sensitive_features=g_te, random_state=seed)).astype(int)
            elif variant == "reductions":
                cons = DemographicParity(difference_bound=RED["eps"]) if RED["constraint"] == "demographic_parity" \
                    else EqualizedOdds(difference_bound=RED["eps"])
                eg = ExponentiatedGradient(make_clf(model_name, seed), constraints=cons, eps=RED["eps"], max_iter=RED["max_iter"])
                eg.fit(Xtr, ytr, sensitive_features=g_tr)
                p = np.asarray(eg._pmf_predict(Xte))[:, 1]          # P(yhat = 1) under the randomised classifier
                yhat = np.asarray(eg.predict(Xte, random_state=seed)).astype(int)
                timings[f"{model_name}/{variant}/oracle_calls"] = int(eg.n_oracle_calls_)
                p_va_red, p_te_red = np.asarray(eg._pmf_predict(Xva))[:, 1], p
            elif variant == "reductions_rethreshold":
                # same randomised classifier, but its P(yhat = 1) re-cut at the base-rate-matching threshold,
                # so the gap can be read at the SAME selection rate as the other variants (the DP guarantee
                # holds only for the method's own decisions above, not for this re-cut)
                thr = choose_threshold(p_va_red, yva)
                p = p_te_red; yhat = (p >= thr).astype(int)
            else:  # proxy_features
                clf = make_clf(model_name, seed).fit(Xtr_px, ytr)
                thr = choose_threshold(clf.predict_proba(Xva_px)[:, 1], yva)
                p = clf.predict_proba(Xte_px)[:, 1]; yhat = (p >= thr).astype(int)
            timings[f"{model_name}/{variant}"] = round(time.time() - t1, 1)
            preds[(model_name, variant)] = (p, yhat)

            row = {"model": model_name, "variant": variant, "group_attr_used": group_attr if variant not in ("baseline", "proxy_features") else "",
                   "uses_group_in_training": variant in ("reweighing", "reductions", "reductions_rethreshold"),
                   "uses_group_at_decision": variant == "eo_postprocess", "threshold": thr}
            row.update(overall(yte, p, yhat))
            reports = {}
            for attr in CFG["sensitive"]:
                rep = group_report(yte, yhat, test[attr], CFG["reference_groups"][attr], B=CFG["bootstrap_B"], seed=seed, min_n=min_n)
                rep.to_csv(RESULTS / f"ablation_{model_name}_{variant}_{attr}{suffix}.csv", index=False, float_format="%.4f")
                reports[attr] = rep.set_index("group")
            inter = test[CFG["intersectional"]].astype(str).agg(" | ".join, axis=1)
            ref = " | ".join(CFG["reference_groups"][a] for a in CFG["intersectional"])
            group_report(yte, yhat, inter, ref, B=CFG["bootstrap_B"], seed=seed, min_n=min_n) \
                .to_csv(RESULTS / f"ablation_{model_name}_{variant}_intersectional{suffix}.csv", index=False, float_format="%.4f")
            for attr, grp, short in HEADLINE:
                r = reports[attr].loc[grp]
                for m in ["selection_rate", "tpr", "fpr"]:
                    row[f"gap_{m}_{short}"] = r[f"gap_{m}"]; row[f"ci_lo_{m}_{short}"] = r[f"ci_lo_{m}"]
                    row[f"ci_hi_{m}_{short}"] = r[f"ci_hi_{m}"]; row[f"p_{m}_{short}"] = r[f"p_{m}"]
                row[f"air_{short}"] = r["adverse_impact_ratio"]
            # paired comparison with this model's baseline
            b, c, chi2, pv, kind = mcnemar(yte, preds[(model_name, "baseline")][1], yhat)
            row.update({"mcnemar_b": b, "mcnemar_c": c, "mcnemar_p_vs_baseline": pv, "fit_seconds": timings[f"{model_name}/{variant}"]})
            summary.append(row)
            print(f"{model_name:20s} {variant:15s} auc {row['auc']:.4f} acc {row['accuracy']:.4f} bal {row['balanced_accuracy']:.4f} "
                  f"sel {row['selection_rate']:.4f} | Black sel gap {row['gap_selection_rate_Black']:+.4f} "
                  f"[{row['ci_lo_selection_rate_Black']:+.4f}, {row['ci_hi_selection_rate_Black']:+.4f}] tpr gap {row['gap_tpr_Black']:+.4f} "
                  f"| Hispanic sel gap {row['gap_selection_rate_Hispanic']:+.4f} | McNemar p {pv:.3g}  ({timings[f'{model_name}/{variant}']}s)")

    summary = pd.DataFrame(summary)
    base = summary[summary.variant == "baseline"].set_index("model")
    summary["delta_accuracy_vs_baseline"] = summary.apply(lambda r: r["accuracy"] - base.loc[r["model"], "accuracy"], axis=1)
    summary["delta_balanced_accuracy_vs_baseline"] = summary.apply(lambda r: r["balanced_accuracy"] - base.loc[r["model"], "balanced_accuracy"], axis=1)
    summary.to_csv(RESULTS / f"ablation_summary{suffix}.csv", index=False, float_format="%.4f")

    # McNemar table: every variant vs its baseline, plus LR vs LightGBM at baseline
    rows = []
    for (m, v), (_, yhat) in preds.items():
        if v == "baseline":
            continue
        b, c, chi2, pv, kind = mcnemar(yte, preds[(m, "baseline")][1], yhat)
        rows.append({"model": m, "a": "baseline", "b": v, "b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "chi2": chi2, "p": pv, "test": kind})
    b, c, chi2, pv, kind = mcnemar(yte, preds[("logistic_regression", "baseline")][1], preds[("lightgbm", "baseline")][1])
    rows.append({"model": "both", "a": "logistic_regression", "b": "lightgbm", "b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "chi2": chi2, "p": pv, "test": kind})
    pd.DataFrame(rows).to_csv(RESULTS / f"ablation_mcnemar{suffix}.csv", index=False, float_format="%.4g")

    # figure: Black-White gaps (selection rate, TPR) with CIs, per variant and model
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    ylab = [v.replace("_", " ") for v in VARIANTS]
    for ax, m, title in zip(axes, ["selection_rate", "tpr"], ["Selection-rate gap, Black vs White (pp)", "TPR gap, Black vs White (pp)"]):
        for j, (model_name, color, off) in enumerate([("logistic_regression", BLUE, -0.16), ("lightgbm", ORANGE, 0.16)]):
            s = summary[summary.model == model_name].set_index("variant").loc[VARIANTS]
            yy = np.arange(len(VARIANTS)) + off
            g = 100 * s[f"gap_{m}_Black"]; lo = 100 * (s[f"gap_{m}_Black"] - s[f"ci_lo_{m}_Black"]); hi = 100 * (s[f"ci_hi_{m}_Black"] - s[f"gap_{m}_Black"])
            ax.errorbar(g, yy, xerr=[lo, hi], fmt="o", color=color, capsize=3, label=model_name.replace("_", " "))
        ax.axvline(0, color="#777", lw=1, ls="--"); ax.set_yticks(np.arange(len(VARIANTS))); ax.set_yticklabels(ylab)
        ax.invert_yaxis(); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Fairness interventions on {group_attr} - test split, bootstrap 95 % CIs", fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / f"ablation_gaps{suffix}.png", dpi=200); plt.close(fig)

    log = {"group_attr": group_attr, "seed": seed, "variants": VARIANTS, "reductions": RED, "eo_postprocess": EO,
           "collapsed_groups": sorted(set(train[group_attr].astype(str)) - set(g_tr)), "timings": timings,
           "wall_seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS / f"run_log_ablations{suffix}.json").write_text(json.dumps(log, indent=2, default=str))
    print(f"\ndone in {log['wall_seconds']}s -> results/ablation_summary{suffix}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-attr", default=GROUP_ATTR)
    main(ap.parse_args().group_attr)
