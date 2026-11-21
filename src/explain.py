"""Explainability for the two baselines: SHAP global + local, stability, faithfulness, sanity check,
and a decomposition of the Black-White / Hispanic-non-Hispanic score gap by input (Track B: local +
global explanations with stability and faithfulness checks).

Attribution:  LightGBM - SHAP TreeExplainer (exact Shapley values for trees, log-odds units)
              logistic regression - exact linear Shapley values phi_j = w_j (x_j - mean_j), the
              closed form behind SHAP's LinearExplainer, in the same log-odds units
One-hot columns are summed back to their original input, so every table is per application field.

Checks:
  stability     global importances re-computed on 5 bootstrap resamples of the training split
                (LightGBM also re-seeded): Spearman rho with the reference ranking + top-5 overlap
  faithfulness  deletion test - replace the top-k inputs (by global |SHAP|) with their training
                median/mode and re-score; the AUC drop is compared with k random inputs (5 draws).
                Local version: replace each row's own top-3 inputs vs 3 random inputs, mean |delta log-odds|
  sanity        label-shuffle check - a model trained on permuted labels should score ~0.5 AUC and
                its importance ranking should not correlate with the real one
  gap decomposition  mean phi_j over Black test rows minus over White rows (sums to the mean log-odds
                difference between the groups): which inputs carry the disparity

Outputs (results/): explain_global_<model>.csv, explain_global.png, explain_local_examples.csv,
  explain_stability.csv, explain_faithfulness.csv, explain_sanity.csv, explain_gap_decomposition.csv,
  explain_gap_decomposition.png, run_log_explain.json
Run:  python src/explain.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import roc_auc_score

from ablations import dense, make_clf
from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split

warnings.filterwarnings("ignore")
N_BOOT = 5
TOP_K = [1, 2, 3, 5]
# Regulation B (12 CFR 1002, Appendix C, sample form C-1) adverse-action reasons an input maps to.
# "process" inputs have no standard reason and should not be driving a denial on their own.
ADVERSE_ACTION = {
    "income": ("Income insufficient for amount of credit requested", "standard"),
    "debt_to_income_ratio": ("Excessive obligations in relation to income", "standard"),
    "combined_loan_to_value_ratio": ("Value or type of collateral not sufficient / insufficient down payment", "standard"),
    "property_value": ("Value or type of collateral not sufficient", "standard"),
    "loan_amount": ("Value or type of collateral not sufficient (amount requested vs collateral)", "standard"),
    "loan_term": ("No standard reason code - loan term requested; explain in plain language", "non-standard"),
    "conforming_loan_limit": ("No standard reason code - amount outside the conforming limit", "non-standard"),
    "applicant_credit_score_type": ("Credit-history related (scoring model used) - the score itself must be disclosed, the model type is not a reason", "non-standard"),
    "co-applicant_credit_score_type": ("Credit-history related (co-applicant scoring model used) - not itself a reason", "non-standard"),
    "co_applicant_present": ("No standard reason - presence of a co-applicant; review before use", "process"),
    "aus-1": ("Automated underwriting system used - the AUS findings, not the system name, must be translated into reasons", "process"),
    "submission_of_application": ("Process attribute (direct vs broker submission) - should not drive a denial", "process"),
    "initially_payable_to_institution": ("Process attribute (loan initially payable to the institution) - should not drive a denial", "process"),
    "preapproval": ("Process attribute (pre-approval requested) - should not drive a denial", "process"),
}


def input_of(feature_name, num, cat):
    """Map a transformed column name back to its original input."""
    if feature_name.startswith("num__"):
        return feature_name[5:]
    rest = feature_name[5:]
    for c in sorted(cat, key=len, reverse=True):
        if rest.startswith(c + "_") or rest == c:
            return c
    return rest


def aggregate(phi, names, inputs):
    """Sum SHAP columns back to original inputs -> DataFrame (rows x inputs)."""
    out = pd.DataFrame(phi, columns=names).T.groupby([input_of(n, *inputs) for n in names]).sum().T
    return out


def shap_values(model_name, clf, X, Xtr_mean):
    if model_name == "lightgbm":
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        return np.asarray(sv), float(np.ravel(ex.expected_value)[-1])
    w = clf.coef_.ravel()
    return (X - Xtr_mean) * w, float(clf.intercept_[0] + Xtr_mean @ w)


def global_importance(phi_df):
    s = phi_df.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"input": s.index, "mean_abs_shap": s.values, "rank": np.arange(1, len(s) + 1),
                         "share": s.values / s.values.sum()})


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed); t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X_cols = num + cat
    train, val, test = split(df, seed)
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr = dense(pre.fit_transform(train[X_cols])); Xva = dense(pre.transform(val[X_cols])); Xte = dense(pre.transform(test[X_cols]))
    names = list(pre.get_feature_names_out()); inputs = (num, cat)
    ytr, yva, yte = train["y"].values, val["y"].values, test["y"].values
    Xtr_mean = Xtr.mean(axis=0)
    fill = {c: (train[c].median() if c in num else train[c].mode().iloc[0]) for c in X_cols}   # deletion baseline values

    log = {"seed": seed, "n_test": int(len(test)), "timings": {}}
    glob, local_rows, stab_rows, faith_rows, sanity_rows, decomp = {}, [], [], [], [], {}
    for model_name in ["logistic_regression", "lightgbm"]:
        t1 = time.time()
        clf = make_clf(model_name, seed).fit(Xtr, ytr)
        thr = choose_threshold(clf.predict_proba(Xva)[:, 1], yva)
        p_te = clf.predict_proba(Xte)[:, 1]; yhat = (p_te >= thr).astype(int)
        auc_ref = roc_auc_score(yte, p_te)
        phi, base_value = shap_values(model_name, clf, Xte, Xtr_mean)
        phi_df = aggregate(phi, names, inputs)
        g = global_importance(phi_df); glob[model_name] = g
        g.to_csv(RESULTS / f"explain_global_{model_name}.csv", index=False, float_format="%.5f")
        log["timings"][f"{model_name}/shap"] = round(time.time() - t1, 1)

        # ---- local explanations for three model-denied applicants ----------------------------
        denied = np.flatnonzero(yhat == 0)
        picks = {"lowest score": denied[np.argmin(p_te[denied])],
                 "borderline (just below threshold)": denied[np.argmax(p_te[denied])],
                 "random denied": rng.choice(denied)}
        for label, i in picks.items():
            contrib = phi_df.iloc[i].sort_values()
            for rank, (inp, val_) in enumerate(contrib.head(4).items(), 1):
                reason, status = ADVERSE_ACTION.get(inp, ("No mapping", "unknown"))
                local_rows.append({"model": model_name, "case": label, "row_in_test": int(i), "score": float(p_te[i]),
                                   "threshold": thr, "actual_outcome": "originated" if yte[i] == 1 else "denied",
                                   "rank": rank, "input": inp, "value": str(test[inp].iloc[i]), "shap_log_odds": float(val_),
                                   "adverse_action_reason": reason, "reason_status": status})

        # ---- stability over bootstrap resamples of the training split ---------------------------
        # (importances of the re-fitted models are computed on a fixed 5,000-row test sample for speed;
        #  the reference ranking is compared on the same sample so the comparison is like for like)
        ref_rank = g.set_index("input")["mean_abs_shap"]
        sample = rng.choice(len(Xte), 5000, replace=False)
        ref_rank_sample = global_importance(phi_df.iloc[sample]).set_index("input")["mean_abs_shap"].reindex(ref_rank.index)
        for b in range(N_BOOT):
            idx = rng.choice(len(Xtr), len(Xtr), replace=True)
            clf_b = make_clf(model_name, seed + 1 + b).fit(Xtr[idx], ytr[idx])
            phi_b, _ = shap_values(model_name, clf_b, Xte[sample], Xtr[idx].mean(axis=0))
            g_b = global_importance(aggregate(phi_b, names, inputs)).set_index("input")["mean_abs_shap"].reindex(ref_rank.index)
            rho = stats.spearmanr(ref_rank_sample.values, g_b.values).statistic
            top5 = len(set(ref_rank_sample.sort_values(ascending=False).head(5).index) & set(g_b.sort_values(ascending=False).head(5).index)) / 5
            stab_rows.append({"model": model_name, "resample": b + 1, "spearman_rho_vs_reference": rho, "top5_overlap": top5,
                              "test_auc": roc_auc_score(yte, clf_b.predict_proba(Xte)[:, 1])})

        # ---- faithfulness: deletion test on the SAME fitted model ----------------------------------
        order = list(g["input"])
        for k in TOP_K:
            def auc_without(cols):
                Xd = test[X_cols].copy()
                for c in cols:
                    Xd[c] = fill[c]
                return roc_auc_score(yte, clf.predict_proba(dense(pre.transform(Xd)))[:, 1])
            auc_top = auc_without(order[:k])
            auc_rand = [auc_without(list(rng.choice(X_cols, k, replace=False))) for _ in range(5)]
            faith_rows.append({"model": model_name, "k": k, "auc_full": auc_ref, "auc_without_top_k": auc_top,
                               "drop_top_k": auc_ref - auc_top, "auc_without_random_k_mean": float(np.mean(auc_rand)),
                               "drop_random_k_mean": auc_ref - float(np.mean(auc_rand)), "top_k_inputs": "; ".join(order[:k])})
        # local faithfulness on 2,000 test rows: remove the row's own top-3 vs 3 random inputs
        sub = rng.choice(len(test), 2000, replace=False)
        logit = lambda p: np.log(p / (1 - p))
        base_logit = logit(np.clip(p_te[sub], 1e-9, 1 - 1e-9))
        Xd_top = test[X_cols].iloc[sub].copy(); Xd_rand = test[X_cols].iloc[sub].copy()
        abs_phi = phi_df.iloc[sub].abs()[X_cols].values
        top3 = np.argsort(-abs_phi, axis=1)[:, :3]                       # each row's own three largest |phi|
        rand3 = np.argsort(rng.random(abs_phi.shape), axis=1)[:, :3]     # three random inputs per row
        for j, c in enumerate(X_cols):                                   # column-wise masked replacement
            m_top = (top3 == j).any(axis=1); m_rand = (rand3 == j).any(axis=1)
            Xd_top.loc[Xd_top.index[m_top], c] = fill[c]; Xd_rand.loc[Xd_rand.index[m_rand], c] = fill[c]
        d_top = np.abs(logit(np.clip(clf.predict_proba(dense(pre.transform(Xd_top)))[:, 1], 1e-9, 1 - 1e-9)) - base_logit).mean()
        d_rand = np.abs(logit(np.clip(clf.predict_proba(dense(pre.transform(Xd_rand)))[:, 1], 1e-9, 1 - 1e-9)) - base_logit).mean()
        faith_rows.append({"model": model_name, "k": "local-3", "auc_full": auc_ref, "auc_without_top_k": float("nan"),
                           "drop_top_k": float(d_top), "auc_without_random_k_mean": float("nan"), "drop_random_k_mean": float(d_rand),
                           "top_k_inputs": "mean |delta log-odds| when the row's own top-3 inputs (vs 3 random inputs) are set to the training median/mode"})

        # ---- sanity: label shuffle (3 permutations; a single draw can land well away from 0.5) --------
        aucs, masses, rhos = [], [], []
        for r_ in range(3):
            y_perm = rng.permutation(ytr)
            clf_s = make_clf(model_name, seed + r_).fit(Xtr, y_perm)
            phi_s, _ = shap_values(model_name, clf_s, Xte[sample], Xtr_mean)
            g_s = global_importance(aggregate(phi_s, names, inputs)).set_index("input")["mean_abs_shap"].reindex(ref_rank.index)
            aucs.append(roc_auc_score(yte, clf_s.predict_proba(Xte)[:, 1])); masses.append(float(g_s.sum()))
            rhos.append(stats.spearmanr(ref_rank_sample.values, g_s.values).statistic)
        sanity_rows.append({"model": model_name, "auc_real_labels": auc_ref, "auc_shuffled_labels_mean": float(np.mean(aucs)), "auc_shuffled_labels_sd": float(np.std(aucs, ddof=1)),
                            "total_mean_abs_shap_real": float(ref_rank_sample.sum()), "total_mean_abs_shap_shuffled_mean": float(np.mean(masses)),
                            "spearman_rho_shuffled_vs_real_mean": float(np.mean(rhos)), "n_permutations": 3})

        # ---- gap decomposition --------------------------------------------------------------------
        for attr, grp, ref in [("derived_race", "Black or African American", "White"), ("derived_race", "Asian", "White"),
                               ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino")]:
            a, b = (test[attr] == grp).values, (test[attr] == ref).values
            diff = (phi_df[a].mean() - phi_df[b].mean()).sort_values()
            decomp[(model_name, grp)] = diff
            logit_gap = float(np.mean(np.log(np.clip(p_te[a], 1e-9, 1) / np.clip(1 - p_te[a], 1e-9, 1))) - np.mean(np.log(np.clip(p_te[b], 1e-9, 1) / np.clip(1 - p_te[b], 1e-9, 1))))
            decomp[(model_name, grp, "meta")] = {"mean_logit_gap": logit_gap, "sum_of_shap_diffs": float(diff.sum()),
                                                 "mean_score_group": float(p_te[a].mean()), "mean_score_reference": float(p_te[b].mean()), "n_group": int(a.sum()), "n_reference": int(b.sum())}
        log["timings"][model_name] = round(time.time() - t1, 1)
        print(f"{model_name}: AUC {auc_ref:.4f}; top inputs: {', '.join(order[:5])}")

    pd.DataFrame(local_rows).to_csv(RESULTS / "explain_local_examples.csv", index=False, float_format="%.4f")
    st = pd.DataFrame(stab_rows); st.to_csv(RESULTS / "explain_stability.csv", index=False, float_format="%.4f")
    pd.DataFrame(faith_rows).to_csv(RESULTS / "explain_faithfulness.csv", index=False, float_format="%.4f")
    pd.DataFrame(sanity_rows).to_csv(RESULTS / "explain_sanity.csv", index=False, float_format="%.4f")
    rows = []
    for key, diff in decomp.items():
        if len(key) == 3:
            continue
        meta = decomp[key + ("meta",)]
        for inp, v in diff.items():
            rows.append({"model": key[0], "group": key[1], "input": inp, "mean_shap_diff_log_odds": float(v),
                         "share_of_total_gap": float(v / meta["sum_of_shap_diffs"]) if meta["sum_of_shap_diffs"] else float("nan"), **meta})
    dec = pd.DataFrame(rows); dec.to_csv(RESULTS / "explain_gap_decomposition.csv", index=False, float_format="%.5f")

    # ---- figures ------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (model_name, color) in zip(axes, [("logistic_regression", BLUE), ("lightgbm", ORANGE)]):
        g = glob[model_name].head(12)[::-1]
        ax.barh(g["input"], g["mean_abs_shap"], color=color)
        ax.set_title(f"{model_name.replace('_', ' ')}: mean |SHAP| (log-odds), test split", fontsize=10)
        ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "explain_global.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (model_name, color) in zip(axes, [("logistic_regression", BLUE), ("lightgbm", ORANGE)]):
        d = decomp[(model_name, "Black or African American")]
        d = d.reindex(d.abs().sort_values(ascending=False).index).head(10)[::-1]
        ax.barh(d.index, d.values, color=[color if v < 0 else "#7bb36a" for v in d.values])
        ax.axvline(0, color="#777", lw=1)
        meta = decomp[(model_name, "Black or African American", "meta")]
        ax.set_title(f"{model_name.replace('_', ' ')}: Black minus White mean SHAP\n(total log-odds gap {meta['mean_logit_gap']:+.3f})", fontsize=10)
        ax.tick_params(axis="y", labelsize=8); ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "explain_gap_decomposition.png", dpi=200); plt.close(fig)

    log["wall_seconds"] = round(time.time() - t0, 1); log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / "run_log_explain.json").write_text(json.dumps(log, indent=2, default=str))
    print(st.groupby("model")[["spearman_rho_vs_reference", "top5_overlap"]].mean().to_string())
    print(pd.DataFrame(sanity_rows).to_string(index=False))
    print(f"done in {log['wall_seconds']}s")


if __name__ == "__main__":
    main()
