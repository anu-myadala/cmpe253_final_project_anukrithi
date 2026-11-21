"""Two extra checks added in November (week 14):

  1. adverse-action reason coverage (risk R5): for every test applicant the model denies, take the
     two inputs with the most negative SHAP contribution and look up their Reg. B status in
     explain.ADVERSE_ACTION. Reports the share of denials whose principal driver maps to a
     standard sample-form reason, and the share where both top-2 drivers do.
  2. proxy recoverability (risk R1): how well can the model inputs alone predict an applicant's
     race / ethnicity? LightGBM classifier, Black vs White and Hispanic vs not Hispanic, with the
     census-tract proxies off and on. Higher AUC = the inputs carry more race information, which is
     why "fairness through unawareness" (dropping race) does not remove the gap.

Outputs: results/reason_coverage.csv, results/reason_coverage_drivers.csv,
         results/proxy_recoverability.csv

Run:  python src/extras.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ablations import dense, make_clf
from baselines import CFG, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from explain import ADVERSE_ACTION, aggregate, shap_values


def reason_coverage(train, val, test, num, cat, seed):
    X_cols = num + cat
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr = dense(pre.fit_transform(train[X_cols])); Xva = dense(pre.transform(val[X_cols])); Xte = dense(pre.transform(test[X_cols]))
    names = list(pre.get_feature_names_out())
    rows, drivers = [], []
    for model_name in ["logistic_regression", "lightgbm"]:
        clf = make_clf(model_name, seed).fit(Xtr, train["y"].values)
        thr = choose_threshold(clf.predict_proba(Xva)[:, 1], val["y"].values)
        denied = clf.predict_proba(Xte)[:, 1] < thr
        phi, _ = shap_values(model_name, clf, Xte[denied], Xtr.mean(axis=0))
        phi = aggregate(phi, names, (num, cat))
        top2 = np.argsort(phi.values, axis=1)[:, :2]                  # two most negative contributions
        first = phi.columns[top2[:, 0]]; second = phi.columns[top2[:, 1]]
        status = lambda c: ADVERSE_ACTION.get(c, ("", "unknown"))[1]
        s1 = np.array([status(c) for c in first]); s2 = np.array([status(c) for c in second])
        rows.append({"model": model_name, "n_denied_by_model": int(denied.sum()),
                     "principal_driver_standard_reason": float(np.mean(s1 == "standard")),
                     "both_top2_standard_reason": float(np.mean((s1 == "standard") & (s2 == "standard"))),
                     "principal_driver_process_input": float(np.mean(s1 == "process"))})
        vc = pd.Series(first).value_counts(normalize=True)
        for inp, share in vc.items():
            drivers.append({"model": model_name, "principal_driver": inp, "share_of_denials": share,
                            "reg_b_status": ADVERSE_ACTION.get(inp, ("", "unknown"))[1],
                            "reason": ADVERSE_ACTION.get(inp, ("", ""))[0]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "reason_coverage.csv", index=False, float_format="%.4f")
    pd.DataFrame(drivers).to_csv(RESULTS / "reason_coverage_drivers.csv", index=False, float_format="%.4f")


def proxy_recoverability(train, test, num, cat, seed):
    rows = []
    for proxies in (False, True):
        n_ = num + (list(CFG["tract_proxies"]) if proxies else [])
        for attr, grp, ref, lab in [("derived_race", "Black or African American", "White", "Black vs White"),
                                    ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino",
                                     "Hispanic vs not Hispanic")]:
            tr = train[train[attr].isin([grp, ref])]; te = test[test[attr].isin([grp, ref])]
            m = make_models(n_, cat, seed)["lightgbm"].fit(tr[n_ + cat], (tr[attr] == grp).astype(int))
            auc = roc_auc_score((te[attr] == grp).astype(int), m.predict_proba(te[n_ + cat])[:, 1])
            rows.append({"target": lab, "tract_proxies": "on" if proxies else "off", "recover_auc": auc,
                         "n_test": len(te)})
            print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "proxy_recoverability.csv", index=False, float_format="%.4f")


def main():
    seed = CFG["seed"]
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists()
    train, val, test = split(df, seed)
    reason_coverage(train, val, test, num, cat, seed)
    proxy_recoverability(train, test, num, cat, seed)


if __name__ == "__main__":
    main()
