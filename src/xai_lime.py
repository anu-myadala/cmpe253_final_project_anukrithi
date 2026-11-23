"""Second local explainer: LIME (Ribeiro et al. 2016) next to SHAP, on the same denied applicants.

For 100 applicants LightGBM denies on the 2024 test split:
  * LIME (lime_tabular, 5,000 samples, 10 features) is run twice with different random seeds;
    weights are summed back from one-hot columns to the original inputs
  * agreement with TreeSHAP: overlap of the top-3 inputs, Spearman rank correlation over all inputs,
    and whether both name the same principal driver (most negative contribution)
  * LIME self-stability: top-3 overlap between its two seeded runs
If two explainers disagree on why someone was denied, an adverse-action notice built on either one
is shaky; this measures how often that happens.

Outputs: results/xai_lime_agreement.csv (per applicant), results/xai_lime_summary.csv

Run:  python src/xai_lime.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from scipy import stats

from ablations import dense, make_clf
from baselines import CFG, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from explain import aggregate, input_of, shap_values

N_APPLICANTS = 100
N_SAMPLES = 5000


def lime_weights(explainer, row, predict, names, inputs, seed):
    explainer.random_state = np.random.RandomState(seed)
    exp = explainer.explain_instance(row, predict, num_features=10, num_samples=N_SAMPLES)
    w = pd.Series(0.0, index=sorted(set(input_of(n, *inputs) for n in names)))
    for idx, val in exp.as_map()[1]:
        w[input_of(names[idx], *inputs)] += val
    return w


def main():
    seed = CFG["seed"]
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat
    train, val, test = split(df, seed)
    pre = make_models(num, cat, seed)["lightgbm"].named_steps["pre"]
    Xtr = dense(pre.fit_transform(train[X])); Xva = dense(pre.transform(val[X])); Xte = dense(pre.transform(test[X]))
    names = list(pre.get_feature_names_out()); inputs = (num, cat)
    clf = make_clf("lightgbm", seed).fit(Xtr, train["y"].values)
    thr = choose_threshold(clf.predict_proba(Xva)[:, 1], val["y"].values)
    den_idx = np.flatnonzero(clf.predict_proba(Xte)[:, 1] < thr)
    pick = np.random.default_rng(seed).choice(den_idx, N_APPLICANTS, replace=False)
    phi, _ = shap_values("lightgbm", clf, Xte[pick], Xtr.mean(axis=0))
    S = aggregate(phi, names, inputs)
    cat_idx = [i for i, n in enumerate(names) if n.startswith("cat__")]
    explainer = LimeTabularExplainer(Xtr, feature_names=names, categorical_features=cat_idx, mode="classification",
                                     discretize_continuous=True, random_state=seed)
    rows = []
    for j, i in enumerate(pick):
        l1 = lime_weights(explainer, Xte[i], clf.predict_proba, names, inputs, seed)
        l2 = lime_weights(explainer, Xte[i], clf.predict_proba, names, inputs, seed + 1)
        s = S.iloc[j].reindex(l1.index).fillna(0.0)
        top = lambda v: set(v.abs().sort_values(ascending=False).index[:3])
        rows.append({"test_row": int(i), "shap_principal": s.idxmin(), "lime_principal": l1.idxmin(),
                     "same_principal_driver": s.idxmin() == l1.idxmin(),
                     "top3_overlap_shap_lime": len(top(s) & top(l1)) / 3,
                     "spearman_shap_lime": stats.spearmanr(s.values, l1.values).statistic,
                     "top3_overlap_lime_seed_a_b": len(top(l1) & top(l2)) / 3})
        if j % 20 == 0:
            print(f"{j}/{N_APPLICANTS}")
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "xai_lime_agreement.csv", index=False, float_format="%.4f")
    summ = pd.DataFrame([{"metric": c, "mean": out[c].astype(float).mean(), "sd": out[c].astype(float).std()}
                         for c in ["same_principal_driver", "top3_overlap_shap_lime", "spearman_shap_lime", "top3_overlap_lime_seed_a_b"]])
    summ.loc[len(summ)] = {"metric": "n_applicants", "mean": len(out), "sd": np.nan}
    summ.to_csv(RESULTS / "xai_lime_summary.csv", index=False, float_format="%.4f")
    print(summ.round(3).to_string(index=False))
    print(out.groupby(["shap_principal", "lime_principal"]).size().sort_values(ascending=False).head(8))


if __name__ == "__main__":
    main()
