"""Robustness hardening of LightGBM (fixes the noise brittleness found in robustness.py).

Diagnosis (robustness_noise_by_input.csv): LightGBM's flips come from inputs with a spike at one
value — loan_term (97.8% of rows are exactly 360 months) and CLTV (24% exactly 80%). The tree model
learned "exactly 360 / exactly 80" as signals, so tiny noise that moves a value off the spike changes
the decision. Two things are fixed here:

  1. a domain-aware perturbation test (the SD-unit test in robustness.py moves loan_term to values
     like 357.3 months, which cannot happen): income and appraised property value each mis-stated by
     up to +/-5% (uniform), CLTV recomputed from the new property value, loan term untouched
  2. hardened LightGBM variants, all trained on the same split and judged on the same test set:
       H1  monotone constraints: approval can only rise with income and fall with CLTV
       H2  H1 + coarser bins (max_bin 63), 31 leaves, min_child_samples 200, L2 = 5
       H3  H2 + noise augmentation: a second copy of the training set with realistic perturbations
     Reported: AUC (with DeLong p vs. the baseline), balanced accuracy, key gaps, flip rate under
     SD-unit noise (0.05, 0.10) and under the realistic perturbation, and gaming success (income
     +10% with the DTI bin held fixed).

Outputs: results/hardening.csv, results/hardening.png

Run:  python src/hardening.py

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from rigor import delong, gap_ci, gaps

N_DRAWS = 3


def dense(M):
    return np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)


def realistic(df, rng):
    e = df.copy()
    e["income"] = e["income"] * rng.uniform(0.95, 1.05, len(e))
    f = rng.uniform(0.95, 1.05, len(e))
    e["combined_loan_to_value_ratio"] = e["combined_loan_to_value_ratio"] / f
    e["property_value"] = e["property_value"] * f
    return e


def sd_noise(df, num, sd, s, rng):
    e = df.copy()
    for c in num:
        e[c] = e[c] + rng.normal(0, s * sd[c], len(e))
    return e


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed)
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat
    train, val, test = split(df, seed)
    pre = clone(make_models(num, cat, seed)["lightgbm"].named_steps["pre"]).fit(train[X])
    n_feat = len(pre.get_feature_names_out())
    mono = [0] * n_feat
    mono[num.index("income")] = 1
    mono[num.index("combined_loan_to_value_ratio")] = -1
    base_params = dict(CFG["models"]["lightgbm"], random_state=seed, verbose=-1)
    reg = dict(max_bin=63, num_leaves=31, min_child_samples=200, reg_lambda=5.0)
    variants = {
        "baseline": (base_params, False),
        "H1 monotone": (dict(base_params, monotone_constraints=mono), False),
        "H2 monotone + regularised": (dict(base_params, monotone_constraints=mono, **reg), False),
        "H3 H2 + noise augmentation": (dict(base_params, monotone_constraints=mono, **reg), True),
    }
    sd = train[num].std()
    Xva, Xte = dense(pre.transform(val[X])), dense(pre.transform(test[X]))
    rows, scores = [], {}
    for name, (params, augment) in variants.items():
        tr = pd.concat([train, realistic(train, rng)]) if augment else train
        clf = lgb.LGBMClassifier(**params).fit(dense(pre.transform(tr[X])), tr["y"].values)
        thr = choose_threshold(clf.predict_proba(Xva)[:, 1], val["y"].values)
        p = clf.predict_proba(Xte)[:, 1]; yhat = (p >= thr).astype(int); scores[name] = p
        r = {"variant": name, "auc": roc_auc_score(test["y"], p),
             "balanced_accuracy": balanced_accuracy_score(test["y"], yhat), "selection_rate": yhat.mean()}
        r.update(gaps(test, yhat)); r.update(gap_ci(test, yhat, seed=seed))
        pred = lambda d: (clf.predict_proba(dense(pre.transform(d[X])))[:, 1] >= thr).astype(int)
        for s in (0.05, 0.10):
            r[f"flip_sd_{s:.2f}"] = np.mean([np.mean(pred(sd_noise(test, num, sd, s, rng)) != yhat) for _ in range(N_DRAWS)])
        r["flip_realistic_5pct"] = np.mean([np.mean(pred(realistic(test, rng)) != yhat) for _ in range(N_DRAWS)])
        den = test[yhat == 0].copy(); den["income"] = den["income"] * 1.10
        r["gaming_income_plus10_dti_fixed"] = float(np.mean(pred(den) == 1))
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    out = pd.DataFrame(rows)
    # DeLong of every variant vs the baseline
    for i, name in enumerate(out.variant):
        if name == "baseline":
            continue
        aucs, cov = delong(test["y"].values, np.vstack([scores["baseline"], scores[name]]))
        z = (aucs[1] - aucs[0]) / np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
        out.loc[i, "delong_p_vs_baseline"] = 2 * (1 - stats.norm.cdf(abs(z)))
    out.to_csv(RESULTS / "hardening.csv", index=False, float_format="%.4f")
    print(out.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(len(out)); w = 0.27
    for k, (c, col, lab) in enumerate([("flip_sd_0.05", BLUE, "SD noise 0.05"), ("flip_realistic_5pct", ORANGE, "±5% income / appraisal"),
                                        ("gaming_income_plus10_dti_fixed", "#2e9e5b", "denials flipped by income +10%")]):
        axes[0].bar(x + (k - 1) * w, out[c] * 100, w, color=col, label=lab)
    axes[0].set_xticks(x); axes[0].set_xticklabels([v.split(" ")[0] for v in out.variant])
    axes[0].set_ylabel("%"); axes[0].set_title("Decision flips / gaming (lower = more robust)", fontsize=10); axes[0].legend(frameon=False, fontsize=7.5)
    axes[1].bar(x, out.auc, color=BLUE); axes[1].set_ylim(0.75, 0.82)
    axes[1].set_xticks(x); axes[1].set_xticklabels([v.split(" ")[0] for v in out.variant]); axes[1].set_ylabel("test AUC")
    axes[1].set_title("Accuracy cost of hardening", fontsize=10)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "hardening.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
