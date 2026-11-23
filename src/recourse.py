"""Counterfactual recourse: what would a denied applicant have to change to be approved, and is that
burden the same for every group? ("recourse gap", Ustun et al. 2019; Karimi et al. 2021)

Actionable changes only (race, sex, age, credit-score model, channel, etc. are never changed):
  * income      +0% ... +30% in 5% steps        (cost 1 per +10%)
  * loan amount -0% ... -30% in 5% steps        (cost 1 per -10%; CLTV is recomputed as the loan
                                                 shrinks against the same property value, i.e. a
                                                 bigger down payment)
  * DTI bin     0, 1 or 2 bins lower             (cost 1.5 per bin; paying down other debt)
For each applicant the model denies on the 2024 test split, every combination (7 x 7 x 3 = 147) is
scored and the cheapest one that flips the decision is kept. Reported per group: share with any
feasible recourse inside the box, median cost, and the Black-White and Hispanic-non-Hispanic gaps in
mean cost with bootstrap 95% CIs.

Outputs: results/recourse_by_group.csv, results/recourse_applicants.csv, results/recourse_cost.png

Run:  python src/recourse.py
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split

DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-49%", "50%-60%", ">60%"]
INC = [0, 5, 10, 15, 20, 25, 30]
LOAN = [0, 5, 10, 15, 20, 25, 30]
DTI = [0, 1, 2]
GROUPS = [("White", "derived_race", "White"), ("Black", "derived_race", "Black or African American"),
          ("Asian", "derived_race", "Asian"), ("Hispanic", "derived_ethnicity", "Hispanic or Latino"),
          ("Not Hispanic", "derived_ethnicity", "Not Hispanic or Latino"), ("Female", "derived_sex", "Female"),
          ("Male", "derived_sex", "Male")]


def lower_dti(s, k):
    out = []
    for v in s.astype(str):
        out.append(DTI_ORDER[max(DTI_ORDER.index(v) - k, 0)] if v in DTI_ORDER else v)
    return pd.Categorical(out, categories=s.cat.categories) if hasattr(s, "cat") else out


def boot_gap(a, b, rng, B=2000):
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(B)]
    return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed)
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat
    train, val, test = split(df, seed)
    rows, per = [], []
    for name, model in make_models(num, cat, seed).items():
        model.fit(train[X], train["y"])
        thr = choose_threshold(model.predict_proba(val[X])[:, 1], val["y"].values)
        den = test[model.predict_proba(test[X])[:, 1] < thr].copy()
        best = np.full(len(den), np.inf); best_combo = [None] * len(den)
        for inc, loan, dti in itertools.product(INC, LOAN, DTI):
            if inc == 0 and loan == 0 and dti == 0:
                continue
            cost = inc / 10 + loan / 10 + 1.5 * dti
            e = den.copy()
            e["income"] = e["income"] * (1 + inc / 100)
            e["loan_amount"] = e["loan_amount"] * (1 - loan / 100)
            e["combined_loan_to_value_ratio"] = e["combined_loan_to_value_ratio"] * (1 - loan / 100)
            if dti:
                e["debt_to_income_ratio"] = lower_dti(e["debt_to_income_ratio"], dti)
            ok = model.predict_proba(e[X])[:, 1] >= thr
            better = ok & (cost < best)
            best[better] = cost
            for i in np.flatnonzero(better):
                best_combo[i] = f"income +{inc}%, loan -{loan}%, DTI -{dti} bin"
        den["recourse_cost"] = best; den["recourse_action"] = best_combo; den["model"] = name
        per.append(den[["model", "derived_race", "derived_ethnicity", "derived_sex", "y", "recourse_cost", "recourse_action"]])
        feas = np.isfinite(best)
        for lab, attr, v in GROUPS:
            m = (den[attr] == v).values
            c = best[m & feas]
            rows.append({"model": name, "group": lab, "n_denied": int(m.sum()), "feasible_share": feas[m].mean() if m.any() else np.nan,
                         "mean_cost_if_feasible": c.mean() if len(c) else np.nan,
                         "median_cost_if_feasible": np.median(c) if len(c) else np.nan})
        for lab, (ga, gb) in {"Black - White": ("Black", "White"), "Hispanic - Not Hispanic": ("Hispanic", "Not Hispanic")}.items():
            a = [g for g in GROUPS if g[0] == ga][0]; b = [g for g in GROUPS if g[0] == gb][0]
            ma = (den[a[1]] == a[2]).values & feas; mb = (den[b[1]] == b[2]).values & feas
            g, lo, hi = boot_gap(best[ma], best[mb], rng)
            fa = (den[a[1]] == a[2]).values; fb = (den[b[1]] == b[2]).values
            fg = feas[fa].mean() - feas[fb].mean()
            rows.append({"model": name, "group": f"GAP {lab}", "n_denied": int(fa.sum() + fb.sum()),
                         "feasible_share": fg, "mean_cost_if_feasible": g, "cost_gap_ci_lo": lo, "cost_gap_ci_hi": hi})
        print(f"{name}: {feas.mean():.3f} of {len(den)} denials have recourse inside the box")
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "recourse_by_group.csv", index=False, float_format="%.4f")
    pd.concat(per).to_csv(RESULTS / "recourse_applicants.csv", index=False, float_format="%.3f")
    print(out.round(3).to_string(index=False))

    d = out[(out.model == "lightgbm") & ~out.group.str.startswith("GAP")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].bar(d.group, d.feasible_share * 100, color=BLUE); axes[0].set_ylabel("denials with feasible recourse (%)")
    axes[1].bar(d.group, d.mean_cost_if_feasible, color=ORANGE); axes[1].set_ylabel("mean effort cost (units)")
    for ax, t in zip(axes, ["Recourse feasibility by group (LightGBM)", "Recourse cost by group (LightGBM)"]):
        ax.set_title(t, fontsize=10); ax.tick_params(axis="x", labelsize=7.5, rotation=20)
        ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "recourse_cost.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
