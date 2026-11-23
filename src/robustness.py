"""Robustness of the two baselines: input-noise perturbation, out-of-distribution scoring (2024 -> 2025)
and bounded adversarial / application-gaming probes (Track B: adversarial, perturbation, OOD, drift;
the drift statistics themselves are in drift_check.py).

1. noise       Gaussian noise on the standardised numeric inputs, sigma in {0.05, 0.1, 0.25, 0.5}
               standard-deviation units (3 draws), and random re-draws of 1 / 5 / 10 % of the
               categorical entries from the training marginals. Reported: decision flip rate against
               the clean decision at the frozen threshold, AUC, Black-White selection-rate gap.
2. ood         adversarial validation (Rabanser et al. 2019): a domain classifier trained to tell the
               2024 test rows from a same-size sample of the 2025 cohort. Its cross-validated AUC is
               the shift score (0.5 = indistinguishable). The full 2025 cohort is then split into
               deciles of the domain score and the frozen models' AUC per decile shows whether the
               most out-of-distribution applications are scored worse.
3. pgd         projected gradient descent on the logistic model in standardised numeric space under an
               L_inf budget eps in {0.1, 0.25, 0.5, 1.0} (10 steps), applied to each model's own denied
               test applications: white-box for the logistic model, a transfer attack for LightGBM.
               A random-sign perturbation with the same budget is the baseline. Categorical inputs
               are left untouched.
4. gaming      a realistic manipulation probe: over-state income by +5 / +10 / +20 / +50 % (once with the
               DTI bin held fixed, once with the bin moved by its midpoint) or cut the loan amount by
               5 / 10 / 20 % (CLTV moved accordingly) and count the model-denied applications that flip.

Outputs (results/): robustness_noise.csv, robustness_noise_by_input.csv, robustness_ood.csv, robustness_ood_deciles.csv,
  robustness_ood_inputs.csv, robustness_pgd.csv, robustness_gaming.csv, robustness_noise.png,
  run_log_robustness.json
Run:  python src/robustness.py
"""
import json
import time
import warnings

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ablations import dense, make_clf
from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from prepare import DTI_ORDER

warnings.filterwarnings("ignore")
SIGMAS = [0.05, 0.1, 0.25, 0.5]
CAT_FRACS = [0.01, 0.05, 0.10]
EPS = [0.1, 0.25, 0.5, 1.0]
INCOME_UP = [0.05, 0.10, 0.20, 0.50]
LOAN_DOWN = [0.05, 0.10, 0.20]
DTI_MID = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "36%-49%": 42.5, "50%-60%": 55.0, ">60%": 65.0}


def rebin_dti(series, factor):
    """Move a DTI bin when income changes by `factor` (DTI scales by 1/factor); Missing stays Missing."""
    def f(v):
        if v not in DTI_MID:
            return v
        x = DTI_MID[v] / factor
        return "<20%" if x < 20 else "20%-<30%" if x < 30 else "30%-<36%" if x < 36 else "36%-49%" if x < 50 else "50%-60%" if x <= 60 else ">60%"
    return pd.Categorical(series.astype(str).map(f), categories=DTI_ORDER)


def black_white_gap(yhat, race):
    r = np.asarray(race)
    return float(yhat[r == "Black or African American"].mean() - yhat[r == "White"].mean())


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed); t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    later = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['drift']}.parquet")
    num, cat = feature_lists(); X_cols = num + cat
    train, val, test = split(df, seed)
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr = dense(pre.fit_transform(train[X_cols])); Xva = dense(pre.transform(val[X_cols])); Xte = dense(pre.transform(test[X_cols]))
    ytr, yva, yte = train["y"].values, val["y"].values, test["y"].values
    race_te = test["derived_race"].astype(str).values
    n_num = len(num)
    models, thr, clean = {}, {}, {}
    for m in ["logistic_regression", "lightgbm"]:
        models[m] = make_clf(m, seed).fit(Xtr, ytr)
        thr[m] = choose_threshold(models[m].predict_proba(Xva)[:, 1], yva)
        p = models[m].predict_proba(Xte)[:, 1]; clean[m] = (p, (p >= thr[m]).astype(int))
    log = {"seed": seed, "thresholds": thr, "timings": {}}

    # ---- 1. noise ---------------------------------------------------------------------------------
    t1 = time.time(); rows = []
    for m in models:
        p0, y0 = clean[m]
        rows.append({"model": m, "perturbation": "none", "level": 0.0, "flip_rate": 0.0, "auc": roc_auc_score(yte, p0),
                     "selection_rate": y0.mean(), "black_white_gap": black_white_gap(y0, race_te)})
        for s in SIGMAS:
            flips, aucs, sels, gaps = [], [], [], []
            for d in range(3):
                Xn = Xte.copy(); Xn[:, :n_num] += rng.normal(0, s, size=(len(Xn), n_num))
                p = models[m].predict_proba(Xn)[:, 1]; yh = (p >= thr[m]).astype(int)
                flips.append((yh != y0).mean()); aucs.append(roc_auc_score(yte, p)); sels.append(yh.mean()); gaps.append(black_white_gap(yh, race_te))
            rows.append({"model": m, "perturbation": "gaussian_numeric_sd_units", "level": s, "flip_rate": np.mean(flips), "auc": np.mean(aucs),
                         "selection_rate": np.mean(sels), "black_white_gap": np.mean(gaps)})
        for f in CAT_FRACS:
            flips, aucs, sels, gaps = [], [], [], []
            for d in range(3):
                Xd = test[X_cols].copy()
                for c in cat:
                    mask = rng.random(len(Xd)) < f
                    Xd.loc[Xd.index[mask], c] = rng.choice(train[c].astype(str).values, mask.sum())
                p = models[m].predict_proba(dense(pre.transform(Xd)))[:, 1]; yh = (p >= thr[m]).astype(int)
                flips.append((yh != y0).mean()); aucs.append(roc_auc_score(yte, p)); sels.append(yh.mean()); gaps.append(black_white_gap(yh, race_te))
            rows.append({"model": m, "perturbation": "categorical_redraw_fraction", "level": f, "flip_rate": np.mean(flips), "auc": np.mean(aucs),
                         "selection_rate": np.mean(sels), "black_white_gap": np.mean(gaps)})
    noise = pd.DataFrame(rows); noise.to_csv(RESULTS / "robustness_noise.csv", index=False, float_format="%.4f")
    # which numeric input is the fragile one: 0.1 SD of noise on ONE input at a time (3 draws)
    rows = []
    for m in models:
        y0 = clean[m][1]
        for j, c in enumerate(num):
            flips = []
            for d in range(3):
                Xn = Xte.copy(); Xn[:, j] += rng.normal(0, 0.1, size=len(Xn))
                flips.append(((models[m].predict_proba(Xn)[:, 1] >= thr[m]).astype(int) != y0).mean())
            rows.append({"model": m, "input": c, "noise_sd_units": 0.1, "flip_rate": float(np.mean(flips)),
                         "share_of_rows_at_modal_value": float((test[c] == test[c].mode().iloc[0]).mean())})
    pd.DataFrame(rows).to_csv(RESULTS / "robustness_noise_by_input.csv", index=False, float_format="%.4f")
    log["timings"]["noise"] = round(time.time() - t1, 1)

    # ---- 2. OOD: adversarial validation -------------------------------------------------------------
    t1 = time.time()
    samp = later.sample(n=len(test), random_state=seed)
    Xdom = np.vstack([Xte, dense(pre.transform(samp[X_cols]))]); ydom = np.r_[np.zeros(len(test)), np.ones(len(samp))]
    dom = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)
    p_dom = cross_val_predict(dom, Xdom, ydom, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    dom_auc = roc_auc_score(ydom, p_dom)
    dom.fit(Xdom, ydom)
    names = list(pre.get_feature_names_out())
    imp = pd.DataFrame({"feature": names, "gain": dom.booster_.feature_importance("gain")})
    imp["input"] = [n[5:] if n.startswith("num__") else next((c for c in sorted(cat, key=len, reverse=True) if n[5:].startswith(c)), n) for n in names]
    imp = imp.groupby("input")["gain"].sum().sort_values(ascending=False); imp = (imp / imp.sum()).reset_index(name="gain_share")
    imp.to_csv(RESULTS / "robustness_ood_inputs.csv", index=False, float_format="%.4f")
    X25 = dense(pre.transform(later[X_cols])); y25 = later["y"].values
    score25 = dom.predict_proba(X25)[:, 1]
    dec = pd.qcut(score25, 10, labels=False, duplicates="drop")
    drows = []
    for d in sorted(set(dec)):
        msk = dec == d
        r = {"domain_score_decile": int(d) + 1, "n": int(msk.sum()), "mean_domain_score": float(score25[msk].mean()), "approval_rate": float(y25[msk].mean())}
        for m in models:
            p = models[m].predict_proba(X25[msk])[:, 1]
            r[f"auc_{m}"] = roc_auc_score(y25[msk], p) if len(set(y25[msk])) > 1 else float("nan")
            r[f"selection_rate_{m}"] = float((p >= thr[m]).mean())
        drows.append(r)
    pd.DataFrame(drows).to_csv(RESULTS / "robustness_ood_deciles.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"domain_classifier": "LightGBM 200 trees, 5-fold CV", "n_2024_test": len(test), "n_2025_sample": len(samp),
                   "domain_auc": dom_auc, "top_separating_inputs": "; ".join(f"{a} ({b:.2f})" for a, b in imp.head(5).values)}]) \
        .to_csv(RESULTS / "robustness_ood.csv", index=False, float_format="%.4f")
    log["timings"]["ood"] = round(time.time() - t1, 1)

    # ---- 3. PGD on the logistic model, transferred to LightGBM --------------------------------------
    t1 = time.time(); rows = []
    w = models["logistic_regression"].coef_.ravel()
    for target in models:
        denied = np.flatnonzero(clean[target][1] == 0); X0 = Xte[denied]
        for eps in EPS:
            Xa = X0.copy(); step = eps / 4
            for _ in range(10):                                   # the logit's gradient w.r.t. the numeric inputs is w (linear model)
                Xa[:, :n_num] += step * np.sign(w[:n_num])
                Xa[:, :n_num] = np.clip(Xa[:, :n_num], X0[:, :n_num] - eps, X0[:, :n_num] + eps)
            Xr = X0.copy(); Xr[:, :n_num] += eps * rng.choice([-1.0, 1.0], size=(len(Xr), n_num))   # same budget, random corner
            adv = float((models[target].predict_proba(Xa)[:, 1] >= thr[target]).mean())
            rnd = float((models[target].predict_proba(Xr)[:, 1] >= thr[target]).mean())
            rows.append({"target_model": target, "attack": "PGD L-inf on numeric inputs, logistic-model gradient", "eps_sd_units": eps, "n_denied": len(denied),
                         "flipped_to_approve_share": adv, "random_sign_baseline_flipped_share": rnd,
                         "kind": "white-box" if target == "logistic_regression" else "transfer from logistic model"})
    pgd = pd.DataFrame(rows); pgd.to_csv(RESULTS / "robustness_pgd.csv", index=False, float_format="%.4f")
    log["timings"]["pgd"] = round(time.time() - t1, 1)

    # ---- 4. application gaming ---------------------------------------------------------------------
    t1 = time.time(); rows = []
    for m in models:
        denied = np.flatnonzero(clean[m][1] == 0); base = test[X_cols].iloc[denied]
        for u in INCOME_UP:
            for dti_handling in ["DTI bin held fixed (conservative)", "DTI bin moved with income (bin midpoints)"]:
                Xd = base.copy(); Xd["income"] = Xd["income"] * (1 + u)
                if dti_handling.startswith("DTI bin moved"):
                    Xd["debt_to_income_ratio"] = rebin_dti(Xd["debt_to_income_ratio"], 1 + u)
                p = models[m].predict_proba(dense(pre.transform(Xd)))[:, 1]
                rows.append({"model": m, "manipulation": "income over-stated", "dti_handling": dti_handling, "change": f"+{int(u*100)} %", "n_denied": len(denied), "flipped_to_approve_share": float((p >= thr[m]).mean())})
        for d in LOAN_DOWN:
            Xd = base.copy(); Xd["loan_amount"] = Xd["loan_amount"] * (1 - d); Xd["combined_loan_to_value_ratio"] = Xd["combined_loan_to_value_ratio"] * (1 - d)
            p = models[m].predict_proba(dense(pre.transform(Xd)))[:, 1]
            rows.append({"model": m, "manipulation": "loan amount reduced (CLTV moved with it)", "dti_handling": "n/a", "change": f"-{int(d*100)} %", "n_denied": len(denied), "flipped_to_approve_share": float((p >= thr[m]).mean())})
    gaming = pd.DataFrame(rows); gaming.to_csv(RESULTS / "robustness_gaming.csv", index=False, float_format="%.4f")
    log["timings"]["gaming"] = round(time.time() - t1, 1)

    # ---- figure ---------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for m, color in [("logistic_regression", BLUE), ("lightgbm", ORANGE)]:
        s = noise[(noise.model == m) & (noise.perturbation != "categorical_redraw_fraction")]
        axes[0].plot(s["level"], 100 * s["flip_rate"], "o-", color=color, label=m.replace("_", " "))
        axes[1].plot(s["level"], s["auc"], "o-", color=color, label=m.replace("_", " "))
    axes[0].set_xlabel("Gaussian noise on numeric inputs (SD units)"); axes[0].set_ylabel("Decisions flipped (%)")
    axes[1].set_xlabel("Gaussian noise on numeric inputs (SD units)"); axes[1].set_ylabel("Test AUC")
    for ax in axes:
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Input-noise robustness on the 2024 test split", fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / "robustness_noise.png", dpi=200); plt.close(fig)

    log["domain_auc"] = dom_auc; log["wall_seconds"] = round(time.time() - t0, 1); log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / "run_log_robustness.json").write_text(json.dumps(log, indent=2, default=str))
    print(noise.to_string(index=False)); print(f"\ndomain-classifier AUC 2024 test vs 2025: {dom_auc:.4f}")
    print(pgd.to_string(index=False)); print(gaming.to_string(index=False)); print(pd.read_csv(RESULTS / "robustness_noise_by_input.csv").to_string(index=False))
    print(f"done in {log['wall_seconds']}s")


if __name__ == "__main__":
    main()
