"""Privacy evaluation of the pipeline (Track B: privacy via epsilon + a leakage probe).

1. mi_probes    membership inference against BOTH fitted baselines, members = training rows,
                non-members = test rows (equal-size samples):
                  confidence  - confidence in the true label (Yeom et al. 2018; the per-example loss
                                threshold is the same ranking, so it is not reported twice)
                  shadow      - Shokri et al. 2017: a shadow model trained on half of the training rows,
                                an attack model (LightGBM) trained on the shadow's in/out confidence
                                features, then applied to the target model
                Attack AUC with a bootstrap 95 % CI, best advantage max(TPR - FPR), and the train-test
                generalisation gap that drives leakage.
2. dp_grid      DP-SGD logistic regression (dp_pilot.py recipe, RDP accountant, delta = 1/n_train) at
                the epsilon grid in config.yaml over 3 seeds: mean +- sd of test AUC, accuracy at the
                base-rate threshold and MI-attack AUC, plus the Black-White and Hispanic-non-Hispanic
                selection-rate gaps at every epsilon (does privacy noise move the disparity?).
3. uniqueness   re-identification check on the public LAR itself (Week-4 k-anonymity lens): share of
                2024 cohort records that are unique, and share in classes of size < 5, for growing
                quasi-identifier sets.

Outputs (results/): privacy_mi.csv, privacy_dp_grid.csv, privacy_dp_fairness.csv, privacy_uniqueness.csv,
  privacy_dp.png, run_log_privacy.json
Run:  python src/privacy.py        (expects results/dp_pilot.csv for the tuned clip norm / learning rate;
                                    runs the validation tuning itself when the file is missing)
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
from sklearn.metrics import roc_auc_score, roc_curve

from ablations import dense, make_clf
from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from dp_pilot import DP, dp_sgd, sigma_for, sigmoid

warnings.filterwarnings("ignore")
SEEDS = [42, 43, 44]
QI_SETS = [["county_code", "loan_amount", "income"],
           ["county_code", "loan_amount", "income", "applicant_age", "derived_sex"],
           ["county_code", "loan_amount", "income", "applicant_age", "derived_sex", "lei"],
           ["county_code", "loan_amount", "income", "applicant_age", "derived_sex", "lei", "property_value"]]


def advantage(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def boot_auc(y, s, rng, B=200):
    idx = np.arange(len(y)); vals = []
    for _ in range(B):
        i = rng.choice(idx, len(idx), replace=True); vals.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def conf_in_true_label(p, y):
    return np.where(y == 1, p, 1 - p)


def gaps_at_threshold(p, thr, test):
    yhat = (p >= thr).astype(int); r = test["derived_race"].astype(str).values; e = test["derived_ethnicity"].astype(str).values
    return {"black_white_gap": float(yhat[r == "Black or African American"].mean() - yhat[r == "White"].mean()),
            "hispanic_gap": float(yhat[e == "Hispanic or Latino"].mean() - yhat[e == "Not Hispanic or Latino"].mean()),
            "selection_rate": float(yhat.mean())}


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed); t0 = time.time()
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X_cols = num + cat
    train, val, test = split(df, seed)
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr = dense(pre.fit_transform(train[X_cols])); Xva = dense(pre.transform(val[X_cols])); Xte = dense(pre.transform(test[X_cols]))
    ytr, yva, yte = train["y"].values.astype(int), val["y"].values.astype(int), test["y"].values.astype(int)
    log = {"seed": seed, "timings": {}}

    # ---- 1. membership-inference probes against the two baselines --------------------------------------
    t1 = time.time(); rows = []
    m_idx = rng.choice(len(Xtr), len(Xte), replace=False)                      # equal-size member sample
    half = rng.permutation(len(Xtr)); sh_in, sh_out = half[: len(half) // 2], half[len(half) // 2:]
    for m in ["logistic_regression", "lightgbm"]:
        clf = make_clf(m, seed).fit(Xtr, ytr)
        p_tr, p_te = clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]
        gen_gap_auc = roc_auc_score(ytr, p_tr) - roc_auc_score(yte, p_te)
        thr = choose_threshold(clf.predict_proba(Xva)[:, 1], yva)
        gen_gap_acc = ((p_tr >= thr) == ytr).mean() - ((p_te >= thr) == yte).mean()
        y_mi = np.r_[np.ones(len(Xte)), np.zeros(len(Xte))]
        # confidence attack
        s = np.r_[conf_in_true_label(p_tr[m_idx], ytr[m_idx]), conf_in_true_label(p_te, yte)]
        lo, hi = boot_auc(y_mi, s, rng)
        rows.append({"model": m, "attack": "confidence in true label", "attack_auc": roc_auc_score(y_mi, s), "ci_lo": lo, "ci_hi": hi,
                     "best_advantage_tpr_minus_fpr": advantage(y_mi, s), "n_members": len(Xte), "n_non_members": len(Xte),
                     "train_auc_minus_test_auc": gen_gap_auc, "train_acc_minus_test_acc": gen_gap_acc})
        # shadow-model attack
        shadow = make_clf(m, seed + 7).fit(Xtr[sh_in], ytr[sh_in])
        def feats(model, X, y):
            p = model.predict_proba(X)[:, 1]; c = conf_in_true_label(p, y)
            return np.c_[p, c, -np.log(np.clip(c, 1e-9, 1)), y]
        n_sh = min(len(sh_in), len(sh_out))
        A = np.vstack([feats(shadow, Xtr[sh_in[:n_sh]], ytr[sh_in[:n_sh]]), feats(shadow, Xtr[sh_out[:n_sh]], ytr[sh_out[:n_sh]])])
        a_y = np.r_[np.ones(n_sh), np.zeros(n_sh)]
        attack = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, verbose=-1).fit(A, a_y)
        s = attack.predict_proba(np.vstack([feats(clf, Xtr[m_idx], ytr[m_idx]), feats(clf, Xte, yte)]))[:, 1]
        lo, hi = boot_auc(y_mi, s, rng)
        rows.append({"model": m, "attack": "shadow model + attack classifier", "attack_auc": roc_auc_score(y_mi, s), "ci_lo": lo, "ci_hi": hi,
                     "best_advantage_tpr_minus_fpr": advantage(y_mi, s), "n_members": len(Xte), "n_non_members": len(Xte),
                     "train_auc_minus_test_auc": gen_gap_auc, "train_acc_minus_test_acc": gen_gap_acc})
        print(f"{m}: MI confidence AUC {rows[-2]['attack_auc']:.4f}, shadow AUC {rows[-1]['attack_auc']:.4f}, gen gap AUC {gen_gap_auc:+.4f}")
    mi = pd.DataFrame(rows); mi.to_csv(RESULTS / "privacy_mi.csv", index=False, float_format="%.4f")
    log["timings"]["mi"] = round(time.time() - t1, 1)

    # ---- 2. DP-SGD grid over seeds, with fairness gaps ----------------------------------------------------
    t1 = time.time()
    ones = lambda M: np.hstack([M, np.ones((M.shape[0], 1))])
    Xtr1, Xva1, Xte1 = ones(Xtr), ones(Xva), ones(Xte)
    ytr_f, yva_f, yte_f = ytr.astype(float), yva.astype(float), yte.astype(float)
    n = len(Xtr1); delta = 1.0 / n; q = DP["batch_size"] / n; steps = int(DP["epochs"] * n / DP["batch_size"])
    # the pilot stores the clip norm it tuned; the learning rate is re-tuned on the validation split (noise-free, cheap)
    pilot = RESULTS / "dp_pilot.csv"
    clip_grid = [float(pd.read_csv(pilot)["clip_norm"].iloc[0])] if pilot.exists() else DP["clip_grid"]
    best = None
    for c_ in clip_grid:
        for l_ in DP["lr_grid"]:
            w, _ = dp_sgd(Xtr1, ytr_f, np.random.default_rng(seed), sigma=0.0, clip=c_, batch=DP["batch_size"], epochs=DP["epochs"], lr=l_)
            a = roc_auc_score(yva_f, sigmoid(Xva1 @ w))
            if best is None or a > best[0]:
                best = (a, c_, l_)
    _, clip, lr = best
    rows, frows = [], []
    for eps in [np.inf] + list(DP["epsilons"]):
        sigma = 0.0 if eps == np.inf else sigma_for(eps, q, steps, delta)
        per_seed = []
        for s_ in SEEDS:
            w, _ = dp_sgd(Xtr1, ytr_f, np.random.default_rng(s_), sigma=sigma, clip=(None if eps == np.inf else clip), batch=DP["batch_size"], epochs=DP["epochs"], lr=lr)
            p_te, p_va, p_tr = sigmoid(Xte1 @ w), sigmoid(Xva1 @ w), sigmoid(Xtr1 @ w)
            thr = float(np.quantile(p_va, 1 - yva_f.mean()))
            y_mi = np.r_[np.ones(len(Xte)), np.zeros(len(Xte))]
            s = np.r_[conf_in_true_label(p_tr[m_idx], ytr[m_idx]), conf_in_true_label(p_te, yte)]
            g = gaps_at_threshold(p_te, thr, test)
            per_seed.append({"seed": s_, "auc": roc_auc_score(yte_f, p_te), "accuracy": float(((p_te >= thr) == yte_f).mean()),
                             "mi_attack_auc": roc_auc_score(y_mi, s), **g})
        ps = pd.DataFrame(per_seed)
        rows.append({"epsilon": eps, "sigma": sigma, "delta": delta, "steps": steps, "clip_norm": clip, "lr": lr, "n_seeds": len(SEEDS),
                     "auc_mean": ps["auc"].mean(), "auc_sd": ps["auc"].std(ddof=1), "accuracy_mean": ps["accuracy"].mean(), "accuracy_sd": ps["accuracy"].std(ddof=1),
                     "mi_attack_auc_mean": ps["mi_attack_auc"].mean(), "mi_attack_auc_sd": ps["mi_attack_auc"].std(ddof=1)})
        frows.append({"epsilon": eps, "black_white_gap_mean": ps["black_white_gap"].mean(), "black_white_gap_sd": ps["black_white_gap"].std(ddof=1),
                      "hispanic_gap_mean": ps["hispanic_gap"].mean(), "hispanic_gap_sd": ps["hispanic_gap"].std(ddof=1), "selection_rate_mean": ps["selection_rate"].mean()})
        print(f"eps {eps:>4}: AUC {rows[-1]['auc_mean']:.4f} +- {rows[-1]['auc_sd']:.4f}  acc {rows[-1]['accuracy_mean']:.4f}  MI {rows[-1]['mi_attack_auc_mean']:.4f}  "
              f"Black-White gap {frows[-1]['black_white_gap_mean']:+.4f} +- {frows[-1]['black_white_gap_sd']:.4f}  Hispanic gap {frows[-1]['hispanic_gap_mean']:+.4f}")
    grid = pd.DataFrame(rows); grid.to_csv(RESULTS / "privacy_dp_grid.csv", index=False, float_format="%.6g")
    fair = pd.DataFrame(frows); fair.to_csv(RESULTS / "privacy_dp_fairness.csv", index=False, float_format="%.4f")
    log["timings"]["dp_grid"] = round(time.time() - t1, 1); log["dp"] = {"clip_norm": clip, "lr": lr, "delta": delta, "steps": steps, "q": q}

    # ---- 3. uniqueness of the public records --------------------------------------------------------------
    rows = []
    for qi in QI_SETS:
        sizes = df.groupby(qi, dropna=False, observed=True).size()
        rec = df.merge(sizes.rename("k").reset_index(), on=qi, how="left")["k"]
        rows.append({"quasi_identifiers": " + ".join(qi), "n_records": len(df), "share_unique": float((rec == 1).mean()),
                     "share_in_class_smaller_than_5": float((rec < 5).mean()), "median_class_size": float(rec.median())})
    uniq = pd.DataFrame(rows); uniq.to_csv(RESULTS / "privacy_uniqueness.csv", index=False, float_format="%.4f")

    # ---- figure -----------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    fin = grid[grid.epsilon != np.inf]; ref = grid[grid.epsilon == np.inf].iloc[0]
    axes[0].errorbar(fin["epsilon"], fin["auc_mean"], yerr=fin["auc_sd"], fmt="o-", color=BLUE, capsize=3, label="DP-SGD logistic (3 seeds)")
    axes[0].axhline(ref["auc_mean"], color="#777", ls="--", lw=1, label=f"non-private SGD ({ref['auc_mean']:.3f})")
    axes[0].set_xscale("log"); axes[0].set_xlabel("epsilon (delta = 1/n_train)"); axes[0].set_ylabel("Test AUC")
    ff = fair[fair.epsilon != np.inf]; fref = fair[fair.epsilon == np.inf].iloc[0]
    axes[1].errorbar(ff["epsilon"], 100 * ff["black_white_gap_mean"], yerr=100 * ff["black_white_gap_sd"], fmt="o-", color=ORANGE, capsize=3, label="Black - White selection-rate gap")
    axes[1].errorbar(ff["epsilon"], 100 * ff["hispanic_gap_mean"], yerr=100 * ff["hispanic_gap_sd"], fmt="s-", color=BLUE, capsize=3, label="Hispanic - non-Hispanic gap")
    axes[1].axhline(100 * fref["black_white_gap_mean"], color=ORANGE, ls="--", lw=1); axes[1].axhline(100 * fref["hispanic_gap_mean"], color=BLUE, ls="--", lw=1)
    axes[1].set_xscale("log"); axes[1].set_xlabel("epsilon (dashed = non-private)"); axes[1].set_ylabel("Gap (pp)")
    for ax in axes:
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Privacy-utility and privacy-fairness curves (DP-SGD logistic regression)", fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / "privacy_dp.png", dpi=200); plt.close(fig)

    log["wall_seconds"] = round(time.time() - t0, 1); log["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RESULTS / "run_log_privacy.json").write_text(json.dumps(log, indent=2, default=str))
    print(uniq.to_string(index=False)); print(f"done in {log['wall_seconds']}s")


if __name__ == "__main__":
    main()
