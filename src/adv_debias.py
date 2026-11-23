"""In-processing fairness: adversarial debiasing (Zhang, Lemoine & Mitchell, AIES 2018), written out
in NumPy for the logistic model, the one in-processing method the Track B deck names.

A logistic predictor is trained together with an adversary that tries to recover the protected
attribute z (1 = Black or Hispanic applicant) from the predictor's logit s. For the equalized-odds
version the adversary also sees the true label: its inputs are [s, s*y, s*(1-y)]. Each predictor step
uses Zhang et al.'s update
      g = grad L_P  -  proj_{grad L_A} grad L_P  -  alpha * grad L_A
so the predictor gets better at its task while removing what helps the adversary. alpha = 0 is the
plain logistic model trained with the same optimiser (the fair comparison point). The protected
attribute is used ONLY during training, never at decision time (unlike post-processing).

Reported per alpha: AUC, balanced accuracy, selection rate, key gaps, and the adversary's AUC on the
test split (0.5 = the scores carry no information about z beyond y).

Outputs: results/adv_debias.csv, results/adv_debias.png

Run:  python src/adv_debias.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from baselines import BLUE, CFG, ORANGE, RESULTS, ROOT, choose_threshold, feature_lists, make_models, split
from rigor import gap_ci, gaps

ALPHAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
EPOCHS, BATCH, LR_P, LR_A = 8, 512, 0.05, 0.05


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def dense(M):
    return np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)


def protected(d):
    return ((d["derived_race"] == "Black or African American") | (d["derived_ethnicity"] == "Hispanic or Latino")).astype(float).values


def train_adv(X, y, z, alpha, seed):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d); u = np.zeros(3); c = 0.0
    for _ in range(EPOCHS):
        for idx in np.array_split(rng.permutation(n), n // BATCH):
            xb, yb, zb = X[idx], y[idx], z[idx]
            s = xb @ w
            gP = xb.T @ (sig(s) - yb) / len(idx)
            feat = np.c_[s, s * yb, s * (1 - yb)]
            pa = sig(feat @ u + c)
            ds = u[0] + u[1] * yb + u[2] * (1 - yb)            # d(adversary logit)/ds
            gA = xb.T @ ((pa - zb) * ds) / len(idx)
            if alpha > 0:
                nA = gA / (np.linalg.norm(gA) + 1e-12)
                g = gP - (gP @ nA) * nA - alpha * gA
            else:
                g = gP
            w -= LR_P * g
            # adversary step (minimises its own log-loss)
            u -= LR_A * feat.T @ (pa - zb) / len(idx)
            c -= LR_A * np.mean(pa - zb)
    return w


def adversary_auc(s_tr, y_tr, z_tr, s_te, y_te, z_te):
    f = lambda s, y: np.c_[s, s * y, s * (1 - y)]
    a = LogisticRegression(max_iter=1000).fit(f(s_tr, y_tr), z_tr)
    return roc_auc_score(z_te, a.predict_proba(f(s_te, y_te))[:, 1])


def main():
    seed = CFG["seed"]
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X = num + cat
    train, val, test = split(df, seed)
    pre = clone(make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]).fit(train[X])
    add1 = lambda d: np.c_[np.ones(len(d)), dense(pre.transform(d[X]))]
    Xtr, Xva, Xte = add1(train), add1(val), add1(test)
    ytr, yva, yte = train["y"].values.astype(float), val["y"].values, test["y"].values
    ztr, zte = protected(train), protected(test)
    rows = []
    for a in ALPHAS:
        w = train_adv(Xtr, ytr, ztr, a, seed)
        pv, pt = sig(Xva @ w), sig(Xte @ w)
        thr = choose_threshold(pv, yva); yhat = (pt >= thr).astype(int)
        r = {"alpha": a, "auc": roc_auc_score(yte, pt), "balanced_accuracy": balanced_accuracy_score(yte, yhat),
             "selection_rate": yhat.mean(),
             "adversary_auc_test": adversary_auc(Xtr @ w, ytr, ztr, Xte @ w, yte, zte)}
        r.update(gaps(test, yhat)); r.update(gap_ci(test, yhat, seed=seed)); rows.append(r)
        print({k: round(float(v), 4) for k, v in r.items()})
    out = pd.DataFrame(rows); out.to_csv(RESULTS / "adv_debias.csv", index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(out.alpha, out.Black_sel_gap * 100, marker="o", color=BLUE, label="Black − White selection gap")
    axes[0].plot(out.alpha, out.Hispanic_sel_gap * 100, marker="s", color=ORANGE, label="Hispanic − not selection gap")
    axes[0].axhline(0, color="#8a8a8a", lw=1); axes[0].set_xlabel("adversary weight α"); axes[0].set_ylabel("gap (pp)")
    axes[0].set_title("Adversarial debiasing: gaps", fontsize=10); axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(out.alpha, out.auc, marker="o", color=BLUE, label="model AUC")
    axes[1].plot(out.alpha, out.adversary_auc_test, marker="s", color=ORANGE, label="adversary AUC (recover z)")
    axes[1].set_xlabel("adversary weight α"); axes[1].set_title("Utility vs information about z", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RESULTS / "adv_debias.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
