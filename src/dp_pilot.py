"""Pilot of the privacy ablation: DP-SGD logistic regression at a grid of ε values.

Per-example gradient clipping (L2 norm C) + Gaussian noise (σ·C) on Poisson-sampled mini-batches,
exactly the DP-SGD recipe from the lecture / Abadi et al. The noise multiplier σ for each target ε
is found with Google's dp-accounting RDP accountant at δ = 1/n_train. The same optimiser with the
noise and clipping switched off is run as the non-private reference, and the scikit-learn baseline
AUC from baselines.py is printed alongside for comparison.

Outputs (results/):
  dp_pilot.csv        ε, σ, steps, test AUC, accuracy at the frozen base-rate threshold, MI-attack AUC
The clipping norm and learning rate are chosen on the validation split without noise (the tuning
itself is not privatised — a known simplification, noted in the report).
Run:  python src/dp_pilot.py                      (grid from config.yaml: dp.epsilons)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant
from sklearn.metrics import roc_auc_score

from baselines import feature_lists, make_models, split

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RESULTS = ROOT / "results"
DP = CFG.get("dp", {"epsilons": [0.5, 1, 2, 4, 8], "epochs": 20, "batch_size": 1024, "clip_grid": [1.0, 3.0, 5.0, 10.0], "lr_grid": [0.5, 1.0, 2.0]})


def epsilon_for(sigma, q, steps, delta):
    acc = rdp_privacy_accountant.RdpAccountant()
    acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), steps)
    return acc.get_epsilon(delta)


def sigma_for(target_eps, q, steps, delta, lo=0.3, hi=50.0):
    """Bisection on the noise multiplier so that the accountant returns the target ε."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon_for(mid, q, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def dp_sgd(X, y, rng, sigma, clip, batch, epochs, lr):
    """Returns weights (bias folded in). sigma=0 and clip=None give plain mini-batch SGD."""
    n, d = X.shape
    w = np.zeros(d)
    q = batch / n
    steps = int(epochs * n / batch)
    for _ in range(steps):
        idx = np.flatnonzero(rng.random(n) < q)          # Poisson sampling
        if len(idx) == 0:
            continue
        p = sigmoid(X[idx] @ w)
        g = (p - y[idx])[:, None] * X[idx]                # per-example gradients (n_b × d)
        if clip is not None:
            norms = np.linalg.norm(g, axis=1, keepdims=True)
            g = g / np.maximum(1.0, norms / clip)         # clip each example to L2 ≤ C
        g_sum = g.sum(axis=0)
        if sigma > 0:
            g_sum = g_sum + rng.normal(0.0, sigma * clip, size=d)
        w -= lr * g_sum / batch                           # expected batch size normalisation
    return w, steps


def mi_attack_auc(p_train, y_train, p_test, y_test, rng):
    """Confidence-threshold membership inference: higher confidence in the true label ⇒ 'member'."""
    conf_in = np.where(y_train == 1, p_train, 1 - p_train)
    conf_out = np.where(y_test == 1, p_test, 1 - p_test)
    m = min(len(conf_in), len(conf_out))
    conf_in = rng.choice(conf_in, m, replace=False); conf_out = rng.choice(conf_out, m, replace=False)
    return roc_auc_score(np.r_[np.ones(m), np.zeros(m)], np.r_[conf_in, conf_out])


def main():
    seed = CFG["seed"]; rng = np.random.default_rng(seed)
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
    num, cat = feature_lists(); X_cols = num + cat
    train, val, test = split(df, seed)
    pre = make_models(num, cat, seed)["logistic_regression"].named_steps["pre"]
    Xtr = pre.fit_transform(train[X_cols]); Xte = pre.transform(test[X_cols]); Xva = pre.transform(val[X_cols])
    Xtr, Xte, Xva = (np.asarray(m.todense()) if hasattr(m, "todense") else np.asarray(m) for m in (Xtr, Xte, Xva))
    ones = lambda M: np.hstack([M, np.ones((M.shape[0], 1))])
    Xtr, Xte, Xva = ones(Xtr), ones(Xte), ones(Xva)
    ytr, yte, yva = train["y"].values.astype(float), test["y"].values.astype(float), val["y"].values.astype(float)
    n, d = Xtr.shape
    delta = 1.0 / n
    q = DP["batch_size"] / n
    steps = int(DP["epochs"] * n / DP["batch_size"])
    base_auc = pd.read_csv(RESULTS / "baseline_overall.csv").set_index("model").loc["logistic_regression", "auc"]
    rows = []
    t0 = time.time()
    # tune the clipping norm and learning rate on the VALIDATION split (no noise), never on the test split
    best = None
    for clip in DP["clip_grid"]:
        for lr in DP["lr_grid"]:
            w, _ = dp_sgd(Xtr, ytr, np.random.default_rng(seed), sigma=0.0, clip=clip, batch=DP["batch_size"], epochs=DP["epochs"], lr=lr)
            auc_va = roc_auc_score(yva, sigmoid(Xva @ w))
            if best is None or auc_va > best[0]:
                best = (auc_va, clip, lr)
    _, clip_norm, lr = best
    DP["clip_norm"], DP["lr"] = clip_norm, lr
    print(f"tuned on validation: clip C = {clip_norm}, lr = {lr} (val AUC {best[0]:.4f})")
    # non-private reference with the same optimiser
    w, _ = dp_sgd(Xtr, ytr, np.random.default_rng(seed), sigma=0.0, clip=None, batch=DP["batch_size"], epochs=DP["epochs"], lr=DP["lr"])
    p_te, p_tr, p_va = sigmoid(Xte @ w), sigmoid(Xtr @ w), sigmoid(Xva @ w)
    thr = float(np.quantile(p_va, 1 - yva.mean()))
    rows.append({"epsilon": np.inf, "sigma": 0.0, "steps": steps, "auc_test": roc_auc_score(yte, p_te),
                 "accuracy_test": float(((p_te >= thr) == yte).mean()), "mi_attack_auc": mi_attack_auc(p_tr, ytr, p_te, yte, rng)})
    print(f"non-private SGD reference: AUC {rows[-1]['auc_test']:.4f}  (scikit-learn LR baseline {base_auc:.4f})")
    for eps in DP["epsilons"]:
        sigma = sigma_for(eps, q, steps, delta)
        w, _ = dp_sgd(Xtr, ytr, np.random.default_rng(seed), sigma=sigma, clip=DP["clip_norm"], batch=DP["batch_size"], epochs=DP["epochs"], lr=DP["lr"])
        p_te, p_tr, p_va = sigmoid(Xte @ w), sigmoid(Xtr @ w), sigmoid(Xva @ w)
        thr = float(np.quantile(p_va, 1 - yva.mean()))
        rows.append({"epsilon": eps, "sigma": sigma, "steps": steps, "auc_test": roc_auc_score(yte, p_te),
                     "accuracy_test": float(((p_te >= thr) == yte).mean()), "mi_attack_auc": mi_attack_auc(p_tr, ytr, p_te, yte, rng)})
        print(f"ε = {eps:>4}: σ = {sigma:6.3f}  AUC {rows[-1]['auc_test']:.4f}  acc {rows[-1]['accuracy_test']:.4f}  MI-AUC {rows[-1]['mi_attack_auc']:.4f}")
    out = pd.DataFrame(rows); out["delta"] = f"{delta:.2e}"; out["clip_norm"] = DP["clip_norm"]; out["batch_size"] = DP["batch_size"]; out["epochs"] = DP["epochs"]
    out.to_csv(RESULTS / "dp_pilot.csv", index=False, float_format="%.4f")
    print(f"done in {time.time()-t0:.0f}s -> results/dp_pilot.csv  (n_train={n:,}, d={d}, δ={delta:.2e}, q={q:.4f}, steps={steps})")


if __name__ == "__main__":
    main()
