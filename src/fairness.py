"""Fairness metric helpers (same definitions as the CMPE 253 Algorithmic Fairness lecture / HW2).

fairness_metrics(tp, fp, tn, fn) -> selection_rate, tpr, fpr, precision
group_report(y, yhat, g, reference)   -> per-group metrics, gaps vs. the reference group,
                                          two-proportion z-tests and bootstrap 95 % CIs

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["selection_rate", "tpr", "fpr", "precision"]


def fairness_metrics(tp, fp, tn, fn):
    """Per-group rates from the four confusion-matrix counts (NaN when a denominator is 0)."""
    n = tp + fp + tn + fn
    return {
        "selection_rate": (tp + fp) / n if n else float("nan"),
        "tpr": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


def counts(y, yhat):
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, fp, tn, fn


def _denominators(tp, fp, tn, fn):
    """The sample size behind each metric (used by the z-test SE, as in HW2)."""
    return {"selection_rate": tp + fp + tn + fn, "tpr": tp + fn, "fpr": fp + tn, "precision": tp + fp}


def two_prop_z(p1, n1, p2, n2):
    """Two-proportion z-test for p2 - p1 (unpooled SE); returns z and two-sided p."""
    if min(n1, n2) == 0 or np.isnan(p1) or np.isnan(p2):
        return float("nan"), float("nan")
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return float("nan"), float("nan")
    z = (p2 - p1) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def group_report(y, yhat, g, reference, B=1000, seed=0, min_n=200):
    """Metrics per group of `g`, gaps (group - reference) with bootstrap CIs and z-tests."""
    y, yhat, g = np.asarray(y), np.asarray(yhat), np.asarray(g).astype(str)
    rng = np.random.default_rng(seed)
    groups = pd.Series(g).value_counts().index.tolist()
    if reference not in groups:
        reference = groups[0]  # fall back to the largest group
    ref_mask = g == reference
    ref_c = counts(y[ref_mask], yhat[ref_mask]); ref_m = fairness_metrics(*ref_c); ref_d = _denominators(*ref_c)

    # bootstrap: resample rows inside each group, recompute gap = group - reference
    ref_idx = np.flatnonzero(ref_mask)
    ref_boot = np.array([list(fairness_metrics(*counts(y[i], yhat[i])).values())
                         for i in (rng.choice(ref_idx, len(ref_idx), replace=True) for _ in range(B))])
    rows = []
    for grp in groups:
        m = g == grp
        c = counts(y[m], yhat[m]); met = fairness_metrics(*c); den = _denominators(*c)
        row = {"group": grp, "n": int(m.sum()), "base_rate": float(y[m].mean()), "small": bool(m.sum() < min_n)}
        row.update(met)
        if grp == reference:
            rows.append(row); continue
        idx = np.flatnonzero(m)
        boot = np.array([list(fairness_metrics(*counts(y[i], yhat[i])).values())
                         for i in (rng.choice(idx, len(idx), replace=True) for _ in range(B))])
        gaps = boot - ref_boot
        for k, name in enumerate(METRICS):
            gap = met[name] - ref_m[name]
            col = gaps[:, k][~np.isnan(gaps[:, k])]
            lo, hi = (np.percentile(col, [2.5, 97.5]) if len(col) else (float("nan"), float("nan")))
            z, p = two_prop_z(ref_m[name], ref_d[name], met[name], den[name])
            row.update({f"gap_{name}": gap, f"ci_lo_{name}": lo, f"ci_hi_{name}": hi, f"z_{name}": z, f"p_{name}": p})
        row["adverse_impact_ratio"] = met["selection_rate"] / ref_m["selection_rate"] if ref_m["selection_rate"] else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(1, "reference", reference)
    return out


def min_detectable_gap(n1, n2, p=0.15, alpha=0.05, power=0.80):
    """Smallest two-proportion difference detectable at the given alpha and power (two-sided)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)))
