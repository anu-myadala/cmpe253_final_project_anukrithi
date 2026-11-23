"""Label-leakage audit of every candidate input (run after prepare.py, before any model).

A field that is filled in AFTER the lender's decision can reveal the label. The keyword list in
config.yaml (pricing, purchaser type, denial reasons ...) catches the obvious ones; this audit catches
the structural ones: for every categorical input, and for the audit-only fields kept by prepare.py,
each category with at least `leakage_audit_min_rows` rows whose approval rate is exactly 0 % or 100 %
is flagged; it FAILS the audit when it identifies at least 1 % of one label class (smaller all-or-nothing
categories are listed as warnings - a 100-row category can be pure by composition). Numeric inputs are checked for a near-perfect single-feature AUC (> 0.95) and for
missingness that differs by label by more than 25 points.

This is what found `initially_payable_to_institution` (code 3 = not applicable, reported only when no
loan was made: 100 % denied) and `preapproval` (a denied pre-approval request is action code 7, so
inside the originated/denied cohort code 1 is 100 % originated). Both are now in `leakage_drop`.

Outputs (results/): leakage_audit_<year>.csv ; exit code 1 when an active input fails
Run:  python src/leakage_audit.py --year 2024
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baselines import CFG, RESULTS, ROOT

MIN_ROWS = int(CFG.get("leakage_audit_min_rows", 100))


def main(year):
    df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{year}.parquet")
    y = df["y"].values
    rows = []
    cats = list(CFG["features"]["categorical"]) + [c for c in df.columns if c.startswith("audit_")]
    for c in cats:
        ct = pd.crosstab(df[c].astype(str), y)
        for cat_value, r in ct.iterrows():
            n = int(r.sum()); rate = float(r.get(1, 0) / n) if n else float("nan")
            det = n >= MIN_ROWS and rate in (0.0, 1.0)
            rows.append({"field": c.replace("audit_", ""), "status": "audit-only (dropped)" if c.startswith("audit_") else "active input",
                         "category": cat_value, "n": n, "approval_rate": rate, "deterministic": det,
                         "share_of_denials_identified": float(r.get(0, 0) / (y == 0).sum()) if det and rate == 0.0 else float("nan"),
                         "share_of_originations_identified": float(r.get(1, 0) / (y == 1).sum()) if det and rate == 1.0 else float("nan")})
    for c in CFG["features"]["numeric"]:
        x = df[c]; m = x.notna()
        auc = roc_auc_score(y[m], x[m]) if m.sum() > MIN_ROWS and len(set(y[m])) > 1 else float("nan")
        auc = max(auc, 1 - auc) if not np.isnan(auc) else auc
        miss_gap = float(abs(x[y == 1].isna().mean() - x[y == 0].isna().mean()))
        rows.append({"field": c, "status": "active input", "category": "(numeric)", "n": int(m.sum()), "approval_rate": float("nan"),
                     "deterministic": bool((not np.isnan(auc) and auc > 0.95) or miss_gap > 0.25),
                     "share_of_denials_identified": float("nan"), "share_of_originations_identified": float("nan"),
                     "single_feature_auc": auc, "missingness_gap_by_label": miss_gap})
    out = pd.DataFrame(rows)
    share = out[["share_of_denials_identified", "share_of_originations_identified"]].max(axis=1)
    out["severity"] = np.where(out["deterministic"] & (share.fillna(1.0) >= 0.01), "fail", np.where(out["deterministic"], "warn", ""))
    out.to_csv(RESULTS / f"leakage_audit_{year}.csv", index=False, float_format="%.4f")
    flagged = out[out["deterministic"]]
    print(f"leakage audit {year}: {len(out)} field/category rows checked, {len(flagged)} flagged")
    if len(flagged):
        print(flagged[["field", "status", "category", "n", "approval_rate", "share_of_denials_identified", "share_of_originations_identified", "severity"]].to_string(index=False))
    active_fail = flagged[(flagged["status"] == "active input") & (flagged["severity"] == "fail")]
    if len(active_fail):
        print("\nFAIL: an active model input is deterministic for the label - add it to leakage_drop in config.yaml")
        sys.exit(1)
    print("PASS: no active input is deterministic for the label")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=CFG["years"]["train_test"])
    main(ap.parse_args().year)
