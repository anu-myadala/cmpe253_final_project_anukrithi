"""Turn a raw HMDA LAR download into the modelling cohort.

Steps (each one is logged as a row-count funnel so the report can show it):
  1. keep only originated (1) / denied (3) applications
  2. apply the cohort filters from config.yaml (owner-occupied, first lien, 1-4 units, ...)
  3. drop post-decision "leakage" fields (pricing, purchaser type, denial reasons)
  4. parse numeric fields, bin DTI, derive co_applicant_present, keep sensitive attributes
  5. write data/clean/cohort_<year>.parquet + summary tables in results/

Run:  python src/prepare.py --year 2024
      python src/prepare.py --year 2024 --raw data/raw/SYNTHETIC_fixture_2024.csv   (smoke test)

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
CLEAN = ROOT / "data" / "clean"
RESULTS = ROOT / "results"

DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36%-49%", "50%-60%", ">60%", "Missing"]


def to_num(s: pd.Series) -> pd.Series:
    """'NA' / 'Exempt' / blanks become NaN; everything else becomes float."""
    return pd.to_numeric(s.replace({"NA": np.nan, "Exempt": np.nan, "": np.nan}), errors="coerce")


def bin_dti(s: pd.Series) -> pd.Series:
    """HMDA reports DTI as bins except 36-49, which is reported as the exact integer."""
    def f(v):
        if v in ("NA", "Exempt", "", None) or pd.isna(v):
            return "Missing"
        if v in DTI_ORDER:
            return v
        try:
            x = float(v)
        except ValueError:
            return "Missing"
        return "36%-49%" if 36 <= x <= 49 else "Missing"
    return s.map(f)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=False)
    # the Data Browser CSV and the Snapshot file spell two fields differently; normalise both
    aliases = {"ageapplicant": "applicant_age", "loan_to_value_ratio": "combined_loan_to_value_ratio"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    return df


def prepare(year: int, raw: Path | None = None) -> pd.DataFrame:
    raw = raw or ROOT / "data" / "raw" / f"hmda_{CFG['state']}_{year}_purchase_orig_denied.csv"
    df = load_raw(raw)
    funnel = [("raw download", len(df))]

    t = CFG["target"]
    df = df[df[t["column"]].isin([t["positive"], t["negative"]])]
    funnel.append(("action_taken in {originated, denied}", len(df)))
    for col, allowed in CFG["cohort"].items():
        df = df[df[col].isin([str(a) for a in allowed])]
        funnel.append((f"{col} in {allowed}", len(df)))

    audit = {c: df[c].replace({"": "Missing", "NA": "Missing"}).astype(str) for c in CFG.get("audit_fields", []) if c in df.columns}
    df = df.drop(columns=[c for c in CFG["leakage_drop"] if c in df.columns])

    out = pd.DataFrame(index=df.index)
    out["y"] = (df[t["column"]] == t["positive"]).astype(int)
    for c in CFG["features"]["numeric"] + CFG["tract_proxies"]:
        out[c] = to_num(df[c])
    out["debt_to_income_ratio"] = pd.Categorical(bin_dti(df["debt_to_income_ratio"]), categories=DTI_ORDER)
    out["co_applicant_present"] = np.where(df["co-applicant_sex"] == "5", "no", "yes")
    for c in CFG["features"]["categorical"]:
        if c not in out.columns:
            out[c] = df[c].replace({"": "Missing", "NA": "Missing"}).astype(str)
    for c in CFG["sensitive"] + ["applicant_age_above_62"]:
        out[c] = df[c].replace({"": "Missing", "NA": "Missing", "8888": "Missing"}).astype(str)
    out["lei"] = df["lei"].values
    out["county_code"] = df["county_code"].values
    out["activity_year"] = int(year)
    for c, v in audit.items():                 # audit-only columns: documented leaks, never read by feature_lists()
        out[f"audit_{c}"] = v.values
    out = out.reset_index(drop=True)

    CLEAN.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CLEAN / f"cohort_{year}.parquet", index=False)

    pd.DataFrame(funnel, columns=["step", "rows"]).to_csv(RESULTS / f"funnel_{year}.csv", index=False)
    rows = []
    for attr in CFG["sensitive"]:
        g = out.groupby(attr, observed=True)["y"].agg(n="size", approval_rate="mean").reset_index()
        g.insert(0, "attribute", attr)
        g = g.rename(columns={attr: "group"})
        g["share"] = g["n"] / len(out)
        rows.append(g)
    summary = pd.concat(rows, ignore_index=True)
    summary["flag_small"] = summary["n"] < CFG["min_group_n"]
    summary.to_csv(RESULTS / f"cohort_summary_{year}.csv", index=False, float_format="%.4f")
    miss = out[CFG["features"]["numeric"]].isna().mean().rename("missing_share")
    miss.to_csv(RESULTS / f"missingness_{year}.csv", float_format="%.4f")

    print(f"cohort {year}: {len(out):,} rows, approval rate {out['y'].mean():.4f}")
    for step, n in funnel:
        print(f"  {n:>9,}  after: {step}")
    print(summary.to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--raw", type=Path, default=None, help="override the raw CSV path (e.g. a fixture)")
    a = ap.parse_args()
    prepare(a.year, a.raw)
