"""SMOKE-TEST ONLY: write a small SYNTHETIC file in the public HMDA LAR layout.

It exists so the pipeline can be exercised end to end before the real download is in place.
The numbers it produces mean nothing about real lending and must never appear in the report.

Run:  python src/make_fixture.py --year 2024 --n 20000

Help received: Claude (Anthropic AI assistant) drafted this script; Anukrithi Myadala reviewed it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def make(year: int, n: int, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed + year)
    race = rng.choice(["White", "Asian", "Black or African American", "Joint", "Race Not Available",
                       "2 or more minority races", "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander"],
                      n, p=[0.40, 0.25, 0.04, 0.06, 0.20, 0.02, 0.02, 0.01])
    eth = rng.choice(["Not Hispanic or Latino", "Hispanic or Latino", "Ethnicity Not Available", "Joint"], n, p=[0.62, 0.17, 0.17, 0.04])
    sex = rng.choice(["Male", "Female", "Joint", "Sex Not Available"], n, p=[0.35, 0.27, 0.28, 0.10])
    age = rng.choice(["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74", "8888"], n, p=[0.04, 0.30, 0.30, 0.17, 0.10, 0.05, 0.02, 0.02])
    income = np.round(np.exp(rng.normal(np.log(150), 0.55, n)))            # thousands of dollars
    prop = np.round(np.exp(rng.normal(np.log(800_000), 0.45, n)) / 10_000) * 10_000
    cltv = np.clip(rng.normal(78, 12, n), 20, 105)
    loan = np.round(prop * cltv / 100 / 5_000) * 5_000
    dti_num = np.clip(rng.normal(38, 10, n), 5, 75)
    dti = np.where(dti_num < 20, "<20%", np.where(dti_num < 30, "20%-<30%", np.where(dti_num < 36, "30%-<36%",
          np.where(dti_num <= 49, np.floor(dti_num).astype(int).astype(str), np.where(dti_num <= 60, "50%-60%", ">60%")))))
    logit = (2.6 - 0.09 * (dti_num - 36) - 0.03 * (cltv - 80) + 0.25 * np.log(income / 150)
             + rng.normal(0, 0.9, n) - 0.15 * (race == "Black or African American") - 0.05 * (eth == "Hispanic or Latino"))
    approved = rng.random(n) < 1 / (1 + np.exp(-logit))
    co_sex = rng.choice(["1", "2", "5"], n, p=[0.25, 0.30, 0.45])

    df = pd.DataFrame({
        "activity_year": year, "lei": rng.choice([f"LEI{i:04d}" for i in range(60)], n), "derived_msa-md": "41940",
        "state_code": "CA", "county_code": rng.choice(["06085", "06001", "06037", "06073"], n), "census_tract": "06085500100",
        "conforming_loan_limit": np.where(loan <= 766_550, "C", "NC"), "derived_loan_product_type": "Conventional:First Lien",
        "derived_dwelling_category": "Single Family (1-4 Units):Site-Built", "derived_ethnicity": eth, "derived_race": race,
        "derived_sex": sex, "action_taken": np.where(approved, "1", "3"), "purchaser_type": np.where(approved, "1", "0"),
        "preapproval": rng.choice(["1", "2"], n, p=[0.1, 0.9]), "loan_type": "1", "loan_purpose": "1", "lien_status": "1",
        "reverse_mortgage": "2", "open-end_line_of_credit": "2", "business_or_commercial_purpose": "2",
        "loan_amount": loan.astype(int), "combined_loan_to_value_ratio": np.round(cltv, 2).astype(str),
        "interest_rate": np.where(approved, "6.5", "NA"), "rate_spread": np.where(approved, "0.3", "NA"), "hoepa_status": "3",
        "total_loan_costs": np.where(approved, "5000", "NA"), "total_points_and_fees": "NA", "origination_charges": "NA",
        "discount_points": "NA", "lender_credits": "NA", "loan_term": "360", "prepayment_penalty_term": "NA",
        "intro_rate_period": "NA", "negative_amortization": "2", "interest_only_payment": "2", "balloon_payment": "2",
        "other_nonamortizing_features": "2", "property_value": prop.astype(int).astype(str), "construction_method": "1",
        "occupancy_type": "1", "manufactured_home_secured_property_type": "3", "manufactured_home_land_property_interest": "5",
        "total_units": "1", "multifamily_affordable_units": "NA", "income": income.astype(int).astype(str),
        "debt_to_income_ratio": dti, "applicant_credit_score_type": rng.choice(["1", "2", "3", "9"], n, p=[0.3, 0.3, 0.3, 0.1]),
        "co-applicant_credit_score_type": np.where(co_sex == "5", "10", "2"), "applicant_ethnicity-1": "2", "co-applicant_ethnicity-1": "2",
        "applicant_race-1": "5", "co-applicant_race-1": "5", "applicant_sex": "1", "co-applicant_sex": co_sex,
        "applicant_age": age, "applicant_age_above_62": np.where(np.isin(age, ["65-74", ">74"]), "Yes", "No"),
        "co-applicant_age": "9999", "co-applicant_age_above_62": "NA", "submission_of_application": rng.choice(["1", "2"], n, p=[0.8, 0.2]),
        "initially_payable_to_institution": rng.choice(["1", "2"], n, p=[0.85, 0.15]), "aus-1": rng.choice(["1", "2", "6"], n, p=[0.55, 0.35, 0.10]),
        "denial_reason-1": np.where(approved, "10", rng.choice(["1", "3", "4"], n)), "denial_reason-2": "", "denial_reason-3": "", "denial_reason-4": "",
        "tract_population": rng.integers(2000, 9000, n), "tract_minority_population_percent": np.round(rng.uniform(10, 95, n), 2),
        "ffiec_msa_md_median_family_income": 180_000, "tract_to_msa_income_percentage": np.round(rng.uniform(40, 220, n)),
        "tract_owner_occupied_units": rng.integers(300, 3000, n), "tract_one_to_four_family_homes": rng.integers(400, 3500, n),
        "tract_median_age_of_housing_units": rng.integers(10, 70, n),
    })
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"SYNTHETIC_fixture_{year}.csv"
    df.to_csv(out, index=False)
    print(f"*** SYNTHETIC fixture written: {out} ({len(df):,} rows, approval rate {approved.mean():.3f}) — not real data ***")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    make(a.year, a.n)
