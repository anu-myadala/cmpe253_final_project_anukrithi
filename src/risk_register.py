"""Measured risk register: the initial-report register (R1-R8) plus three risks the experiments found
(R9-R11), each with the measured evidence pulled live from results/, a framework mapping (MITRE ATLAS
technique where the risk is adversarial, NIST AI RMF 1.0 subcategory for all), and a PROPOSED residual
score on the initial report's scales. R9-R11 are new, so their "initial" score is the inherent score before the control the experiments tested (likelihood and impact 1-5; 1-5 Low, 6-10 Medium, 11-15 High,
16-25 Critical). The residual scores are proposals for the author to confirm or change: they are a
judgement, the evidence column is not.

Output: results/risk_register_measured.csv

Run:  python src/risk_register.py      (after every other stage)

Help received: Claude (Anthropic AI assistant) helped design the pipeline structure and
drafted the first version of this script; Anukrithi Myadala reviewed, ran and edited it.
"""
import pandas as pd

from baselines import RESULTS


def band(s):
    return "Low" if s <= 5 else "Medium" if s <= 10 else "High" if s <= 15 else "Critical"


def r(f):
    return pd.read_csv(RESULTS / f)


def main():
    ab = r("ablation_summary.csv").set_index(["model", "variant"])
    rec = r("proxy_recoverability.csv").set_index(["target", "tract_proxies"])["recover_auc"]
    mi = r("privacy_mi.csv")
    uniq = r("privacy_uniqueness.csv")
    drift = r("drift_2024_to_2025.csv")
    f25 = r("fairness_lightgbm_derived_race_2025.csv").set_index("group")
    lh = r("rigor_lender_holdout.csv")
    rc = r("reason_coverage.csv").set_index("model")
    lime = r("xai_lime_summary.csv").set_index("metric")["mean"]
    hd = r("hardening.csv").set_index("variant")
    mt = r("rigor_multiple_testing.csv")
    rt = r("human_ai_routing.csv")
    cal = r("human_ai_calibration.csv").set_index(["model", "group"])
    rt90 = rt[(rt.model == "lightgbm") & (rt.coverage == 0.9)].iloc[0]
    female = mt[(mt.table == "fairness_lightgbm_derived_sex") & (mt.group == "Female") & (mt.metric == "selection_rate")].iloc[0]
    lgb_mi = mi[mi.model == "lightgbm"]
    rows = [
        ("R1", "Geographic proxies re-introduce race", "fairness / ECOA", 4, 5,
         f"Inputs recover race without proxies (Black vs White AUC {rec[('Black vs White', 'off')]:.3f}) and more with them "
         f"({rec[('Black vs White', 'on')]:.3f}); LightGBM Black-White selection gap {ab.loc[('lightgbm', 'baseline'), 'gap_selection_rate_Black']*100:.1f} pp "
         f"-> {ab.loc[('lightgbm', 'proxy_features'), 'gap_selection_rate_Black']*100:.1f} pp with proxies",
         "Proxies excluded; recoverability monitored; fairness gaps reported with CIs", "-", "MEASURE 2.11", 3, 5),
        ("R2", "Label bias: target is the lender's past decision", "validity / fairness", 5, 4,
         f"Observed approval Black {cal.loc[('lightgbm', 'Black'), 'observed_approval']:.3f} vs mean predicted {cal.loc[('lightgbm', 'Black'), 'mean_predicted']:.3f}: "
         "the model is less severe than the historical label, but still has a significant gap",
         "Stated limitation; base rates and calibration by group reported", "-", "MAP 2.3 / MEASURE 2.11", 4, 4),
        ("R3", "Membership inference on training records", "privacy", 2, 3,
         f"Attack AUC LR {mi[mi.model == 'logistic_regression'].attack_auc.max():.3f}, LightGBM up to {lgb_mi.attack_auc.max():.3f} "
         f"(shadow model, advantage {lgb_mi.best_advantage_tpr_minus_fpr.max():.3f}); DP-SGD at eps 2 costs about 0.01 AUC",
         "DP-SGD available; labels-only outputs; prefer LR / regularised GBM", "AML.T0024.000 (Infer Training Data Membership)", "MEASURE 2.10", 1, 3),
        ("R4", "Drift or lender mix degrades accuracy / fairness", "reliability / fairness", 4, 3,
         f"Input PSI max {drift[~drift.feature.str.startswith('score')].psi.max():.3f} (stable) but the 2025 Black-White gap is "
         f"{f25.loc['Black or African American', 'gap_selection_rate']*100:.1f} pp; unseen-lender AUC {lh[(lh.model == 'lightgbm') & lh.split.str.startswith('lender')].auc.item():.3f} "
         f"vs {lh[(lh.model == 'lightgbm') & lh.split.str.startswith('random')].auc.item():.3f}",
         "PSI/KS plus gap monitoring by year and by lender; retrain trigger", "-", "MEASURE 2.4 / MANAGE 4.1", 3, 3),
        ("R5", "Wrong or unstable adverse-action reasons (Reg. B)", "explainability / compliance", 3, 4,
         f"Standard Reg. B principal reason for {rc.loc['lightgbm', 'principal_driver_standard_reason']*100:.0f}% of LightGBM denials, both top-2 for "
         f"{rc.loc['lightgbm', 'both_top2_standard_reason']*100:.0f}%; SHAP and LIME name the same principal driver {lime['same_principal_driver']*100:.0f}% of the time",
         "SHAP (exact for trees) as the reason source; human-written reason when the driver is a process input", "-", "MEASURE 2.9", 3, 4),
        ("R6", "Applicant manipulates income / DTI", "decision integrity", 3, 3,
         f"Income +10% flips {hd.loc['baseline', 'gaming_income_plus10_dti_fixed']*100:.1f}% of LightGBM denials; hardened model (H3) "
         f"{hd.loc['H3 H2 + noise augmentation', 'gaming_income_plus10_dti_fixed']*100:.1f}%",
         "Monotone + noise-augmented model; income / DTI verification", "AML.T0043 (Craft Adversarial Data), AML.T0015 (Evade ML Model)",
         "MEASURE 2.7", 2, 3),
        ("R7", "Small groups make fairness estimates misleading", "statistical validity", 4, 2,
         f"{int(mt.sig_raw.sum())} of {len(mt)} gap tests significant at 0.05, {int(mt.sig_holm.sum())} after Holm; e.g. Female selection gap "
         f"p {female.p_raw:.4f} -> Holm {female.p_holm:.3f}",
         "Slices < 200 flagged; Holm / BH adjustment; power note", "-", "MEASURE 2.11", 2, 2),
        ("R8", "Post-decision fields leak the label", "validity / reproducibility", 3, 3,
         "Happened: two HMDA reporting artifacts gave AUC 0.867; 0.803 after removal; audit now fails the build on any leaking input",
         "Automated leakage audit gate in `make all`", "-", "MEASURE 2.5", 1, 3),
        ("R9", "Unequal human-review burden from abstention", "fairness / human oversight", 4, 2,
         f"At 90% automation Hispanic applicants are routed to manual review {rt90['routed_share_Hispanic']*100:.1f}% of the time vs "
         f"{rt90['routed_share_Not Hispanic']*100:.1f}% (White {rt90['routed_share_White']*100:.1f}%, Black {rt90['routed_share_Black']*100:.1f}%)",
         "Track routing rate by group; set review SLA per group", "-", "MEASURE 2.11 / GOVERN 3.2", 3, 2),
        ("R10", "Re-identification of public loan-level rows", "privacy", 3, 4,
         f"{uniq.share_unique.iloc[1]*100:.0f}% of 2024 rows are unique on county + loan amount + income + age + sex",
         "Publish aggregates only; never release row-level scores", "-", "MEASURE 2.10", 2, 4),
        ("R11", "Brittle decisions under small input errors", "reliability", 3, 3,
         f"+/-5% income / appraisal error flips {hd.loc['baseline', 'flip_realistic_5pct']*100:.1f}% of LightGBM decisions; hardened H3 "
         f"{hd.loc['H3 H2 + noise augmentation', 'flip_realistic_5pct']*100:.1f}% at AUC {hd.loc['H3 H2 + noise augmentation', 'auc']:.3f}",
         "H3 hardened LightGBM (monotone, regularised, noise-augmented)", "-", "MEASURE 2.5 / 2.7", 2, 3),
    ]
    out = pd.DataFrame(rows, columns=["id", "risk", "goal", "L_initial", "I_initial", "measured_evidence", "controls",
                                      "mitre_atlas", "nist_ai_rmf", "L_residual_proposed", "I_residual_proposed"])
    out["score_initial"] = out.L_initial * out.I_initial
    out["band_initial"] = out.score_initial.map(band)
    out["score_residual_proposed"] = out.L_residual_proposed * out.I_residual_proposed
    out["band_residual_proposed"] = out.score_residual_proposed.map(band)
    out.to_csv(RESULTS / "risk_register_measured.csv", index=False)
    print(out[["id", "score_initial", "band_initial", "score_residual_proposed", "band_residual_proposed", "measured_evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
