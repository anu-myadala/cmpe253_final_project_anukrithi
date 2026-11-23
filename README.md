# CMPE 253 project (Track B) — fair-lending model-risk pipeline on public HMDA data

Anukrithi Myadala · San José State University · Fall 2026

A reproducible privacy / fairness / explainability / robustness evaluation of mortgage-approval
models trained on the public HMDA loan-level data (California, conventional home-purchase
applications, 2024 for development, 2025 held out for drift). Everything runs offline on public
data; no live system is touched. Sensitive attributes are evaluation-only, never model inputs.

## Reproduce everything

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # pinned versions
make download        # 1. pull the 2024 and 2025 CSVs from the CFPB Data Browser API (state-wide, minutes)
make all             # 2. every stage in order (prepare ... risk register -> docs), about 20 minutes on a 2-core machine
make dashboard       # 4. demo: streamlit app over results/ (plus a what-if scorer)
```

Tested with Python 3.13 (Nov 2026 rerun; every table except the fairlearn reductions rows is identical to the October run).
The download goes through `curl` because the API redirects to a cached file and rejects Python's default client.

A `Dockerfile` is included for a containerised run (see its header for the commands).
`make all` takes about 15 minutes on a 2-core laptop; the slow step is the exponentiated-gradient
reduction for LightGBM. Each stage can be run alone (`make ablations`, ...). Every run writes a
`results/run_log*.json` with the config snapshot, package versions, split sizes and timings; the seed
(42) is fixed in `config.yaml`, which is the single source of truth for every knob.

The HMDA Data Browser API accepts at most two filter criteria, so the download uses home-purchase +
originated/denied and `prepare.py` applies the conventional-loan and other cohort rules locally.

Smoke test without the real data (numbers are meaningless, never report them): `make smoke`.

## What each stage measures

| Stage (script) | Track B element | Main outputs in `results/` |
|---|---|---|
| `prepare.py` | data pipeline | `funnel_*.csv`, `cohort_summary_*.csv`, `missingness_*.csv`, `data/clean/*.parquet` |
| `leakage_audit.py` | engineering / validity | `leakage_audit_*.csv` — flags any input category that is 0 % or 100 % approved; found two post-decision fields the keyword list missed |
| `baselines.py` | ≥ 2 baselines, statistical rigor | `baseline_overall.csv`, `fairness_<model>_<attr>.csv` (gaps, bootstrap CIs, z-tests, AIR), `intersectional_*.csv`, `power_note.csv`, ROC / TPR-FPR plots |
| `drift_check.py` | robustness: drift | `drift_2024_to_2025.csv` (PSI, KS), `drift_performance_*.csv`, `fairness_*_2025.csv` |
| `ablations.py` | ablations, fairness interventions | `ablation_summary.csv`, `ablation_<model>_<variant>_<attr>.csv`, `ablation_mcnemar.csv`, `ablation_gaps.png` |
| `explain.py` | explainability: local + global, stability, faithfulness | `explain_global_*.csv/png`, `explain_local_examples.csv`, `explain_stability.csv`, `explain_faithfulness.csv`, `explain_sanity.csv`, `explain_gap_decomposition.csv/png` |
| `robustness.py` | robustness: perturbation, OOD, adversarial | `robustness_noise*.csv/png`, `robustness_ood*.csv`, `robustness_pgd.csv`, `robustness_gaming.csv` |
| `dp_pilot.py`, `privacy.py` | privacy: ε and leakage probe | `dp_pilot.csv`, `privacy_mi.csv`, `privacy_dp_grid.csv`, `privacy_dp_fairness.csv`, `privacy_uniqueness.csv`, `privacy_dp.png` |
| `extras.py` | explainability + fairness: reason coverage, proxy recoverability | `reason_coverage.csv`, `reason_coverage_drivers.csv`, `proxy_recoverability.csv` |
| `rigor.py` | statistical rigor, evaluation design | `rigor_repeated_*.csv` (10 split seeds), `rigor_delong.csv`, `rigor_multiple_testing.csv` (Holm / BH), `rigor_lender_holdout.csv`, `rigor_duplicates.csv`, `manifest.json` |
| `adv_debias.py` | fairness: in-processing | `adv_debias.csv/png` — adversarial debiasing (Zhang et al. 2018) in NumPy |
| `human_ai.py` | human oversight, calibration | `human_ai_calibration.csv`, `human_ai_routing.csv`, reliability and routing plots |
| `recourse.py` | explainability: counterfactuals, fairness of recourse | `recourse_by_group.csv`, `recourse_applicants.csv`, `recourse_cost.png` |
| `hardening.py` | robustness fix | `hardening.csv/png` — monotone, regularised, noise-augmented LightGBM |
| `xai_lime.py` | explainability: second local method | `xai_lime_agreement.csv`, `xai_lime_summary.csv` |
| `risk_register.py` | governance | `risk_register_measured.csv` — R1–R11 with evidence, MITRE ATLAS / NIST AI RMF mapping, proposed residual scores |
| `make_docs.py` | documentation | `docs/model_card.md`, `docs/datasheet.md` |
| `app/dashboard.py` | demo | Streamlit dashboard over `results/` |

## Layout

```
config.yaml            all settings (cohort, target, features, leakage drop, split, seed, thresholds, models, ablations, DP)
Makefile               reproducible run order
Dockerfile             containerised run (optional)
src/download_hmda.py   CFPB Data Browser API download
src/prepare.py         cohort construction, leakage drop, feature / target / sensitive-attribute build
src/leakage_audit.py   label-leakage audit of every input category (fails the build when an input leaks)
src/fairness.py        fairness_metrics (lecture definition), gaps, z-tests, bootstrap CIs, power note
src/baselines.py       two baselines + evaluation tables and plots
src/drift_check.py     2024 -> 2025 drift check (PSI, KS, AUC, gaps)
src/ablations.py       reweighing, equalized-odds thresholds, exponentiated-gradient reductions, proxy features, McNemar
src/explain.py         SHAP global / local, bootstrap stability, deletion faithfulness, label-shuffle sanity, gap decomposition
src/robustness.py      noise perturbation, adversarial validation (OOD), PGD + transfer, application-gaming probe
src/dp_pilot.py        DP-SGD logistic regression over the ε grid with an RDP accountant + MI probe (pilot)
src/privacy.py         membership-inference probes (confidence, shadow model), DP grid over seeds with fairness, record uniqueness
src/extras.py          Reg. B reason coverage of model denials; race / ethnicity recoverability with and without tract proxies
src/rigor.py           10 split seeds, DeLong, Holm / BH, unseen-lender split, duplicate check, data manifest
src/adv_debias.py      adversarial debiasing (in-processing) in NumPy
src/human_ai.py        calibration by group, abstain-to-human routing
src/recourse.py        counterfactual recourse and recourse gaps by group
src/hardening.py       hardened LightGBM vs realistic input errors and gaming
src/xai_lime.py        LIME vs SHAP agreement
src/risk_register.py   measured risk register with MITRE ATLAS / NIST AI RMF mapping
src/make_docs.py       model card and datasheet generated from results/
src/make_fixture.py    synthetic LAR-layout file for smoke tests only
app/dashboard.py       Streamlit demo
data/raw, data/clean   downloads and cleaned parquet files (not committed)
results/               tables (csv), plots (png), run logs (json) — committed so the numbers in the report can be checked
docs/                  model card, datasheet
```

## Acknowledgement

Code drafted with AI assistance (Claude); reviewed, run and modified by the author.
