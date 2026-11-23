# CMPE 253 Track B project - reproducible run. `make all` reproduces every table and figure in results/
# from the two raw HMDA files (config.yaml holds every knob; the seed is fixed there).
PY ?= python3
YEAR_TRAIN ?= 2024
YEAR_DRIFT ?= 2025

.PHONY: all setup download prepare audit baselines drift dp ablations explain robustness privacy extras rigor human_ai recourse hardening advdebias lime risk docs figures dashboard smoke clean

all: prepare audit baselines drift dp ablations explain robustness privacy extras rigor human_ai recourse hardening advdebias lime risk docs
	@echo "all results written to results/"

setup:
	$(PY) -m pip install -r requirements.txt

download:                      ## pull the two state-wide CSVs from the CFPB Data Browser API (minutes)
	$(PY) src/download_hmda.py

prepare:                       ## cohort filters, leakage drop, feature build -> data/clean/*.parquet
	$(PY) src/prepare.py --year $(YEAR_TRAIN)
	$(PY) src/prepare.py --year $(YEAR_DRIFT)

audit:                         ## label-leakage audit; fails the build if an active input leaks the label
	$(PY) src/leakage_audit.py --year $(YEAR_TRAIN)

baselines:                     ## LR + LightGBM, fairness tables with CIs, ROC / TPR-FPR figures
	$(PY) src/baselines.py

drift:                         ## 2024 -> 2025 PSI / KS, frozen-threshold performance and gaps on the later year
	$(PY) src/drift_check.py

dp:                            ## DP-SGD pilot (tunes clip norm / learning rate on the validation split)
	$(PY) src/dp_pilot.py

ablations:                     ## reweighing, equalized-odds thresholds, reductions, proxy features, McNemar
	$(PY) src/ablations.py

explain:                       ## SHAP global/local, stability, faithfulness, label-shuffle sanity, gap decomposition
	$(PY) src/explain.py

robustness:                    ## noise perturbation, adversarial validation (OOD), PGD + transfer, gaming probe
	$(PY) src/robustness.py

privacy:                       ## membership-inference probes, DP grid over seeds with fairness, record uniqueness
	$(PY) src/privacy.py

extras:                        ## Reg. B reason coverage of denials + how well the inputs recover race (proxy check)
	$(PY) src/extras.py

rigor:                         ## 10 split seeds, DeLong, Holm/BH correction, unseen-lender split, duplicates, data manifest
	$(PY) src/rigor.py

human_ai:                      ## calibration by group + abstain-to-human routing (selective prediction)
	$(PY) src/human_ai.py

recourse:                      ## cheapest actionable change that flips each denial, by group (recourse gap)
	$(PY) src/recourse.py

hardening:                     ## monotone / regularised / noise-augmented LightGBM vs realistic input errors and gaming
	$(PY) src/hardening.py

advdebias:                     ## in-processing adversarial debiasing (Zhang et al. 2018), NumPy
	$(PY) src/adv_debias.py

lime:                          ## LIME vs SHAP agreement on the same denied applicants
	$(PY) src/xai_lime.py

risk:                          ## measured risk register (R1-R11) with MITRE ATLAS / NIST AI RMF mapping
	$(PY) src/risk_register.py

docs:                          ## model card and datasheet from results/
	$(PY) src/make_docs.py

dashboard:                     ## demo: streamlit app over results/
	$(PY) -m streamlit run app/dashboard.py

smoke:                         ## end-to-end check on a synthetic fixture (numbers are meaningless)
	$(PY) src/make_fixture.py --year $(YEAR_TRAIN) --n 20000
	$(PY) src/prepare.py --year $(YEAR_TRAIN) --raw data/raw/SYNTHETIC_fixture_$(YEAR_TRAIN).csv
	$(PY) src/baselines.py --year $(YEAR_TRAIN) --tag synthetic

clean:
	rm -f results/*.csv results/*.png results/*.json
