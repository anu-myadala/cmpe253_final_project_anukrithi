"""Generate docs/model_card.md and docs/datasheet.md from config.yaml and results/ so every number in
them is the number the pipeline produced. Factual fields are filled automatically; the judgement
fields are left as [TODO] markers for the author to write (they are hers, not the pipeline's).

Run:  python src/make_docs.py          (after `make all`)
"""
import json
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
Y, YD = CFG["years"]["train_test"], CFG["years"]["drift"]


def csv(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def gap_line(rep, group, metric="selection_rate"):
    r = rep.set_index("group").loc[group]
    return f"{100 * r[f'gap_{metric}']:+.1f} pp [{100 * r[f'ci_lo_{metric}']:+.1f}, {100 * r[f'ci_hi_{metric}']:+.1f}] (n = {int(r['n']):,}, p = {r[f'p_{metric}']:.3g})"


def main():
    DOCS.mkdir(exist_ok=True)
    base = csv("baseline_overall.csv").set_index("model")
    funnel = csv(f"funnel_{Y}.csv"); funnel_d = csv(f"funnel_{YD}.csv")
    log = json.loads((RESULTS / "run_log.json").read_text())
    race = {m: csv(f"fairness_{m}_derived_race.csv") for m in base.index}
    eth = {m: csv(f"fairness_{m}_derived_ethnicity.csv") for m in base.index}
    sex = {m: csv(f"fairness_{m}_derived_sex.csv") for m in base.index}
    race25 = csv("fairness_lightgbm_derived_race_2025.csv")
    drift_perf = csv("drift_performance_2024_to_2025.csv")
    ab = csv("ablation_summary.csv"); mi = csv("privacy_mi.csv"); dpg = csv("privacy_dp_grid.csv")
    noise = csv("robustness_noise.csv"); ood = csv("robustness_ood.csv"); gaming = csv("robustness_gaming.csv")
    glob = {m: csv(f"explain_global_{m}.csv") for m in base.index}
    stab = csv("explain_stability.csv"); sanity = csv("explain_sanity.csv"); audit = csv(f"leakage_audit_{Y}.csv")
    n = log["n"]; versions = log["versions"]
    todo = lambda what: f"> [TODO - author: {what}]"

    def metrics_table():
        rows = ["| Model | AUC | Accuracy | Balanced acc. | Brier | Threshold | Selection rate |", "|---|---|---|---|---|---|---|"]
        for m, r in base.iterrows():
            rows.append(f"| {m.replace('_', ' ')} | {r['auc']:.3f} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['brier']:.3f} | {r['threshold']:.3f} | {r['selection_rate']:.3f} |")
        return "\n".join(rows)

    def gap_table():
        rows = ["| Model | Black vs White (selection) | Black vs White (TPR) | Hispanic vs non-Hispanic (selection) | Female vs Male (selection) |", "|---|---|---|---|---|"]
        for m in base.index:
            rows.append(f"| {m.replace('_', ' ')} | {gap_line(race[m], 'Black or African American')} | {gap_line(race[m], 'Black or African American', 'tpr')} | "
                        f"{gap_line(eth[m], 'Hispanic or Latino')} | {gap_line(sex[m], 'Female')} |")
        return "\n".join(rows)

    def ablation_table():
        if ab is None:
            return "_run `make ablations`_"
        rows = ["| Model | Variant | AUC | Accuracy | Balanced acc. | Selection rate | Black-White gap (pp) [95 % CI] | Hispanic gap (pp) | McNemar p vs baseline |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in ab.iterrows():
            auc = "n/a (labels only)" if pd.isna(r["auc"]) else f"{r['auc']:.3f}"
            rows.append(f"| {r['model'].replace('_', ' ')} | {r['variant']} | {auc} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['selection_rate']:.3f} | "
                        f"{100 * r['gap_selection_rate_Black']:+.1f} [{100 * r['ci_lo_selection_rate_Black']:+.1f}, {100 * r['ci_hi_selection_rate_Black']:+.1f}] | "
                        f"{100 * r['gap_selection_rate_Hispanic']:+.1f} | {r['mcnemar_p_vs_baseline']:.3g} |")
        return "\n".join(rows)

    def privacy_lines():
        out = []
        if mi is not None:
            for _, r in mi.iterrows():
                out.append(f"- {r['model'].replace('_', ' ')}, {r['attack']}: attack AUC {r['attack_auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], best advantage {r['best_advantage_tpr_minus_fpr']:.3f}; train-test AUC gap {r['train_auc_minus_test_auc']:+.4f}")
        if dpg is not None:
            ref = dpg[dpg.epsilon == float("inf")].iloc[0]
            out.append(f"- DP-SGD logistic regression (delta = {ref['delta']:.1e}, {int(ref['n_seeds'])} seeds): non-private AUC {ref['auc_mean']:.3f}; " +
                       "; ".join(f"eps = {r['epsilon']:g}: {r['auc_mean']:.3f} +- {r['auc_sd']:.3f}" for _, r in dpg[dpg.epsilon != float('inf')].iterrows()))
        return "\n".join(out) if out else "_run `make privacy`_"

    def robustness_lines():
        out = []
        if noise is not None:
            for m in base.index:
                s = noise[(noise.model == m) & (noise.perturbation == "gaussian_numeric_sd_units")]
                out.append(f"- {m.replace('_', ' ')}: decisions flipped by Gaussian input noise at 0.1 / 0.25 / 0.5 SD = " +
                           " / ".join(f"{100 * v:.1f} %" for v in s.sort_values('level')['flip_rate'].values[[1, 2, 3]]) +
                           f"; AUC at 0.5 SD {s[s.level == 0.5]['auc'].iloc[0]:.3f}")
        if ood is not None:
            out.append(f"- Adversarial validation {Y} test vs {YD}: domain-classifier AUC {ood['domain_auc'].iloc[0]:.3f} (0.5 = indistinguishable); top separating inputs: {ood['top_separating_inputs'].iloc[0]}")
        if gaming is not None:
            for m in base.index:
                g = gaming[(gaming.model == m) & (gaming.manipulation == "income over-stated")]
                for handling, lab in [("DTI bin held fixed", "DTI bin held fixed"), ("DTI bin moved", "DTI bin moved with income")]:
                    gg = g[g.dti_handling.astype(str).str.startswith(handling)].set_index("change")["flipped_to_approve_share"]
                    out.append(f"- {m.replace('_', ' ')}: share of model-denied applications flipped by an income over-statement of +10 % / +20 % / +50 % ({lab}): " +
                               " / ".join(f"{100 * gg.get(c, float('nan')):.1f} %" for c in ["+10 %", "+20 %", "+50 %"]))
        if drift_perf is not None:
            for _, r in drift_perf.iterrows():
                out.append(f"- Drift {Y} -> {YD}, {r['model'].replace('_', ' ')} at the frozen threshold: AUC {r['auc_2024_test']:.3f} -> {r['auc_2025']:.3f}, selection rate {r['selection_rate_2024_test']:.3f} -> {r['selection_rate_2025']:.3f}")
        return "\n".join(out) if out else "_run `make robustness drift`_"

    def explain_lines():
        out = []
        for m in base.index:
            if glob[m] is not None:
                top = glob[m].head(5)
                out.append(f"- {m.replace('_', ' ')} top-5 inputs by mean |SHAP|: " + ", ".join(f"{a} ({100 * b:.0f} %)" for a, b in zip(top['input'], top['share'])))
        if stab is not None:
            s = stab.groupby("model")[["spearman_rho_vs_reference", "top5_overlap"]].mean()
            out.append("- Stability over 5 bootstrap refits (Spearman rho of global importances / top-5 overlap): " + "; ".join(f"{m.replace('_', ' ')} {r['spearman_rho_vs_reference']:.3f} / {r['top5_overlap']:.2f}" for m, r in s.iterrows()))
        if sanity is not None:
            out.append("- Label-shuffle sanity check (mean of 3 permutations): " + "; ".join(f"{r['model'].replace('_', ' ')} AUC {r['auc_real_labels']:.3f} -> {r['auc_shuffled_labels_mean']:.3f} +- {r['auc_shuffled_labels_sd']:.3f} on shuffled labels, total |SHAP| mass {r['total_mean_abs_shap_real']:.2f} -> {r['total_mean_abs_shap_shuffled_mean']:.2f}, importance rank correlation {r['spearman_rho_shuffled_vs_real_mean']:.2f}" for _, r in sanity.iterrows()))
        return "\n".join(out) if out else "_run `make explain`_"

    def extra_lines():
        out = []
        rs = csv("rigor_repeated_summary.csv")
        if rs is not None:
            for m in base.index:
                g = rs[rs.model == m].set_index("metric")
                out.append(f"- {m.replace('_', ' ')} over 10 split seeds (mean [95% CI]): AUC {g.loc['auc', 'mean']:.3f} [{g.loc['auc', 'ci_lo']:.3f}, {g.loc['auc', 'ci_hi']:.3f}]; "
                           f"Black-White selection gap {100 * g.loc['Black_sel_gap', 'mean']:+.1f} pp [{100 * g.loc['Black_sel_gap', 'ci_lo']:+.1f}, {100 * g.loc['Black_sel_gap', 'ci_hi']:+.1f}]; "
                           f"Hispanic gap {100 * g.loc['Hispanic_sel_gap', 'mean']:+.1f} pp [{100 * g.loc['Hispanic_sel_gap', 'ci_lo']:+.1f}, {100 * g.loc['Hispanic_sel_gap', 'ci_hi']:+.1f}]")
        dl = csv("rigor_delong.csv")
        if dl is not None:
            r = dl.iloc[2]
            out.append(f"- LightGBM - LR AUC (DeLong): {r['estimate']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], p = {r['p']:.1e}")
        mt = csv("rigor_multiple_testing.csv")
        if mt is not None:
            out.append(f"- Multiple testing over {len(mt)} gap tests: {int(mt.sig_raw.sum())} significant at 0.05 raw, {int(mt.sig_bh.sum())} after Benjamini-Hochberg, {int(mt.sig_holm.sum())} after Holm")
        lh = csv("rigor_lender_holdout.csv")
        if lh is not None:
            for _, r in lh[lh.split.str.startswith("lender")].iterrows():
                out.append(f"- Unseen-lender split ({int(r['n_test_lenders'])} lenders held out), {r['model'].replace('_', ' ')}: AUC {r['auc']:.3f}")
        hd = csv("hardening.csv")
        if hd is not None:
            for _, r in hd.iterrows():
                out.append(f"- LightGBM {r['variant']}: AUC {r['auc']:.3f}; decisions flipped by +/-5 % income / appraisal error {100 * r['flip_realistic_5pct']:.1f} %; denials flipped by income +10 % {100 * r['gaming_income_plus10_dti_fixed']:.1f} %")
        ad = csv("adv_debias.csv")
        if ad is not None:
            for _, r in ad[ad.alpha.isin([0, 2])].iterrows():
                out.append(f"- Adversarial debiasing alpha = {r['alpha']:g}: AUC {r['auc']:.3f}, adversary AUC {r['adversary_auc_test']:.3f}, Black gap {100 * r['Black_sel_gap']:+.1f} pp, Hispanic gap {100 * r['Hispanic_sel_gap']:+.1f} pp, Hispanic TPR gap {100 * r['Hispanic_tpr_gap']:+.1f} pp")
        lm = csv("xai_lime_summary.csv")
        if lm is not None:
            v = lm.set_index("metric")["mean"]
            out.append(f"- SHAP vs LIME on 100 denials: same principal reason {100 * v['same_principal_driver']:.0f} %, top-3 overlap {v['top3_overlap_shap_lime']:.2f}; LIME seed-to-seed top-3 overlap {v['top3_overlap_lime_seed_a_b']:.2f}")
        rc = csv("recourse_by_group.csv")
        if rc is not None:
            for _, r in rc[rc.group.str.startswith("GAP")].iterrows():
                out.append(f"- Recourse {r['model'].replace('_', ' ')}, {r['group'][4:]}: cost gap {r['mean_cost_if_feasible']:+.2f} [{r['cost_gap_ci_lo']:+.2f}, {r['cost_gap_ci_hi']:+.2f}] effort units")
        return "\n".join(out) if out else "_run the week-14 stages_"

    leaks = audit[audit["deterministic"]] if audit is not None else None
    leak_lines = "\n".join(f"- `{r['field']}` = {r['category']}: {int(r['n']):,} rows, approval rate {100 * r['approval_rate']:.0f} % ({r['status']})" for _, r in leaks.iterrows()) if leaks is not None else ""

    card = f"""# Model card - mortgage-approval models used in the CMPE 253 fair-lending model-risk pipeline

_Generated by `src/make_docs.py` from `config.yaml` and `results/` on {time.strftime('%Y-%m-%d')}; the structure follows Mitchell et al. (2019). Numbers are the pipeline's; the judgement fields are the author's._

## Model details
- Two baselines trained on the same inputs and split: scikit-learn logistic regression (C = {CFG['models']['logistic_regression']['C']}) and LightGBM ({CFG['models']['lightgbm']['n_estimators']} trees, learning rate {CFG['models']['lightgbm']['learning_rate']}, {CFG['models']['lightgbm']['num_leaves']} leaves).
- Inputs ({len(CFG['features']['numeric']) + len(CFG['features']['categorical'])}): numeric {', '.join(CFG['features']['numeric'])}; categorical {', '.join(CFG['features']['categorical'])}. Sensitive attributes ({', '.join(CFG['sensitive'])}) are never inputs. Census-tract proxies are off except in the proxy ablation.
- Preprocessing: median imputation + standardisation (numeric), one-hot with minimum frequency 20 (categorical). Seed {CFG['seed']}. Versions: {', '.join(f'{k} {v}' for k, v in versions.items())}.
- Author: Anukrithi Myadala (SJSU, CMPE 253, Fall 2026). Research/teaching artefact, not a production model.

## Intended use
- Primary use: a measurement test-bed for model-risk evaluation (fairness, privacy, explainability, robustness) on public data.
- Out of scope: any real credit decision. The label is the lender's historical decision, not creditworthiness, so the models reproduce historical practice.
{todo('one paragraph on who the intended users of such an evaluation would be (model-risk / fair-lending teams) and what they would and would not do with it')}

## Factors
- Evaluation groups: race, ethnicity, sex, age (HMDA derived fields) and race x sex slices; reference groups {', '.join(f'{k} = {v}' for k, v in CFG['reference_groups'].items())}; slices under {CFG['min_group_n']} applications are reported but not interpreted.
{todo('note the relevant factors the public file cannot show (credit score, assets, employment) and how that limits attribution')}

## Metrics
- Overall: AUC, accuracy, balanced accuracy, Brier score at a threshold chosen on the validation split ({CFG['threshold_rule']}).
- Per group: selection rate, TPR, FPR, precision; gaps vs the reference with bootstrap 95 % CIs (B = {CFG['bootstrap_B']}) and two-proportion z-tests; adverse-impact ratio; power note.

## Training and evaluation data
- HMDA public LAR, California, {Y}: {int(funnel['rows'].iloc[0]):,} downloaded rows -> {int(funnel['rows'].iloc[-1]):,} in the cohort (conventional, home purchase, originated or denied, owner-occupied, first lien, site-built, 1-4 units); approval rate {100 * base['base_rate'].iloc[0]:.1f} %.
- Split: train {n['train']:,} / validation {n['val']:,} / test {n['test']:,} (stratified). Drift year {YD}: {int(funnel_d['rows'].iloc[-1]):,} rows, never used for training.
- Leakage controls: post-decision fields dropped by keyword list plus an automated audit of every input category; the audit caught two structural leaks that the keyword list missed:
{leak_lines}

## Quantitative analyses
### Performance on the {Y} test split
{metrics_table()}

### Fairness gaps vs the reference group ({Y} test split)
{gap_table()}

Black vs White selection-rate gap on the {YD} cohort with the frozen LightGBM model: {gap_line(race25, 'Black or African American') if race25 is not None else 'run `make drift`'}.

### Interventions (ablations)
{ablation_table()}

### Explainability
{explain_lines()}

### Robustness and drift
{robustness_lines()}

### Privacy
{privacy_lines()}

### Additional evaluations (rigor, in-processing, hardening, explanation agreement, recourse)
{extra_lines()}

## Ethical considerations
{todo('discuss the label-as-historical-decision problem, proxy discrimination through geography, the ECOA/Reg B constraint on using protected class at decision time, and the small-group power limits')}

## Caveats and recommendations
{todo('what a model-risk team should and should not conclude from these numbers; which monitoring (drift, gaps on the later year) you would require; what would change with credit-score data')}
"""
    (DOCS / "model_card.md").write_text(card)

    sheet = f"""# Datasheet - HMDA California conventional home-purchase cohort used in the CMPE 253 project

_Generated by `src/make_docs.py` on {time.strftime('%Y-%m-%d')}; structure follows Gebru et al. (2021). Facts from `config.yaml` and `results/`; judgement fields are the author's._

## Motivation
- Built for the CMPE 253 Track B project to measure fairness, privacy, explainability and robustness of mortgage-approval models on public data. Created by Anukrithi Myadala (SJSU); no funding.
{todo('one or two sentences on why HMDA specifically (public, loan-level, protected attributes present, regulatory relevance)')}

## Composition
- Instances are loan applications (one row each) from the public HMDA Loan/Application Register, state = {CFG['state']}, action taken in {{originated, denied}}, loan purpose = home purchase, then filtered to {', '.join(f'{k} in {v}' for k, v in CFG['cohort'].items())}.
- {Y}: {int(funnel['rows'].iloc[-1]):,} applications ({100 * base['base_rate'].iloc[0]:.1f} % originated); {YD}: {int(funnel_d['rows'].iloc[-1]):,}.
- Fields kept: {len(CFG['features']['numeric'])} numeric and {len(CFG['features']['categorical'])} categorical inputs, {len(CFG['sensitive'])} sensitive attributes (evaluation only), {len(CFG['tract_proxies'])} census-tract proxies (ablation only), lender id and county (privacy uniqueness check only).
- Label: y = 1 originated, 0 denied - the lender's decision, not ground-truth creditworthiness. No gold standard exists in the file.
- Missingness is logged per numeric input in `results/missingness_{Y}.csv`; DTI is reported by HMDA in bins (36-49 as integers), binned here with `prepare.bin_dti`.
- Sensitive content: protected-class fields are present by design of the public release; no direct identifiers (names, addresses) exist in the file.

## Collection process
- Downloaded from the CFPB/FFIEC HMDA Data Browser API (`{CFG['api_base']}`) with filters {CFG['api_filters']}; the API allows two filter criteria, so the remaining cohort rules run locally in `src/prepare.py`. Funnel counts per filter are in `results/funnel_{Y}.csv`.
- The data are self-reported by lenders under HMDA / Regulation C and published after CFPB disclosure modifications (some fields binned or redacted).

## Preprocessing / cleaning / labelling
- `src/prepare.py` applies the cohort filters, drops post-decision fields ({len(CFG['leakage_drop'])} fields in `leakage_drop`, including the two found by `src/leakage_audit.py`), converts numerics, bins DTI, builds `co_applicant_present`, keeps sensitive attributes as evaluation-only columns. Raw files are kept unmodified in `data/raw/`.

## Uses
- Used here for: baseline approval models, subgroup fairness evaluation, fairness interventions, SHAP explanations, robustness probes, membership-inference and DP experiments, 2024 -> 2025 drift.
- Should not be used for: any real lending decision; any claim about an individual lender's intent; inference about individual applicants.
{todo('note the re-identification result from results/privacy_uniqueness.csv and what it implies for redistributing the cohort')}

## Distribution and maintenance
- The raw and cleaned files are not redistributed with the repository (they are public at the source and re-created by `make download prepare`). Maintained by the author for the duration of the course; the CFPB publishes new HMDA years annually.
"""
    (DOCS / "datasheet.md").write_text(sheet)
    print(f"wrote {DOCS / 'model_card.md'} and {DOCS / 'datasheet.md'}")


if __name__ == "__main__":
    main()
