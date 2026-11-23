"""Demo dashboard for the CMPE 253 Track B project: reads results/ (no model is re-run except the
optional what-if scorer, which fits the two baselines once and caches them).

Run:  streamlit run app/dashboard.py          (or: make dashboard)
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
MODELS = {"LightGBM": "lightgbm", "Logistic regression": "logistic_regression"}
ATTRS = {"Race": "derived_race", "Ethnicity": "derived_ethnicity", "Sex": "derived_sex", "Age": "applicant_age"}

st.set_page_config(page_title="HMDA fair-lending model-risk pipeline", layout="wide")


def load(name, **kw):
    p = RESULTS / name
    if not p.exists():
        st.info(f"`results/{name}` not found - run the matching `make` target first.")
        return None
    return pd.read_csv(p, **kw)


def pct(x, digits=1):
    return "n/a" if pd.isna(x) else f"{100 * x:+.{digits}f} pp"


def image(name, caption=None, width=720):
    p = RESULTS / name
    if p.exists():
        st.image(str(p), caption=caption, width=width)


def gap_chart(rep, metric, title):
    d = rep[rep[f"gap_{metric}"].notna()].copy()
    d["gap"] = 100 * d[f"gap_{metric}"]; d["lo"] = 100 * d[f"ci_lo_{metric}"]; d["hi"] = 100 * d[f"ci_hi_{metric}"]
    d["label"] = d["group"] + " (n=" + d["n"].astype(str) + (")" if True else "")
    d["flag"] = np.where(d["small"], "below min_group_n (not interpreted)", "reported")
    base = alt.Chart(d).encode(y=alt.Y("label:N", sort=None, title=None))
    bars = base.mark_rule(color="#2a78d6").encode(x=alt.X("lo:Q", title=f"{title} vs reference (pp), 95 % bootstrap CI"), x2="hi:Q")
    pts = base.mark_point(filled=True, size=70).encode(x="gap:Q", color=alt.Color("flag:N", scale=alt.Scale(range=["#eb6834", "#2a78d6"]), title=None),
                                                       tooltip=["group", "n", alt.Tooltip("gap:Q", format=".2f"), alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#777").encode(x="x:Q")
    return (bars + pts + zero).properties(height=28 * len(d) + 30)


st.title("Fair-lending model-risk pipeline on public HMDA data")
st.caption("CMPE 253 (AI Threat Intelligence), Track B - Anukrithi Myadala, SJSU, Fall 2026. "
           f"California conventional home-purchase applications, {CFG['years']['train_test']} development year, "
           f"{CFG['years']['drift']} held out for drift. Sensitive attributes are evaluation-only; seed {CFG['seed']}.")

tabs = st.tabs(["Overview", "Fairness", "Interventions", "Explainability", "Robustness", "Privacy", "Drift", "Leakage audit", "What-if scorer",
                "Statistical rigor", "Adversarial debiasing", "Human + AI", "Recourse", "Hardening", "Risk register", "Model card"])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Cohort funnel")
        f = load(f"funnel_{CFG['years']['train_test']}.csv")
        if f is not None:
            st.dataframe(f, hide_index=True, width="stretch")
        st.subheader("Baselines on the untouched test split")
        b = load("baseline_overall.csv")
        if b is not None:
            st.dataframe(b.round(4), hide_index=True, width="stretch")
            st.caption("Threshold rule: " + CFG["threshold_rule"] + " (chosen on the validation split, frozen for the test split). "
                       "Split 60/20/20 stratified on the label.")
    with c2:
        image("roc_baselines.png", "ROC curves, test split", width=520)
        image("tpr_fpr_by_race.png", "LightGBM TPR / FPR by race", width=620)

# ---------------------------------------------------------------- Fairness
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    model = MODELS[c1.selectbox("Model", list(MODELS))]
    attr = ATTRS[c2.selectbox("Attribute", list(ATTRS))]
    year = c3.selectbox("Year", [f"{CFG['years']['train_test']} test split", f"{CFG['years']['drift']} (frozen model, drift year)"])
    name = f"fairness_{model}_{attr}" + ("" if year.startswith(str(CFG["years"]["train_test"])) else f"_{CFG['years']['drift']}") + ".csv"
    rep = load(name)
    if rep is not None:
        st.caption(f"Reference group: **{rep['reference'].iloc[0]}**. Slices under min_group_n = {CFG['min_group_n']} are shown but not interpreted.")
        cols = ["group", "n", "base_rate", "selection_rate", "tpr", "fpr", "precision", "gap_selection_rate", "ci_lo_selection_rate", "ci_hi_selection_rate",
                "p_selection_rate", "gap_tpr", "ci_lo_tpr", "ci_hi_tpr", "gap_fpr", "adverse_impact_ratio", "small"]
        st.dataframe(rep[cols].round(4), hide_index=True, width="stretch")
        a, b_ = st.columns(2)
        a.altair_chart(gap_chart(rep, "selection_rate", "Selection-rate gap"), width="stretch")
        b_.altair_chart(gap_chart(rep, "tpr", "TPR gap"), width="stretch")
    st.subheader("Intersectional slices (race x sex)")
    inter = load(f"intersectional_{model}.csv")
    if inter is not None:
        st.dataframe(inter[["group", "n", "selection_rate", "tpr", "fpr", "gap_selection_rate", "ci_lo_selection_rate", "ci_hi_selection_rate", "small"]].round(4), hide_index=True, width="stretch")
    pw = load("power_note.csv")
    if pw is not None:
        with st.expander("Power note: smallest selection-rate gap detectable at 80 % power"):
            st.dataframe(pw.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Interventions
with tabs[2]:
    s = load("ablation_summary.csv")
    if s is not None:
        show = s[["model", "variant", "uses_group_in_training", "uses_group_at_decision", "auc", "accuracy", "balanced_accuracy", "selection_rate",
                  "gap_selection_rate_Black", "ci_lo_selection_rate_Black", "ci_hi_selection_rate_Black", "gap_tpr_Black", "gap_fpr_Black", "air_Black",
                  "gap_selection_rate_Hispanic", "gap_tpr_Hispanic", "gap_selection_rate_Female", "mcnemar_p_vs_baseline", "delta_accuracy_vs_baseline"]]
        st.dataframe(show.round(4), hide_index=True, width="stretch")
        st.caption("reweighing = Kamiran & Calders (pre-processing); eo_postprocess = Hardt et al. thresholds (needs the group at decision time - an upper bound, "
                   "not deployable under ECOA/Reg B); reductions = Agarwal et al. exponentiated gradient (demographic parity, group used in training only); "
                   "reductions_rethreshold = the same classifier re-cut at the base-rate threshold; proxy_features = baseline + census-tract proxies.")
    image("ablation_gaps.png", "Black-White gaps per variant with bootstrap 95 % CIs", width=860)
    mc = load("ablation_mcnemar.csv")
    if mc is not None:
        with st.expander("McNemar paired tests (variant vs baseline, and LR vs LightGBM)"):
            st.dataframe(mc, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Explainability
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Global importance (mean |SHAP|, log-odds)")
        m = MODELS[st.selectbox("Model ", list(MODELS), key="xmodel")]
        g = load(f"explain_global_{m}.csv")
        if g is not None:
            st.altair_chart(alt.Chart(g.head(12)).mark_bar(color="#2a78d6").encode(x=alt.X("mean_abs_shap:Q", title="mean |SHAP|"),
                            y=alt.Y("input:N", sort="-x", title=None), tooltip=["input", alt.Tooltip("share:Q", format=".1%")]).properties(height=320), width="stretch")
    with c2:
        st.subheader("Why was this application denied?")
        loc = load("explain_local_examples.csv")
        if loc is not None:
            loc = loc[loc.model == m]
            case = st.selectbox("Case", loc["case"].unique())
            d = loc[loc.case == case]
            st.caption(f"Score {d['score'].iloc[0]:.3f} vs threshold {d['threshold'].iloc[0]:.3f}; lender's actual outcome: {d['actual_outcome'].iloc[0]}")
            st.dataframe(d[["rank", "input", "value", "shap_log_odds", "adverse_action_reason", "reason_status"]].round(3), hide_index=True, width="stretch")
    image("explain_gap_decomposition.png", "Which inputs carry the Black-White score gap (mean SHAP difference, log-odds)", width=860)
    c1, c2, c3 = st.columns(3)
    for col, name, title in [(c1, "explain_stability.csv", "Stability over 5 bootstrap refits"), (c2, "explain_faithfulness.csv", "Faithfulness (deletion test)"), (c3, "explain_sanity.csv", "Label-shuffle sanity check")]:
        t = load(name)
        if t is not None:
            col.subheader(title); col.dataframe(t.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Robustness
with tabs[4]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Input-noise perturbation")
        n = load("robustness_noise.csv")
        if n is not None:
            st.dataframe(n.round(4), hide_index=True, width="stretch")
    with c2:
        image("robustness_noise.png", width=640)
    st.subheader("Out-of-distribution: adversarial validation (2024 test vs 2025)")
    o = load("robustness_ood.csv"); od = load("robustness_ood_deciles.csv")
    if o is not None:
        st.metric("Domain-classifier AUC (0.5 = years indistinguishable)", f"{o['domain_auc'].iloc[0]:.3f}")
        st.caption("Most separating inputs: " + str(o["top_separating_inputs"].iloc[0]))
    if od is not None:
        st.dataframe(od.round(4), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    p = load("robustness_pgd.csv"); gm = load("robustness_gaming.csv")
    if p is not None:
        c1.subheader("PGD (L-inf, numeric inputs) and transfer"); c1.dataframe(p.round(4), hide_index=True, width="stretch")
    if gm is not None:
        c2.subheader("Application-gaming probe (model-denied applications)"); c2.dataframe(gm.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Privacy
with tabs[5]:
    mi = load("privacy_mi.csv")
    if mi is not None:
        st.subheader("Membership-inference probes against the fitted baselines")
        st.dataframe(mi.round(4), hide_index=True, width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        g = load("privacy_dp_grid.csv"); f = load("privacy_dp_fairness.csv")
        if g is not None:
            st.subheader("DP-SGD logistic regression (3 seeds)"); st.dataframe(g.round(4), hide_index=True, width="stretch")
        if f is not None:
            st.subheader("Gaps under DP"); st.dataframe(f.round(4), hide_index=True, width="stretch")
    with c2:
        image("privacy_dp.png", width=640)
    u = load("privacy_uniqueness.csv")
    if u is not None:
        st.subheader("Re-identification check on the public records (k-anonymity lens)")
        st.dataframe(u.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Drift
with tabs[6]:
    d = load("drift_2024_to_2025.csv"); dp_ = load("drift_performance_2024_to_2025.csv")
    if d is not None:
        st.subheader("PSI / KS, 2024 training deciles vs 2025"); st.dataframe(d.round(4), hide_index=True, width="stretch")
    if dp_ is not None:
        st.subheader("Frozen models scored on 2025"); st.dataframe(dp_.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Leakage audit
with tabs[7]:
    la = load(f"leakage_audit_{CFG['years']['train_test']}.csv")
    if la is not None:
        st.caption("Every category of every candidate input is checked for a 0 % / 100 % approval rate. The two fields below were caught by this audit "
                   "(not by the keyword list) and moved to leakage_drop.")
        st.dataframe(la[la["deterministic"]].round(4), hide_index=True, width="stretch")
        with st.expander("Full audit table"):
            st.dataframe(la.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- What-if scorer
with tabs[8]:
    st.caption("Fits the two baselines once (cached) and scores one application with its local SHAP explanation and the adverse-action mapping. "
               "Sensitive attributes are never inputs.")

    @st.cache_resource(show_spinner="fitting the two baselines once ...")
    def fit_models():
        from ablations import dense, make_clf
        from baselines import choose_threshold, feature_lists, make_models, split
        df = pd.read_parquet(ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet")
        num, cat = feature_lists(); cols = num + cat
        train, val, test = split(df, CFG["seed"])
        pre = make_models(num, cat, CFG["seed"])["logistic_regression"].named_steps["pre"]
        Xtr = dense(pre.fit_transform(train[cols])); Xva = dense(pre.transform(val[cols]))
        out = {}
        for m in MODELS.values():
            clf = make_clf(m, CFG["seed"]).fit(Xtr, train["y"].values)
            out[m] = (clf, choose_threshold(clf.predict_proba(Xva)[:, 1], val["y"].values))
        return pre, out, num, cat, train, Xtr.mean(axis=0)

    if (ROOT / "data" / "clean" / f"cohort_{CFG['years']['train_test']}.parquet").exists():
        pre, fitted, num, cat, train, Xtr_mean = fit_models()
        c1, c2 = st.columns(2)
        app = {}
        with c1:
            for c in num:
                med = float(train[c].median())
                app[c] = st.number_input(c, value=med, step=med / 20 if med else 1.0)
        with c2:
            for c in cat:
                opts = list(train[c].astype(str).value_counts().index)
                app[c] = st.selectbox(c, opts)
        row = pd.DataFrame([app])
        row["debt_to_income_ratio"] = pd.Categorical(row["debt_to_income_ratio"], categories=train["debt_to_income_ratio"].cat.categories)
        X = pre.transform(row[num + cat]); X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        from explain import ADVERSE_ACTION, aggregate, shap_values
        names = list(pre.get_feature_names_out())
        for label, m in MODELS.items():
            clf, thr = fitted[m]
            p = float(clf.predict_proba(X)[0, 1])
            st.markdown(f"**{label}**: score {p:.3f} vs threshold {thr:.3f} -> **{'approve' if p >= thr else 'deny'}**")
            phi, _ = shap_values(m, clf, X, Xtr_mean)
            contrib = aggregate(phi, names, (num, cat)).iloc[0].sort_values()
            neg = contrib[contrib < 0].head(4)
            if p < thr:
                best = None
                dti_order = ["<20%", "20%-<30%", "30%-<36%", "36%-49%", "50%-60%", ">60%"]
                for inc in [0, 5, 10, 15, 20, 25, 30]:
                    for loan in [0, 5, 10, 15, 20, 25, 30]:
                        for k in [0, 1, 2]:
                            cost = inc / 10 + loan / 10 + 1.5 * k
                            if cost == 0 or (best and cost >= best[0]):
                                continue
                            e = row.copy()
                            e["income"] = e["income"] * (1 + inc / 100)
                            e["loan_amount"] = e["loan_amount"] * (1 - loan / 100)
                            e["combined_loan_to_value_ratio"] = e["combined_loan_to_value_ratio"] * (1 - loan / 100)
                            v = str(e["debt_to_income_ratio"].iloc[0])
                            if k and v in dti_order:
                                e["debt_to_income_ratio"] = pd.Categorical([dti_order[max(dti_order.index(v) - k, 0)]],
                                                                           categories=train["debt_to_income_ratio"].cat.categories)
                            Xe = pre.transform(e[num + cat]); Xe = np.asarray(Xe.todense()) if hasattr(Xe, "todense") else np.asarray(Xe)
                            if float(clf.predict_proba(Xe)[0, 1]) >= thr:
                                best = (cost, inc, loan, k)
                st.markdown("Cheapest recourse inside the box: " + (f"income +{best[1]}%, loan amount -{best[2]}%, DTI {best[3]} bin(s) lower "
                            f"(effort {best[0]:.1f})" if best else "none within +30% income / -30% loan / 2 DTI bins"))
            if len(neg):
                st.dataframe(pd.DataFrame({"input": neg.index, "shap_log_odds": neg.values.round(3),
                                           "adverse_action_reason": [ADVERSE_ACTION.get(i, ("No mapping", ""))[0] for i in neg.index],
                                           "status": [ADVERSE_ACTION.get(i, ("", "unknown"))[1] for i in neg.index]}), hide_index=True, width="stretch")
    else:
        st.info("data/clean parquet not found - run `make prepare` first.")

# ---------------------------------------------------------------- Statistical rigor
with tabs[9]:
    st.caption("10 different split seeds (mean and 95% t-interval), DeLong AUC test, Holm / Benjamini-Hochberg correction of every gap test, "
               "and a lender-holdout split where no test lender was seen in training.")
    rs = load("rigor_repeated_summary.csv")
    if rs is not None:
        st.subheader("Across 10 split seeds")
        st.dataframe(rs[rs.metric.isin(["auc", "balanced_accuracy", "Black_sel_gap", "Hispanic_sel_gap", "Black_tpr_gap", "Hispanic_tpr_gap"])].round(4),
                     hide_index=True, width="stretch")
    dl = load("rigor_delong.csv")
    if dl is not None:
        st.subheader("Is LightGBM really better? (DeLong and paired t over seeds)"); st.dataframe(dl, hide_index=True, width="stretch")
    mt = load("rigor_multiple_testing.csv")
    if mt is not None:
        st.subheader(f"Multiple testing: {len(mt)} gap tests - significant raw {int(mt.sig_raw.sum())}, BH {int(mt.sig_bh.sum())}, Holm {int(mt.sig_holm.sum())}")
        st.dataframe(mt.round(5), hide_index=True, width="stretch")
    lh = load("rigor_lender_holdout.csv")
    if lh is not None:
        st.subheader("Group-aware split: unseen lenders"); st.dataframe(lh.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Adversarial debiasing
with tabs[10]:
    st.caption("In-processing (Zhang et al. 2018): the logistic model trains against an adversary that tries to recover whether the applicant is "
               "Black or Hispanic from its score; race is used only in training, never at decision time.")
    image("adv_debias.png")
    ad = load("adv_debias.csv")
    if ad is not None:
        st.dataframe(ad.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Human + AI
with tabs[11]:
    st.caption("Calibration by group, and a human-in-the-loop policy: the model abstains on the applications closest to its threshold.")
    image("human_ai_reliability.png"); image("human_ai_routing.png")
    for f in ["human_ai_calibration.csv", "human_ai_routing.csv"]:
        d = load(f)
        if d is not None:
            st.dataframe(d.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Recourse
with tabs[12]:
    st.caption("Cheapest actionable change (income up, loan amount down, DTI bin down) that turns each denial into an approval, by group.")
    image("recourse_cost.png")
    rc = load("recourse_by_group.csv")
    if rc is not None:
        st.dataframe(rc.round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Hardening
with tabs[13]:
    st.caption("LightGBM's decisions flipped under small input errors because it keyed on exact values (360-month term, 80% CLTV). "
               "Monotone constraints, regularisation and noise augmentation fix most of it.")
    image("hardening.png")
    hd = load("hardening.csv")
    if hd is not None:
        st.dataframe(hd.round(4), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Risk register
with tabs[14]:
    st.caption("Initial-report register (R1-R8) plus risks found in testing (R9-R11), with measured evidence, MITRE ATLAS / NIST AI RMF mapping "
               "and proposed residual scores (1-5 Low, 6-10 Medium, 11-15 High, 16-25 Critical).")
    rr = load("risk_register_measured.csv")
    if rr is not None:
        st.dataframe(rr, hide_index=True, width="stretch")

# ---------------------------------------------------------------- Model card
with tabs[15]:
    mc = ROOT / "docs" / "model_card.md"
    st.markdown(mc.read_text() if mc.exists() else "docs/model_card.md not found - run `python src/make_docs.py`.")
