from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from synth_engine import (
    cart_sequential,
    evaluate,
    gaussian_copula,
    infer_schema,
    recommend,
    schema_types,
    smote_nc,
    smoothed_bootstrap,
    validate_data,
)

st.set_page_config(page_title="BootstrapMD | Clinical Data Synthesis", page_icon="✦", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Manrope', sans-serif; color: #12304a;}
.stApp {background: #f5f8fb;} .block-container {max-width: 1200px; padding-top: 2.2rem; padding-bottom: 3rem;}
.hero {padding: 0 0 .55rem 0;} .hero h1 {font-size: 2.05rem; letter-spacing: -.06em; margin: 0; color: #0e2d47; font-weight: 800;}
.hero p {margin: .35rem 0 0; color: #61788d; font-size: .95rem;}
.eyebrow {color:#197a8a; font-family:'DM Mono',monospace; font-size:.70rem; letter-spacing:.11em; text-transform:uppercase; font-weight:500;}
.card {background:#fff; border:1px solid #dce6ee; border-radius:18px; padding:1.35rem; box-shadow:0 5px 20px rgba(23,57,79,.045); margin-bottom:1rem;}
.step {display:inline-flex; align-items:center; justify-content:center; width:25px; height:25px; border-radius:50%; background:#e4f4f2; color:#167c88; font:500 .72rem 'DM Mono',monospace; margin-right:.55rem;}
.section-title {font-size:1rem; font-weight:800; color:#143650; margin:0 0 .25rem;}.section-copy {font-size:.83rem;color:#6a8193;margin:0 0 1rem;}
div[data-testid="stMetric"] {background:#f8fbfc;border:1px solid #e2ebf1;border-radius:12px;padding:.65rem .8rem;} div[data-testid="stMetricLabel"] {font-size:.72rem;color:#6a8193;} div[data-testid="stMetricValue"] {font-size:1.25rem;color:#143650;}
.privacy {background:#fff7ec;border:1px solid #f2d7b4;border-radius:12px;padding:.8rem 1rem;color:#7d5423;font-size:.82rem;line-height:1.45;}
div.stButton > button {background:#126b7a;color:#fff;border:0;border-radius:9px;padding:.55rem 1.1rem;font-weight:700;} div.stButton > button:hover {background:#0b5664;color:#fff;}
[data-testid="stSidebar"] {background:#102e47;} [data-testid="stSidebar"] * {color:#e9f1f6 !important;}
</style>""", unsafe_allow_html=True)


def card_start(number: int, title: str, copy: str) -> None:
    st.markdown(f'<div class="card"><p class="section-title"><span class="step">{number}</span>{title}</p><p class="section-copy">{copy}</p>', unsafe_allow_html=True)


def card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def get_data() -> pd.DataFrame | None:
    if "source_data" not in st.session_state:
        return None
    return st.session_state.source_data


st.markdown('<div class="hero"><div class="eyebrow">Clinical research workspace</div><h1>BootstrapMD</h1><p>High-fidelity data synthesis with transparent validation for research teams.</p></div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### Synthesis workspace")
    st.caption("Methodologically clear. Privacy-aware. Designed for research—not patient care.")
    st.divider()
    st.markdown("**Workflow**")
    st.caption("01 Upload & inspect\n\n02 Confirm schema\n\n03 Generate & evaluate\n\n04 Export results")
    st.divider()
    st.caption("© 2026 BootstrapMD Research Systems")

left, right = st.columns([1.02, .98], gap="large")
with left:
    card_start(1, "Source dataset", "Import a CSV, then review inferred column types before any synthesis runs.")
    uploaded = st.file_uploader("Import CSV data", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            if data.empty:
                st.error("The uploaded CSV has no rows.")
            else:
                st.session_state.source_data = data
                if "schema" not in st.session_state or list(st.session_state.schema["column"]) != list(data.columns):
                    st.session_state.schema = infer_schema(data)
                st.success(f"Loaded {len(data):,} records × {len(data.columns)} columns")
                st.dataframe(data.head(8), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    elif get_data() is None:
        st.info("Drag a CSV here to begin.")
    card_end()

    if get_data() is not None:
        card_start(2, "Confirm data types", "The model uses these selections to preserve each column’s intended meaning.")
        edited_schema = st.data_editor(st.session_state.schema, key="schema_editor", hide_index=True, width="stretch", column_config={"type": st.column_config.SelectboxColumn(options=["continuous", "integer", "categorical"]), "include": st.column_config.CheckboxColumn()})
        st.session_state.schema = edited_schema
        types = schema_types(edited_schema)
        issues = validate_data(get_data(), types)
        if issues:
            st.caption("Validity and constraint checks")
            st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
        else:
            st.success("No data-quality concerns detected in the selected columns.")
        card_end()

with right:
    card_start(3, "Synthesis configuration", "Choose the target population, any useful outcome variable, and the statistical methods to compare.")
    data = get_data()
    if data is None:
        st.info("Your configuration options will appear after you upload a CSV.")
    else:
        included_cols = st.session_state.schema.loc[st.session_state.schema["include"], "column"].tolist()
        population = st.selectbox("Target population size", [len(data), max(250, len(data) * 2), max(500, len(data) * 5), max(1000, len(data) * 10)], format_func=lambda x: f"{x:,} synthetic records")
        target = st.selectbox("Utility outcome (optional)", ["None"] + included_cols, help="A categorical outcome enables TSTR/TRTR utility evaluation.")
        methods = st.multiselect("Synthesis methods", ["Smoothed bootstrap", "SMOTE-NC", "Gaussian copula", "CART sequential"], default=["Smoothed bootstrap", "SMOTE-NC", "Gaussian copula", "CART sequential"])
        goal = st.radio("Primary goal", ["Balanced", "Maximize utility", "Maximize privacy"], horizontal=True)
        st.markdown('<div class="privacy"><strong>Privacy boundary.</strong> These empirical risk signals are not a differential-privacy guarantee. Do not upload protected health information to an unapproved environment or treat synthetic output as automatically safe to release.</div>', unsafe_allow_html=True)
        run = st.button("Generate & evaluate", type="primary", disabled=not methods)
        if run:
            types = schema_types(st.session_state.schema)
            active_data = data[list(types)]
            target_col = None if target == "None" else target
            registry = {"Smoothed bootstrap": smoothed_bootstrap, "SMOTE-NC": lambda d, t, n: smote_nc(d, t, n, target_col), "Gaussian copula": gaussian_copula, "CART sequential": cart_sequential}
            outputs, evaluations = {}, []
            progress = st.progress(0, text="Preparing synthesis run…")
            try:
                for index, method in enumerate(methods, start=1):
                    progress.progress((index - 1) / len(methods), text=f"Running {method}…")
                    synthetic = registry[method](active_data, types, population)
                    outputs[method] = synthetic
                    evaluations.append(evaluate(active_data, synthetic, types, target_col, method))
                st.session_state.outputs = outputs
                st.session_state.evaluations = evaluations
                st.session_state.goal = goal
                st.success("Synthesis and evaluation complete.")
            except Exception as exc:
                st.exception(exc)
            finally:
                progress.empty()
    card_end()

if "evaluations" in st.session_state:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title"><span class="step">4</span>Analysis workspace</p><p class="section-copy">Compare effect-size-based fidelity, practical predictive utility, and empirical disclosure signals.</p>', unsafe_allow_html=True)
    rows = []
    for result in st.session_state.evaluations:
        row = {"method": result.method, "fidelity": result.fidelity, "utility": "—" if result.utility is None else result.utility, "privacy": result.privacy, "grade": result.grade, **result.details}
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    selected = recommend(st.session_state.evaluations, st.session_state.goal)
    a, b, c, d = st.columns(4)
    a.metric("Recommended method", selected.method)
    b.metric("Fidelity", f"{selected.fidelity:.1f}/100")
    c.metric("Utility", "—" if selected.utility is None else f"{selected.utility:.1f}/100")
    d.metric("Privacy signal", f"{selected.privacy:.1f}/100")
    st.dataframe(scorecard, hide_index=True, width="stretch")
    st.caption("Fidelity combines distribution, categorical-balance, correlation, and propensity metrics. Utility is TSTR relative to TRTR when a categorical outcome is selected. Lower pMSE, KS, TVD, correlation delta, duplication, and MIA signal are better.")
    download_cols = st.columns(len(st.session_state.outputs) + 1)
    for index, (method, output) in enumerate(st.session_state.outputs.items()):
        download_cols[index].download_button(f"Download {method}", output.to_csv(index=False).encode(), file_name=f"bootstrapmd_{method.lower().replace(' ', '_').replace('-', '_')}.csv", mime="text/csv")
    report = f"BootstrapMD synthesis report\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\nGoal: {st.session_state.goal}\n\n" + scorecard.to_csv(index=False)
    download_cols[-1].download_button("Download scorecard", report.encode(), file_name="bootstrapmd_scorecard.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)
