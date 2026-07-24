from __future__ import annotations

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


st.set_page_config(page_title="BootstrapMD", page_icon="B", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}
    html, body, [class*="css"] {font-family:Inter, sans-serif; color:#263248;}
    .stApp {background:linear-gradient(135deg,#eff3fb 0%,#f7f8fc 56%,#edf2fb 100%);}
    .block-container {max-width:1280px; padding:1.1rem 1.15rem 1.35rem;}
    .app-shell {background:rgba(255,255,255,.93);border:1px solid rgba(226,231,242,.9);border-radius:28px;padding:1.25rem 1.35rem 1.45rem;box-shadow:0 20px 54px rgba(55,70,110,.12);}
    .brand {font-size:1.25rem;font-weight:800;letter-spacing:-.05em;color:#1f2c43;margin:0;}.brand span{color:#3471eb}.tagline {font-size:.76rem;color:#8a99b0;margin:.18rem 0 1rem;}
    .left-panel {background:#fbfbff;border-radius:20px;padding:1.15rem 1.05rem;min-height:620px;border:1px solid #f0f1f7;}
    .workspace {padding:.55rem .55rem 0 1.35rem;}.workspace-title {font-size:1.08rem;font-weight:800;color:#263248;margin:.1rem 0 .15rem;}.workspace-copy{font-size:.81rem;color:#91a0b6;margin:0 0 1rem;}
    .step-label {display:flex;align-items:center;gap:.6rem;margin:.3rem 0 1rem;color:#91a1b9;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}.step-number {display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:50%;background:#2f6df0;color:white;font-family:'DM Mono',monospace;letter-spacing:0;}.step-muted .step-number{background:#eef3ff;color:#7f9ce4;}
    [data-testid="stFileUploader"] {border:2px dashed #dce5f3;border-radius:21px;background:#fff;min-height:198px;display:flex;align-items:center;justify-content:center;padding:.85rem;transition:.2s ease;}
    [data-testid="stFileUploader"]:hover {border-color:#7fa6ff;background:#fbfdff;}
    [data-testid="stFileUploader"] section {background:transparent;border:0;}
    [data-testid="stFileUploader"] button {background:#f7f9fe;color:#2f6df0;border:1px solid #e6ecf8;border-radius:10px;font-weight:700;}
    .upload-note {font-size:.76rem;color:#92a0b4;text-align:center;line-height:1.6;margin:.85rem 0 1.3rem;}.upload-note strong{display:block;color:#27344b;font-size:.9rem;margin-bottom:.15rem;}
    .dataset-ready {background:#ecf8f2;border:1px solid #c8ebda;border-radius:12px;padding:.7rem .8rem;color:#277557;font-size:.78rem;margin:.75rem 0 1rem;}
    .config-title {font-size:.68rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#a0aec3;margin:.35rem 0 .45rem;}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {border-radius:12px;border-color:#e5eaf4;background:#fff;min-height:46px;}
    div.stButton > button {border:0;border-radius:12px;background:#7ca4f7;color:white;font-weight:800;min-height:52px;width:100%;box-shadow:0 8px 18px rgba(79,124,224,.18);}.stButton>button:hover {background:#4e81ec;color:white;}
    .empty-workspace {height:370px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border-radius:18px;background:linear-gradient(145deg,#fff 0%,#fbfcff 100%);border:1px solid #f0f2f7;color:#a2afc2;}.empty-icon {font-size:3.2rem;line-height:1;color:#dfe5ef;margin-bottom:.7rem;}.empty-workspace strong{color:#7e8da3;font-size:1rem;}.empty-workspace p{font-size:.78rem;margin:.35rem 0 0;}
    .privacy {font-size:.72rem;line-height:1.45;color:#84603a;border-left:3px solid #f1bf70;padding:.55rem .65rem;background:#fffaf1;border-radius:4px;margin:.7rem 0;}
    .result-card {background:#f9fbff;border:1px solid #e8edf6;border-radius:16px;padding:1rem;margin-top:.7rem;}
    div[data-testid="stMetric"] {background:#fff;border:1px solid #edf0f6;border-radius:12px;padding:.7rem;} div[data-testid="stMetricLabel"] {font-size:.69rem;color:#8f9db3;} div[data-testid="stMetricValue"] {font-size:1rem;color:#27344b;}
    @media(max-width:800px){.workspace{padding:1.1rem 0 0}.left-panel{min-height:0}.app-shell{padding:.85rem}.block-container{padding:.45rem}.empty-workspace{height:220px}}
    </style>""",
    unsafe_allow_html=True,
)


def load_csv(uploaded_file) -> pd.DataFrame:
    """Read an upload once, with resilient CSV parsing for real-world exports."""
    uploaded_file.seek(0)
    options = {"low_memory": False, "on_bad_lines": "warn"}
    try:
        return pd.read_csv(uploaded_file, **options)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1", **options)


def process_upload(uploaded_file) -> str | None:
    key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("upload_key") == key:
        return None
    try:
        with st.spinner("Reading and validating your CSV..."):
            data = load_csv(uploaded_file)
        if data.empty:
            return "This CSV has no records."
        if len(data.columns) < 1:
            return "This CSV has no usable columns."
        st.session_state.source_data = data
        st.session_state.schema = infer_schema(data)
        st.session_state.upload_key = key
        st.session_state.pop("outputs", None)
        st.session_state.pop("evaluations", None)
    except Exception as exc:
        return f"We could not read that CSV: {exc}"
    return None


def app_data() -> pd.DataFrame | None:
    return st.session_state.get("source_data")


st.markdown('<div class="app-shell"><p class="brand">Bootstrap<span>MD</span></p><p class="tagline">Clinical research data synthesis workspace</p>', unsafe_allow_html=True)
left, right = st.columns([.94, 2.06], gap="large")

with left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div class="step-label"><span class="step-number">1</span>Source dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Import CSV data",
        type=["csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        help="CSV files up to 500 MB are supported.",
    )
    st.markdown('<p class="upload-note"><strong>Import CSV data</strong>Drag and drop, or click above to browse.<br>CSV only - up to 500 MB</p>', unsafe_allow_html=True)
    if uploaded is not None:
        error = process_upload(uploaded)
        if error:
            st.error(error)
        elif app_data() is not None:
            st.markdown(f'<div class="dataset-ready"><strong>{uploaded.name}</strong><br>Ready: {len(app_data()):,} records x {len(app_data().columns)} columns</div>', unsafe_allow_html=True)

    st.markdown('<div class="step-label step-muted"><span class="step-number">2</span>Synthesis config</div>', unsafe_allow_html=True)
    data = app_data()
    if data is None:
        st.markdown('<p class="config-title">Target population size</p>')
        st.selectbox("Target population size", ["Upload a dataset first"], disabled=True, label_visibility="collapsed")
        st.button("Generate synthetic data", disabled=True, type="primary")
    else:
        included = st.session_state.schema.loc[st.session_state.schema["include"], "column"].tolist()
        population = st.selectbox("Target population size", [len(data), max(250, len(data) * 2), max(500, len(data) * 5), max(1000, len(data) * 10)], format_func=lambda n: f"{n:,} synthetic patients")
        target = st.selectbox("Utility outcome", ["None"] + included, help="Select a categorical outcome to calculate predictive utility.")
        methods = st.multiselect("Methods", ["Smoothed bootstrap", "SMOTE-NC", "Gaussian copula", "CART sequential"], default=["Smoothed bootstrap", "SMOTE-NC", "Gaussian copula", "CART sequential"])
        goal = st.selectbox("Primary goal", ["Balanced", "Maximize utility", "Maximize privacy"])
        st.markdown('<div class="privacy"><strong>Privacy limit:</strong> these are empirical risk signals, not a differential-privacy guarantee. Do not upload PHI to an unapproved environment.</div>', unsafe_allow_html=True)
        run = st.button("Generate synthetic data", type="primary", disabled=not methods)
        if run:
            types = schema_types(st.session_state.schema)
            active = data[list(types)]
            target_col = None if target == "None" else target
            registry = {
                "Smoothed bootstrap": smoothed_bootstrap,
                "SMOTE-NC": lambda d, t, n: smote_nc(d, t, n, target_col),
                "Gaussian copula": gaussian_copula,
                "CART sequential": cart_sequential,
            }
            outputs, evaluations = {}, []
            progress = st.progress(0, text="Preparing synthesis...")
            try:
                for index, method in enumerate(methods, 1):
                    progress.progress((index - 1) / len(methods), text=f"Running {method}...")
                    output = registry[method](active, types, population)
                    outputs[method] = output
                    evaluations.append(evaluate(active, output, types, target_col, method))
                st.session_state.outputs = outputs
                st.session_state.evaluations = evaluations
                st.session_state.goal = goal
            except Exception as exc:
                st.error(f"Synthesis did not finish: {exc}")
            finally:
                progress.empty()
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="workspace"><p class="workspace-title">Analysis workspace</p><p class="workspace-copy">Review data readiness, compare methods, and download verified outputs.</p>', unsafe_allow_html=True)
    if data is None:
        st.markdown('<div class="empty-workspace"><div class="empty-icon">&#128190;</div><strong>Workspace empty</strong><p>Upload a dataset to begin statistical synthesis.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown("#### Dataset readiness")
        schema = st.data_editor(
            st.session_state.schema,
            key="schema_editor",
            hide_index=True,
            width="stretch",
            column_config={
                "type": st.column_config.SelectboxColumn(options=["continuous", "integer", "categorical"]),
                "include": st.column_config.CheckboxColumn(),
            },
        )
        st.session_state.schema = schema
        types = schema_types(schema)
        issues = validate_data(data, types)
        if issues:
            st.caption("Review these validity checks before generating data.")
            st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
        else:
            st.success("Dataset checks complete. You can generate synthetic data.")
        st.markdown("#### Data preview")
        st.dataframe(data.head(8), hide_index=True, width="stretch")

    if "evaluations" in st.session_state:
        results = st.session_state.evaluations
        chosen = recommend(results, st.session_state.goal)
        rows = []
        for item in results:
            rows.append({"method": item.method, "fidelity": item.fidelity, "utility": "-" if item.utility is None else item.utility, "privacy": item.privacy, "grade": item.grade, **item.details})
        scorecard = pd.DataFrame(rows)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### Synthesis scorecard")
        one, two, three, four = st.columns(4)
        one.metric("Recommended", chosen.method)
        two.metric("Fidelity", f"{chosen.fidelity:.1f}/100")
        three.metric("Utility", "-" if chosen.utility is None else f"{chosen.utility:.1f}/100")
        four.metric("Privacy signal", f"{chosen.privacy:.1f}/100")
        st.dataframe(scorecard, hide_index=True, width="stretch")
        download_columns = st.columns(len(st.session_state.outputs) + 1)
        for index, (name, output) in enumerate(st.session_state.outputs.items()):
            download_columns[index].download_button(f"Download {name}", output.to_csv(index=False).encode(), file_name=f"bootstrapmd_{name.lower().replace(' ', '_').replace('-', '_')}.csv", mime="text/csv")
        report = f"BootstrapMD synthesis report\nGenerated: {datetime.now():%Y-%m-%d %H:%M}\nGoal: {st.session_state.goal}\n\n" + scorecard.to_csv(index=False)
        download_columns[-1].download_button("Download scorecard", report.encode(), file_name="bootstrapmd_scorecard.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
