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


st.set_page_config(page_title="BootstrapMD", page_icon="B", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700;800&display=swap');
    /* A quiet, light utility strip keeps the sidebar reopen control usable. */
    [data-testid="stHeader"] {position:absolute !important;background:transparent !important;height:0 !important;border:0 !important;z-index:1000000 !important;}
    [data-testid="stToolbar"] {background:transparent !important;padding:0 !important;}
    [data-testid="stDecoration"], #MainMenu, footer, [data-testid="stMainMenuButton"], [data-testid="stBaseButton-header"] {display:none !important;}
    [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {display:flex !important;visibility:visible !important;position:fixed !important;left:14px !important;top:14px !important;align-items:center !important;justify-content:center !important;width:42px !important;height:42px !important;min-width:42px !important;min-height:42px !important;background:#111827 !important;border:1px solid rgba(255,255,255,.24) !important;border-radius:10px !important;box-shadow:0 6px 16px rgba(12,35,55,.24) !important;}
    [data-testid="stSidebarCollapsedControl"] button, [data-testid="stExpandSidebarButton"] button {display:flex !important;align-items:center !important;justify-content:center !important;width:100% !important;height:100% !important;background:transparent !important;border:0 !important;color:#fff !important;}
    [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stExpandSidebarButton"] svg {width:22px !important;height:22px !important;color:#fff !important;stroke:#fff !important;stroke-width:2.7px !important;fill:none !important;opacity:1 !important;}
    [data-testid="stSidebarCollapsedControl"] svg path, [data-testid="stExpandSidebarButton"] svg path {stroke:#fff !important;fill:none !important;}
    /* Render our own menu glyph so the control stays legible across Streamlit versions. */
    [data-testid="stSidebarCollapsedControl"]::after, [data-testid="stExpandSidebarButton"]::after {content:'☰';position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#fff;font-family:Arial,sans-serif;font-size:24px;font-weight:700;line-height:1;pointer-events:none;z-index:2;text-shadow:0 1px 2px rgba(0,0,0,.28);}
    [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stExpandSidebarButton"] svg {visibility:hidden !important;}
    html, body, [class*="css"] {font-family:Inter, sans-serif; color:#263248;}
    .stApp {background:#eef2f9;}
    .block-container {max-width:1320px; padding:4.4rem 1.2rem 1.4rem;}
    div[data-testid="stHorizontalBlock"]:has(.step-label) {background:#fff;border:1px solid #e4e9f2;border-radius:28px;min-height:655px;overflow:hidden;gap:0 !important;box-shadow:0 18px 48px rgba(50,67,105,.12);}
    .brand,.tagline {display:none;}
    div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:first-child {background:#fbfbff;padding:2.05rem 2rem;min-height:655px;}
    div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:last-child {padding:2.2rem 2.3rem 1.5rem;min-height:655px;border-left:1px solid #eef1f6;background:#fff;}.workspace-title {font-size:1.05rem;font-weight:800;color:#273249;margin:.1rem 0 .12rem;}.workspace-title:before {content:'▥';display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;border-radius:8px;background:#f1f5fb;color:#60728e;font-size:1rem;margin-right:.65rem;vertical-align:middle;}.workspace-copy{font-size:.79rem;color:#99a7ba;margin:.2rem 0 1.3rem 2.55rem;}
    .step-label {display:flex;align-items:center;gap:.7rem;margin:.05rem 0 1.25rem;color:#8ea0bb;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;}.step-number {display:inline-flex;align-items:center;justify-content:center;width:33px;height:33px;border-radius:50%;background:#2f6df0;color:white;font-family:'DM Mono',monospace;letter-spacing:0;}.step-muted {margin-top:2.2rem;}.step-muted .step-number{background:#edf3ff;color:#7f9ce4;}
    [data-testid="stFileUploader"] {border:2px dashed #dce5f3;border-radius:22px;background:#fff;min-height:208px;display:flex;align-items:center;justify-content:center;padding:.85rem;transition:.2s ease;}
    [data-testid="stFileUploader"]:hover {border-color:#7fa6ff;background:#fbfdff;}
    [data-testid="stFileUploader"] section {min-height:184px;background:transparent;border:0;position:relative;}
    [data-testid="stFileUploader"] section > span {position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:max-content;}
    [data-testid="stFileUploader"] section > div {display:none;}
    [data-testid="stFileUploader"] button {background:#f7f9fe;color:#2f6df0;border:1px solid #e6ecf8;border-radius:11px;font-weight:700;font-size:1rem;padding:.66rem 1.45rem;box-shadow:0 5px 14px rgba(47,109,240,.08);}
    .upload-note {font-size:.76rem;color:#92a0b4;text-align:center;line-height:1.6;margin:.85rem 0 1.5rem;}.upload-note strong{display:block;color:#27344b;font-size:.9rem;margin-bottom:.15rem;}
    .dataset-ready {background:#ecf8f2;border:1px solid #c8ebda;border-radius:12px;padding:.7rem .8rem;color:#277557;font-size:.78rem;margin:.75rem 0 1rem;}
    .config-title {font-size:.68rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:#a0aec3;margin:.35rem 0 .45rem;}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {border-radius:12px;border-color:#e5eaf4;background:#fff;min-height:46px;}
    /* Data-editor dropdowns render in a portal, outside the workspace card. Give every popup a clear light surface. */
    div[data-baseweb="popover"], div[data-baseweb="popover"] [role="listbox"] {background:#fff !important;color:#27344b !important;border-color:#dfe7f3 !important;}
    div[data-baseweb="popover"] [role="option"] {background:#fff !important;color:#27344b !important;font-weight:600 !important;}
    div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] [role="option"][aria-selected="true"] {background:#eaf1ff !important;color:#1f5fd2 !important;}
    div.stButton > button {border:0;border-radius:12px;background:#7ca4f7;color:white;font-weight:800;min-height:52px;width:100%;box-shadow:0 8px 18px rgba(79,124,224,.18);}.stButton>button:hover {background:#4e81ec;color:white;}
    [data-testid="stSelectbox"] button,[data-testid="stSelectbox"] [role="combobox"] {background:#fff !important;color:#52637c !important;border-color:#e5eaf4 !important;}
    [data-testid="stSelectbox"] button:disabled,[data-testid="stSelectbox"] [role="combobox"][aria-disabled="true"] {background:#fbfcff !important;color:#a4b0c2 !important;}
    div.stButton > button:disabled {background:#eef3ff !important;color:#aab9d1 !important;box-shadow:none !important;opacity:1 !important;}
    .empty-workspace {height:450px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;border-radius:18px;background:#fff;color:#a2afc2;}.empty-icon {font-size:3.2rem;line-height:1;color:#dfe5ef;margin-bottom:.7rem;}.empty-workspace strong{color:#8a98ad;font-size:1rem;}.empty-workspace p{font-size:.78rem;margin:.35rem 0 0;}
    .privacy {font-size:.72rem;line-height:1.45;color:#84603a;border-left:3px solid #f1bf70;padding:.55rem .65rem;background:#fffaf1;border-radius:4px;margin:.7rem 0;}
    .result-card {background:#f9fbff;border:1px solid #e8edf6;border-radius:16px;padding:1rem;margin-top:.7rem;}
    div[data-testid="stMetric"] {background:#fff;border:1px solid #edf0f6;border-radius:12px;padding:.7rem;} div[data-testid="stMetricLabel"] {font-size:.69rem;color:#8f9db3;} div[data-testid="stMetricValue"] {font-size:1rem;color:#27344b;}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#102f4b 0%,#0b2238 100%);border-right:1px solid rgba(255,255,255,.08);}
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding-top:1.1rem;} [data-testid="stSidebar"] * {color:#e9f2fb;}
    [data-testid="stSidebarCollapseButton"] {visibility:visible !important;display:block !important;}
    [data-testid="stSidebarCollapseButton"] button {background:rgba(255,255,255,.08) !important;border-radius:8px !important;}
    .sidebar-brand {font-size:1.28rem;font-weight:800;letter-spacing:-.06em;margin-bottom:.15rem;}.sidebar-brand span{color:#73a7ff}.sidebar-caption {font-size:.73rem;line-height:1.5;color:#adc0d2 !important;margin-bottom:1.5rem;}
    .side-nav {font-size:.73rem;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#87a5c0 !important;margin:1.25rem 0 .55rem;}.side-step {padding:.62rem .72rem;border-radius:10px;background:rgba(255,255,255,.065);font-size:.79rem;margin:.38rem 0;color:#e7f0fb !important;}.side-step span{color:#79aaff !important;font-family:'DM Mono',monospace;margin-right:.5rem;}
    .tour-panel {font-family:'Inter',sans-serif;}.tour-kicker{font-size:.68rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#83a9ff;margin-bottom:.35rem;}.tour-heading{font-size:1.12rem;font-weight:800;color:#fff;margin:0 0 .35rem;}.tour-description{font-size:.86rem;line-height:1.5;color:#d3def0;margin:0;}.tour-help{font-size:.76rem;color:#9fb2cf;margin-top:.65rem;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) {position:fixed !important;right:26px;bottom:24px;z-index:1000002;width:min(390px,calc(100vw - 40px)) !important;background:#10253b;border:1px solid rgba(150,182,255,.25);border-radius:18px;box-shadow:0 20px 56px rgba(8,24,43,.34);padding:1rem 1rem .85rem;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stLayoutWrapper"] {position:static !important;width:auto;background:transparent;border:0;box-shadow:none;padding:0;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stVerticalBlock"], div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stHorizontalBlock"] {width:100% !important;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex:1 1 0 !important;width:auto !important;min-width:0 !important;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stButton"] button {width:100% !important;min-height:42px;border-radius:10px;font-weight:780;white-space:nowrap;padding:.5rem .55rem;font-size:.82rem;transition:transform .16s ease,filter .16s ease;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stColumn"]:first-child button {background:transparent !important;color:#d8e4f8 !important;border:1px solid #45617f !important;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stColumn"]:nth-child(2) button {background:#203b59 !important;color:#dce8fb !important;border:1px solid #365779 !important;}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stColumn"]:last-child button {background:linear-gradient(135deg,#4b8dff,#6ea4ff) !important;color:#fff !important;border:1px solid #79adff !important;box-shadow:0 7px 16px rgba(69,130,255,.28);}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stButton"] button:not(:disabled):hover {transform:translateY(-1px);filter:brightness(1.08);}
    div[data-testid="stLayoutWrapper"]:has(.tour-panel) [data-testid="stButton"] button:disabled {opacity:.42 !important;background:#1a2c42 !important;color:#9aabc0 !important;border-color:#314b66 !important;}
    .tour-step-0 ~ [data-testid="stFileUploader"], body:has(.tour-step-0) [data-testid="stFileUploader"] {position:relative;z-index:1000001;outline:3px solid #4e81ec;outline-offset:6px;box-shadow:0 0 0 9999px rgba(12,27,46,.52);}
    body:has(.tour-step-0) [data-testid="stFileUploader"]::before {content:'Start here  ↓';position:absolute;left:50%;top:-49px;transform:translateX(-50%);white-space:nowrap;background:#2f6df0;color:#fff;padding:8px 12px;border-radius:999px;font-size:.78rem;font-weight:800;letter-spacing:.01em;box-shadow:0 8px 18px rgba(47,109,240,.28);}
    body:has(.tour-step-1) [data-testid="stDataEditor"] {position:relative;z-index:1000001;outline:3px solid #4e81ec;outline-offset:6px;box-shadow:0 0 0 9999px rgba(12,27,46,.52);}
    body:has(.tour-step-1) [data-testid="stDataEditor"]::before {content:'Use each Type dropdown to correct an inference  ↓';position:absolute;left:50%;top:-49px;transform:translateX(-50%);white-space:nowrap;background:#2f6df0;color:#fff;padding:8px 12px;border-radius:999px;font-size:.78rem;font-weight:800;box-shadow:0 8px 18px rgba(47,109,240,.28);}
    body:has(.tour-step-2) div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:first-child {position:relative;z-index:1000001;outline:3px solid #4e81ec;outline-offset:-3px;box-shadow:0 0 0 9999px rgba(12,27,46,.52);}
    body:has(.tour-step-2) div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:first-child::before {content:'Your setup panel  ←';position:absolute;right:-20px;top:22px;transform:translateX(100%);white-space:nowrap;background:#2f6df0;color:#fff;padding:8px 12px;border-radius:999px;font-size:.78rem;font-weight:800;box-shadow:0 8px 18px rgba(47,109,240,.28);}
    body:has(.tour-step-3) div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:last-child {position:relative;z-index:1000001;outline:3px solid #4e81ec;outline-offset:-3px;box-shadow:0 0 0 9999px rgba(12,27,46,.52);}
    body:has(.tour-step-3) div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:last-child::before {content:'Results appear here  ↓';position:absolute;left:50%;top:18px;transform:translateX(-50%);white-space:nowrap;background:#2f6df0;color:#fff;padding:8px 12px;border-radius:999px;font-size:.78rem;font-weight:800;box-shadow:0 8px 18px rgba(47,109,240,.28);}
    body:has(.tour-step-4) [data-testid="stSidebar"] {position:relative;z-index:1000001;outline:3px solid #4e81ec;outline-offset:-3px;box-shadow:0 0 0 9999px rgba(12,27,46,.52);}
    body:has(.tour-step-4) [data-testid="stSidebar"]::after {content:'Replay this guide anytime  ↓';position:absolute;left:20px;bottom:84px;white-space:nowrap;background:#2f6df0;color:#fff;padding:8px 12px;border-radius:999px;font-size:.78rem;font-weight:800;box-shadow:0 8px 18px rgba(47,109,240,.28);}
    @media (max-width:800px) {div[data-testid="stLayoutWrapper"]:has(.tour-panel){right:16px;bottom:16px;}body:has(.tour-step-1) [data-testid="stDataEditor"]::before,body:has(.tour-step-2) div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:first-child::before{display:none;}}
    @media(max-width:800px){div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:first-child,div[data-testid="stHorizontalBlock"]:has(.step-label) > [data-testid="stColumn"]:last-child{min-height:0;padding:1.35rem;border-left:0;border-top:1px solid #eef1f6}div[data-testid="stHorizontalBlock"]:has(.step-label){min-height:0}.block-container{padding:.45rem}.empty-workspace{height:220px}}
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
        if st.session_state.get("tour_open") and st.session_state.get("tour_step") == 0:
            st.session_state.tour_step = 1
            st.rerun()
    except Exception as exc:
        return f"We could not read that CSV: {exc}"
    return None


def app_data() -> pd.DataFrame | None:
    return st.session_state.get("source_data")


def advance_tour_after_schema_edit() -> None:
    """Move the walkthrough forward when its highlighted schema task is completed."""
    if st.session_state.get("tour_open") and st.session_state.get("tour_step") == 1:
        st.session_state.tour_step = 2


TOUR_STEPS = [
    {"title": "Bring in a source dataset", "copy": "Drop in a CSV here. We will infer its structure before you generate anything.", "tip": "The highlighted upload card is the first stop."},
    {"title": "Confirm or correct column types", "copy": "After upload, use the Type dropdown in every row to correct any inferred type before generating synthetic data.", "tip": "Use continuous for measurements, integer for counts, categorical for labels; uncheck Include to exclude a column."},
    {"title": "Choose your synthesis setup", "copy": "This panel sets the population size, goal, and generation methods. It unlocks after your CSV is loaded.", "tip": "Start with the suggested settings for a fast first run."},
    {"title": "Work in the analysis space", "copy": "Your data preview, checks, scorecard, and downloads all appear here after you confirm the schema.", "tip": "This area updates as you move through the workflow."},
    {"title": "Keep the guide close", "copy": "The sidebar keeps the workflow visible. You can reopen this guide whenever you need it.", "tip": "Use the Play quick tour button at the bottom of the sidebar."},
]

st.session_state.setdefault("tour_open", True)
st.session_state.setdefault("tour_step", 0)


def show_tour() -> None:
    step_index = st.session_state.tour_step
    step = TOUR_STEPS[step_index]
    with st.container(border=True):
        st.markdown(
            f'<div class="tour-panel tour-step-{step_index}"><div class="tour-kicker">Guided tour · {step_index + 1} of {len(TOUR_STEPS)}</div>'
            f'<p class="tour-heading">{step["title"]}</p><p class="tour-description">{step["copy"]}</p>'
            f'<div class="tour-help">{step["tip"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.progress((step_index + 1) / len(TOUR_STEPS))
        back, skip, next_step = st.columns([.8, 1, 1])
        if back.button("Back", key="tour_back", disabled=step_index == 0, use_container_width=True):
            st.session_state.tour_step -= 1
            st.rerun()
        if skip.button("Skip", key="tour_skip", use_container_width=True):
            st.session_state.tour_open = False
            st.rerun()
        label = "Finish" if step_index == len(TOUR_STEPS) - 1 else "Next"
        if next_step.button(label, key="tour_next", type="primary", use_container_width=True):
            st.session_state.tour_open = step_index != len(TOUR_STEPS) - 1
            st.session_state.tour_step = min(step_index + 1, len(TOUR_STEPS) - 1)
            st.rerun()


if st.session_state.tour_open:
    show_tour()


with st.sidebar:
    st.markdown('<div class="sidebar-brand">Bootstrap<span>MD</span></div><p class="sidebar-caption">A focused workspace for research-grade synthetic data.</p>', unsafe_allow_html=True)
    st.markdown('<p class="side-nav">Workflow</p>', unsafe_allow_html=True)
    st.markdown('<div class="side-step"><span>01</span>Import dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step"><span>02</span>Confirm schema</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step"><span>03</span>Generate data</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-step"><span>04</span>Review scorecard</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Fast mode keeps modeling and scorecards responsive on large CSV files.")
    st.caption("Privacy signals are empirical diagnostics, not a release guarantee.")
    if st.button("Play quick tour", key="replay_tour", use_container_width=True):
        st.session_state.tour_step = 0
        st.session_state.tour_open = True
        st.rerun()


left, right = st.columns([.94, 2.06], gap="small")

with left:
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
        population = st.selectbox("Target population size", [1_000, 2_000, 5_000, 10_000], format_func=lambda n: f"{n:,} synthetic patients")
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
with right:
    st.markdown('<p class="workspace-title">Analysis workspace</p><p class="workspace-copy">Review data readiness, compare methods, and download verified outputs.</p>', unsafe_allow_html=True)
    if data is None:
        st.markdown('<div class="empty-workspace"><div class="empty-icon">▥</div><strong>Workspace empty</strong><p>Upload a dataset to begin statistical synthesis.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown("#### Confirm or correct column types")
        st.caption("We infer each column automatically. Use the **Type** dropdown in any row to correct it before you generate. Uncheck **Include** to leave a column out of synthesis.")
        schema = st.data_editor(
            st.session_state.schema,
            key="schema_editor",
            on_change=advance_tour_after_schema_edit,
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
