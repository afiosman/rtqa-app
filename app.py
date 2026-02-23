# app.py — Radiotherapy Suite
# streamlit run app.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import hashlib
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

from core.io_log import extract_logs_texts
from core.preprocess import preprocess_and_merge
from core.report import generate_pdf_qa_report_bytes

from core.analysis import (
    add_errors,
    error_describe,
    leaf_metrics,
    classify,
    compute_gap_df_from_merged,
    plot_virtual_picket_fence,
    make_combined_leaf_index,
    max_by_leaf,
    plot_max_errors_by_leaf,
    summarize_overall_max_error,
    append_trending_csv,
    plot_overall_max_trending,
)

# =============================================================================
# App identity
# =============================================================================
APP_VERSION = "1.0.0"
APP_NAME = "Radiotherapy QA Platform"
APP_SHORT_NAME = "RT QA"

st.set_page_config(
    page_title=APP_SHORT_NAME,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Theme + CSS
# =============================================================================
THEME = {
    "primary": "#1f2a44",
    "accent": "#C99700",
    "bg": "#f5f7fa",
    "panel": "#ffffff",
    "border": "#e5e7eb",
    "text": "#111827",
    "muted": "#4b5563",
    "success": "#0f766e",
    "warn": "#b45309",
    "danger": "#b91c1c",
}


def inject_css(t: dict) -> None:
    st.markdown(
        f"""
<style>
:root {{
  --primary: {t["primary"]};
  --accent: {t["accent"]};
  --bg: {t["bg"]};
  --panel: {t["panel"]};
  --border: {t["border"]};
  --text: {t["text"]};
  --muted: {t["muted"]};
  --success: {t["success"]};
  --warn: {t["warn"]};
  --danger: {t["danger"]};
  --radius: 16px;
  --shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  --icon: var(--accent);
}}

.stApp {{ background: var(--bg); }}
footer {{ visibility: hidden; }}

.block-container {{
  padding-top: 1.0rem !important;
  padding-bottom: 2.0rem !important;
  max-width: 1280px;
}}

header[data-testid="stHeader"] {{
  visibility: visible !important;
  height: 3.25rem !important;
  background: transparent !important;
  border-bottom: none !important;
}}
header[data-testid="stHeader"] > div {{ background: transparent !important; }}

button[data-testid="collapsedControl"] {{
  visibility: visible !important;
  opacity: 1 !important;
  display: flex !important;
  pointer-events: auto !important;
  z-index: 9999 !important;
}}

/* Accent icons */
button svg, button svg *,
[data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
[data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
[data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
[data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
  fill: var(--icon) !important;
  stroke: var(--icon) !important;
}}
[data-testid="stSidebar"] [data-testid*="icon"],
[data-testid="stToolbar"] [data-testid*="icon"],
[data-testid="stHeader"] [data-testid*="icon"],
[data-testid="stExpander"] [data-testid*="icon"] {{
  color: var(--icon) !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
  border-right: 1px solid rgba(255,255,255,0.06);
  z-index: 9998 !important;
}}
section[data-testid="stSidebar"] * {{
  color: rgba(255,255,255,0.92) !important;
}}
section[data-testid="stSidebar"] .stTextInput input {{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  color: rgba(255,255,255,0.92) !important;
}}
section[data-testid="stSidebar"] .stTextInput input::placeholder {{
  color: rgba(255,255,255,0.70) !important;
}}

/* Sidebar selectbox (closed control) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
  box-shadow: none !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] span,
section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
  color: rgba(255,255,255,0.92) !important;
  font-weight: 750 !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder {{
  color: rgba(255,255,255,0.72) !important;
}}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
section[data-testid="stSidebar"] div[data-baseweb="select"] svg * {{
  stroke: var(--accent) !important;
  fill: var(--accent) !important;
}}

/* Dropdown popover */
div[data-baseweb="popover"] {{
  z-index: 100000 !important;
}}
div[data-baseweb="popover"] div[data-baseweb="menu"] {{
  background: #ffffff !important;
  border: 1px solid rgba(17,24,39,0.12) !important;
  border-radius: 14px !important;
  box-shadow: 0 14px 30px rgba(17, 24, 39, 0.18) !important;
  padding: 6px !important;
}}
div[data-baseweb="popover"] div[data-baseweb="menu"] li {{
  background: #ffffff !important;
  color: #111827 !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
  margin: 4px 2px !important;
}}
div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover {{
  background: rgba(31,42,68,0.06) !important;
}}
div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"] {{
  background: rgba(17,24,39,0.04) !important;
  border: 1px solid rgba(17,24,39,0.08) !important;
}}
div[data-baseweb="popover"] * {{
  color: #111827 !important;
}}

/* Sidebar cards (pure HTML blocks) */
.sidebar-card {{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 12px 12px;
  margin: 10px 0 10px 0;
}}
.sidebar-card h4 {{
  margin: 0 0 8px 0;
  font-size: 0.95rem;
  font-weight: 850;
  color: rgba(255,255,255,0.96);
}}
.sidebar-muted {{
  color: rgba(255,255,255,0.78) !important;
  font-size: 0.86rem;
  line-height: 1.35;
}}

/* Sidebar expander */
section[data-testid="stSidebar"] details {{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
  overflow: hidden !important;
}}
section[data-testid="stSidebar"] details > summary {{
  background: rgba(255,255,255,0.06) !important;
  padding: 12px 12px !important;
  border-radius: 14px !important;
}}
section[data-testid="stSidebar"] details[open] > summary {{
  background: rgba(255,255,255,0.06) !important;
  border-bottom: 1px solid rgba(255,255,255,0.10) !important;
}}
section[data-testid="stSidebar"] details > summary * {{
  color: rgba(255,255,255,0.92) !important;
}}

/* Inline code in sidebar */
section[data-testid="stSidebar"] code {{
  background: rgba(255,255,255,0.10) !important;
  color: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 8px !important;
  padding: 0.05rem 0.35rem !important;
}}

/* Topbar */
.topbar {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 16px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}}
.brand {{ display:flex; flex-direction:column; gap:2px; }}
.brand-title {{
  font-size: 1.05rem;
  font-weight: 900;
  color: var(--text);
}}
.brand-sub {{
  font-size: 0.90rem;
  color: var(--muted);
  line-height: 1.45;
}}
.topbar-right {{
  display:flex;
  align-items:center;
  gap:10px;
  flex-wrap:wrap;
  justify-content:flex-end;
}}
.badge {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border-radius:999px;
  padding:6px 10px;
  border:1px solid var(--border);
  background: rgba(31,42,68,0.03);
  font-size:0.85rem;
  color: var(--text);
}}
.badge-dot {{
  width:9px; height:9px; border-radius:50%;
  background: var(--muted);
}}
.badge.success .badge-dot {{ background: var(--success); }}
.badge.warn .badge-dot {{ background: var(--warn); }}
.badge.danger .badge-dot {{ background: var(--danger); }}

.kbd {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
  font-size:0.82rem;
  padding:2px 6px;
  border:1px solid var(--border);
  border-radius:8px;
  background: rgba(17,24,39,0.03);
  color: var(--text);
}}

.section-title {{
  margin: 18px 0 6px 0;
  font-weight: 900;
  font-size: 1.08rem;
  color: var(--text);
}}
.section-sub {{
  margin: 0 0 12px 0;
  color: var(--muted);
  line-height: 1.45;
}}

.stButton button {{
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 700 !important;
}}
.primary-btn button {{
  background: var(--primary) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}}
.primary-btn button:hover {{ filter: brightness(1.06); }}
.ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

/* Status banner */
.status-banner {{
  padding: 0.85rem 1rem;
  border-radius: 0.85rem;
  border: 1px solid var(--border);
  margin: 0.5rem 0 0.75rem 0;
  background: var(--panel);
  box-shadow: var(--shadow);
}}
.status-title {{
  font-weight: 900;
  font-size: 1.02rem;
  margin-bottom: 0.15rem;
}}
.status-sub {{ color: var(--muted); line-height: 1.4; }}
.status-chip {{
  display:inline-flex;
  align-items:center;
  gap:8px;
  border-radius:999px;
  padding:5px 10px;
  border:1px solid var(--border);
  font-size:0.82rem;
  margin-left:8px;
}}
.status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
.status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
.status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

/* Tabs */
div[data-testid="stTabs"] {{ width: 100%; }}
div[data-baseweb="tab-list"] {{
  padding: 0.30rem !important;
  background: rgba(17,24,39,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  overflow: hidden !important;
}}
button[data-baseweb="tab"] {{
  flex: 1 !important;
  justify-content: center !important;
  border-radius: 16px !important;
  padding: 0.62rem 1.10rem !important;
  font-weight: 900 !important;
  font-size: 0.96rem !important;
  color: rgba(31,42,68,0.78) !important;
  background: transparent !important;
  border-bottom: none !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  background: rgba(201,151,0,0.12) !important;
  color: var(--primary) !important;
  border: 1px solid rgba(201,151,0,0.25) !important;
}}

/* ============================================================
   KEY FIX: style st.container(border=True) to match your cards
   ============================================================ */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
  padding: 16px !important;
  position: relative !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]::before {{
  content: "";
  position: absolute;
  left: 0;
  top: 14px;
  bottom: 14px;
  width: 3px;
  border-radius: 999px;
  background: rgba(201,151,0,0.55);
}}
</style>
""",
        unsafe_allow_html=True,
    )


inject_css(THEME)

# =============================================================================
# Defaults / constants
# =============================================================================
PREVIEW_ROWS = 100
PLAN_FOLDER = Path("data")
OUTPUTS_DIR = Path("outputs")
TREND_DIR = Path("data") / "trending"
TREND_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# State management
# =============================================================================
def ensure_state() -> None:
    defaults = {
        "system_status": "ready",
        "upload_complete": False,
        "analysis_ready": False,
        "drop_stack_by_plan_name": True,
        "last_upload_signature": None,
        "last_parsed_sig": None,
        "qa_mode": "PF Log-file analysis",
        "site_name": "",
        "machine_name": "MRIdian",
        "reviewer_name": "",
        # tolerances (dynamic)
        "tol_profile": "MPPG (0.5/1.0)",
        "tolerance_mm": 0.5,
        "action_mm": 1.0,
        # last run summary
        "last_run": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


ensure_state()

# =============================================================================
# Helpers
# =============================================================================
def _plans_available(plan_folder: Path) -> bool:
    return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


@st.cache_data(show_spinner=False)
def load_plan_data(folder: Path):
    dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
    dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
    dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
    return dfP_all, dfP_upper, dfP_lower


def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
    sig = []
    for f in files:
        b = f.getvalue()
        sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
    return tuple(sorted(sig))


@st.cache_data(show_spinner=False)
def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
    return extract_logs_texts(list(texts_and_names))


def parse_uploaded(files):
    sig = _uploaded_signature(files)
    if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
        return (
            st.session_state["df_all"],
            st.session_state["df_upper"],
            st.session_state["df_lower"],
            st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
        )

    texts_and_names = []
    for f in files:
        raw = f.getvalue()
        text = raw.decode("utf-8", errors="ignore")
        texts_and_names.append((text, f.name))

    out = _parse_uploaded_texts_cached(tuple(texts_and_names))

    if isinstance(out, tuple) and len(out) == 4:
        df_all, df_upper, df_lower, df_errors = out
    else:
        df_all, df_upper, df_lower = out
        df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

    st.session_state["last_parsed_sig"] = sig
    st.session_state["df_all"] = df_all
    st.session_state["df_upper"] = df_upper
    st.session_state["df_lower"] = df_lower
    st.session_state["df_errors"] = df_errors
    return df_all, df_upper, df_lower, df_errors


def _reset_results_on_new_upload(uploaded_files) -> None:
    if not uploaded_files:
        return

    current_sig = _uploaded_signature(uploaded_files)
    last_sig = st.session_state.get("last_upload_signature")

    if current_sig != last_sig:
        for k in (
            "df_all",
            "df_upper",
            "df_lower",
            "df_errors",
            "dfP_all",
            "dfP_upper",
            "dfP_lower",
            "mU",
            "mL",
            "analysis_out",
            "pdf_bytes",
            "pdf_name",
            "trend_all",
            "last_parsed_sig",
            "upload_complete",
            "analysis_ready",
            "last_run",
        ):
            st.session_state.pop(k, None)

        _parse_uploaded_texts_cached.clear()
        st.session_state["last_upload_signature"] = current_sig


def _safe_first(df: pd.DataFrame, col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        return df[col].iloc[0]
    except Exception:
        return None


def _ensure_merged() -> None:
    if "mU" in st.session_state and "mL" in st.session_state:
        return

    if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
        raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

    if not _plans_available(PLAN_FOLDER):
        raise RuntimeError(
            "Plan PKL files not found in ./data. "
            "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
        )

    if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
        dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
        st.session_state["dfP_all"] = dfP_all
        st.session_state["dfP_upper"] = dfP_upper
        st.session_state["dfP_lower"] = dfP_lower

    out = preprocess_and_merge(
        dfP_upper=st.session_state["dfP_upper"],
        dfP_lower=st.session_state["dfP_lower"],
        dfL_upper=st.session_state["df_upper"],
        dfL_lower=st.session_state["df_lower"],
        drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
    )
    st.session_state["mU"] = out["mU"]
    st.session_state["mL"] = out["mL"]


def _upload_is_complete() -> bool:
    df_u = st.session_state.get("df_upper", None)
    df_l = st.session_state.get("df_lower", None)
    n_u = 0 if df_u is None else len(df_u)
    n_l = 0 if df_l is None else len(df_l)
    return (n_u > 0) and (n_l > 0)


def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
    s = (status or "").strip().upper()
    if s == "PASS":
        chip = '<span class="status-chip pass">PASS</span>'
        title = f"{scope_name}: Within tolerance"
    elif s in ("WARN", "WARNING"):
        chip = '<span class="status-chip warn">WARN</span>'
        title = f"{scope_name}: Warning (between tolerance and action)"
    else:
        chip = '<span class="status-chip fail">FAIL</span>'
        title = f"{scope_name}: Action level exceeded"

    st.markdown(
        f"""
<div class="status-banner">
  <div class="status-title">{title}{chip}</div>
  <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _status_dot_class(system_status: str) -> str:
    s = (system_status or "").lower()
    if s == "ready":
        return "success"
    if s in ("parsing", "missing_plan"):
        return "warn"
    return "danger"


def combine_status(su: str, sl: str) -> str:
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    su = (su or "FAIL").upper().strip()
    sl = (sl or "FAIL").upper().strip()
    return max([su, sl], key=lambda x: order.get(x, 2))


def render_topbar(qa_mode: str, tol_mm: float, act_mm: float) -> None:
    badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
    badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

    last_run = st.session_state.get("last_run")
    if isinstance(last_run, dict) and last_run.get("status"):
        lr = f"Last QA: {last_run['status']} • {last_run.get('timestamp','')}"
    else:
        lr = "Last QA: —"

    tol_tag = f"Tol {tol_mm:.2f} / Act {act_mm:.2f} mm"

    st.markdown(
        f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
    <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
  </div>
  <div class="topbar-right">
    <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
    <div class="badge"><span>{qa_mode}</span></div>
    <div class="badge"><span>{tol_tag}</span></div>
    <div class="badge"><span>{lr}</span></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _clear_log_related_state() -> None:
    for k in (
        "df_all",
        "df_upper",
        "df_lower",
        "df_errors",
        "mU",
        "mL",
        "analysis_out",
        "pdf_bytes",
        "pdf_name",
        "trend_all",
        "last_parsed_sig",
        "upload_complete",
        "analysis_ready",
        "last_upload_signature",
        "last_run",
    ):
        st.session_state.pop(k, None)
    _parse_uploaded_texts_cached.clear()


def _get_bins_from_merged(df: pd.DataFrame) -> list[int]:
    if df is None or df.empty:
        return []
    if "GantryBin" in df.columns:
        b = pd.to_numeric(df["GantryBin"], errors="coerce").dropna()
        if not b.empty:
            return sorted({int(round(x)) for x in b.to_numpy(dtype=float)})
    if "GantryDeg" in df.columns:
        g = pd.to_numeric(df["GantryDeg"], errors="coerce").dropna()
        if not g.empty:
            bins = np.floor((g % 360.0) / 90.0) * 90.0
            return sorted({int(round(x)) for x in bins.to_numpy(dtype=float)})
    return []


# =============================================================================
# Sidebar (clean) + tolerance profiles
# =============================================================================
with st.sidebar:
    st.markdown(f"### {APP_SHORT_NAME}")
    st.caption("Platform for Radiotherapy QA")

    st.markdown(
        """
<div class="sidebar-card">
  <h4>QA mode</h4>
</div>
""",
        unsafe_allow_html=True,
    )

    qa_mode = st.selectbox(
        "Workflow",
        options=["PF Log-file analysis", "PF Film analysis"],
        index=0 if st.session_state.get("qa_mode") != "PF Film analysis" else 1,
        help="PF Log-file analysis runs the delivery-log pipeline. PF Film analysis is a placeholder (to be implemented).",
    )
    if qa_mode != st.session_state.get("qa_mode"):
        st.session_state["qa_mode"] = qa_mode
        if qa_mode == "PF Film analysis":
            _clear_log_related_state()

    st.markdown(
        """
<div class="sidebar-card">
  <h4>QA criteria</h4>
</div>
""",
        unsafe_allow_html=True,
    )

    profiles = {
        "MPPG (0.5/1.0)": (0.5, 1.0),
        "TG-142 (1.0/1.5)": (1.0, 1.5),
        "Custom": None,
    }

    profile_keys = list(profiles.keys())
    current_profile = st.session_state.get("tol_profile", profile_keys[0])
    if current_profile not in profiles:
        current_profile = profile_keys[0]

    profile = st.selectbox("Tolerance profile", profile_keys, index=profile_keys.index(current_profile))
    st.session_state["tol_profile"] = profile

    if profile == "Custom":
        tol_mm = st.number_input(
            "Tolerance (mm)",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state.get("tolerance_mm", 0.5)),
            step=0.1,
        )
        act_mm = st.number_input(
            "Action (mm)",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state.get("action_mm", 1.0)),
            step=0.1,
        )
    else:
        tol_mm, act_mm = profiles[profile]

    tol_mm = float(tol_mm)
    act_mm = float(act_mm)
    if act_mm < tol_mm:
        st.warning("Action should be ≥ tolerance. Adjusting action to match tolerance.")
        act_mm = tol_mm

    st.session_state["tolerance_mm"] = tol_mm
    st.session_state["action_mm"] = act_mm

    st.markdown(
        f"""
<div class="sidebar-card">
  <div class="sidebar-muted">
    <b>Selected criteria</b>: {profile}<br/>
    <b>Tolerance</b>: {tol_mm:.2f} mm<br/>
    <b>Action</b>: {act_mm:.2f} mm<br/>
    PASS ≤ {tol_mm:.2f} &nbsp;|&nbsp; WARN ({tol_mm:.2f}, {act_mm:.2f}] &nbsp;|&nbsp; FAIL &gt; {act_mm:.2f}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="sidebar-card">
  <h4>Matching</h4>
</div>
""",
        unsafe_allow_html=True,
    )

    st.session_state["drop_stack_by_plan_name"] = st.checkbox(
        "Auto-select stacks by plan naming convention",
        value=st.session_state.get("drop_stack_by_plan_name", True),
        disabled=(st.session_state.get("qa_mode") == "PF Film analysis"),
    )
    st.caption("Merge runs in Analysis tab (log mode only).")

    with st.expander("Help / SOP"):
        st.markdown(
            """
- **Upload**: provide both **upper** and **lower** stack MRIdian log **.txt** files.  
- **Analysis**: click **Run analysis** to match plan PKLs with delivery records and compute errors.  
- **Reports**: enter Site/Machine/Reviewer (optional) and generate a PDF.  
- **Trends**: append the overall max error to a CSV history and plot drift over time.  
"""
        )

    st.markdown("---")
    st.caption(f"Version {APP_VERSION} • Research & QA use only • For technical issues: osmanaf@vcu.edu")

# =============================================================================
# Header
# =============================================================================
render_topbar(
    st.session_state.get("qa_mode", "PF Log-file analysis"),
    tol_mm=float(st.session_state["tolerance_mm"]),
    act_mm=float(st.session_state["action_mm"]),
)

st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
if st.session_state.get("qa_mode") == "PF Film analysis":
    st.markdown(
        '<div class="section-sub">Film-based picket fence QA mode is selected (pipeline will be implemented next).</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="section-sub">Log-file–based verification of MRIdian MLC positional accuracy with standardized reporting and longitudinal trending.</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# Tabs
# =============================================================================
tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
    ["📤 Upload & Intake", "📊 Analysis", "📄 Reports & Export", "📈 Longitudinal Trends"]
)

# =============================================================================
# TAB 1: UPLOAD & INTAKE
# =============================================================================
with tab_upload:
    st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)

    if st.session_state.get("qa_mode") == "PF Film analysis":
        with st.container(border=True):
            st.markdown("**Film QA (Coming soon)**")
            st.caption("Next: add film image uploader + picket detection + offsets table.")
        st.info("Switch back to **PF Log-file analysis** in the sidebar to use the current log pipeline.")
        st.stop()

    st.markdown(
        '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.12, 0.88], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
            uploaded = st.file_uploader(
                "Drag files here or browse",
                type=["txt"],
                accept_multiple_files=True,
                key="log_uploader",
            )

        _reset_results_on_new_upload(uploaded)

        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        clear_btn = st.button("Clear session", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if clear_btn:
            keep = {
                "qa_mode",
                "drop_stack_by_plan_name",
                "site_name",
                "machine_name",
                "reviewer_name",
                "tol_profile",
                "tolerance_mm",
                "action_mm",
            }
            for k in list(st.session_state.keys()):
                if k not in keep:
                    st.session_state.pop(k, None)
            _parse_uploaded_texts_cached.clear()
            st.toast("Session cleared.", icon="🧹")

        if not uploaded:
            st.session_state["system_status"] = "ready"
            st.session_state["analysis_ready"] = False
            st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
        else:
            st.session_state["system_status"] = "parsing"
            with st.spinner("Parsing delivery logs…"):
                df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
            st.session_state["system_status"] = "ready"

            if df_errors is not None and len(df_errors) > 0:
                st.warning("Some files could not be parsed.")
                with st.expander("Parsing details"):
                    st.dataframe(df_errors, use_container_width=True)

            n_upper = 0 if df_upper is None else int(len(df_upper))
            n_lower = 0 if df_lower is None else int(len(df_lower))

            m1, m2, m3 = st.columns(3)
            m1.metric("Uploaded files", len(uploaded))
            m2.metric("Upper stack records", n_upper)
            m3.metric("Lower stack records", n_lower)

            chips = []
            chips.append("Upper ✅" if n_upper > 0 else "Upper ❌")
            chips.append("Lower ✅" if n_lower > 0 else "Lower ❌")
            chips.append("Plan PKLs ✅" if _plans_available(PLAN_FOLDER) else "Plan PKLs ❌")
            st.caption(" • ".join(chips))

            if n_upper == 0 or n_lower == 0:
                st.session_state["upload_complete"] = False
                st.session_state["analysis_ready"] = False
                st.session_state["system_status"] = "error"
                missing = "Upper" if n_upper == 0 else "Lower"
                present = "Lower" if n_upper == 0 else "Upper"
                st.error(
                    f"**Incomplete upload — {missing} stack missing.** "
                    f"Full QA requires at least one {present} and one {missing} stack log."
                )
            else:
                st.session_state["upload_complete"] = True
                st.session_state["analysis_ready"] = True
                st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Run analysis**.")

            with st.expander("Data preview (verification)"):
                st.write("All records")
                st.dataframe(df_all.head(PREVIEW_ROWS), use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Upper stack")
                    st.dataframe(df_upper.head(PREVIEW_ROWS), use_container_width=True)
                with c2:
                    st.write("Lower stack")
                    st.dataframe(df_lower.head(PREVIEW_ROWS), use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown("**Plan Reference (Local)**")

            if _plans_available(PLAN_FOLDER):
                st.success("Plan reference detected in `./data` ✅")
                if st.session_state.get("system_status") == "missing_plan":
                    st.session_state["system_status"] = "ready"
            else:
                st.session_state["system_status"] = "missing_plan"
                st.error("Plan reference files not found in `./data` ❌")
                st.caption("Place the three PKLs in `./data` to enable matching.")

            st.markdown("---")
            st.markdown("**Notes**")
            st.caption("• Upload both upper & lower stack logs (auto-parsed).")
            st.caption("• Click **Run analysis** in the Analysis tab to run matching + QA.")
            st.caption("• Reports require completed analysis.")

# =============================================================================
# TAB 2: ANALYSIS
# =============================================================================
with tab_analysis:
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

    if st.session_state.get("qa_mode") == "PF Film analysis":
        st.warning("Film picket fence analysis is not implemented yet. Switch to **PF Log-file analysis** to run QA.")
        st.stop()

    st.markdown(
        '<div class="section-sub">Match delivery logs to nominal plan PKLs and compute MLC positional accuracy metrics.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
        st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
        st.stop()

    if not _plans_available(PLAN_FOLDER):
        st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
        st.stop()

    tol_mm = float(st.session_state["tolerance_mm"])
    act_mm = float(st.session_state["action_mm"])

    cA, cB, cC = st.columns([0.24, 0.26, 0.50], gap="small")
    with cA:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        analyze_btn = st.button("Run analysis", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        reset_btn = st.button("Reset cache", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cC:
        st.caption("Run after upload. Reset if plan PKLs or matching option changed.")

    if reset_btn:
        for k in ("mU", "mL", "analysis_out"):
            st.session_state.pop(k, None)
        st.toast("Matching cache cleared.", icon="🔁")

    if not analyze_btn and "analysis_out" not in st.session_state:
        st.info("Ready. Click **Run analysis** to compute QA metrics.")
        st.stop()

    try:
        st.session_state["system_status"] = "parsing"
        with st.spinner("Matching plan to delivery logs…"):
            _ensure_merged()
        st.session_state["system_status"] = "ready"
        st.success(f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}")
    except Exception as e:
        st.session_state["system_status"] = "error"
        st.error(str(e))
        st.stop()

    mU = st.session_state["mU"]
    mL = st.session_state["mL"]

    mUe = add_errors(mU)
    mLe = add_errors(mL)
    st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

    metricsU = leaf_metrics(mUe, "upper")
    metricsL = leaf_metrics(mLe, "lower")

    statusU, maxU = classify(metricsU, warn_mm=tol_mm, fail_mm=act_mm)
    statusL, maxL = classify(metricsL, warn_mm=tol_mm, fail_mm=act_mm)

    status_all = combine_status(statusU, statusL)
    max_all = float(max(maxU, maxL))
    st.session_state["last_run"] = {
        "status": status_all,
        "max_abs_mm": max_all,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tol_mm": tol_mm,
        "act_mm": act_mm,
    }

    st.markdown('<div class="section-title">Overall QA Verdict</div>', unsafe_allow_html=True)
    _status_banner("Overall", status_all, max_all)

    st.markdown('<div class="section-title">Stack QA Status</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        _status_banner("Upper stack", statusU, maxU)
    with c2:
        _status_banner("Lower stack", statusL, maxL)

    st.markdown('<div class="section-title">Gantry Angles (Detected; binned)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Unique gantry bins present in the matched delivery records (90° bins).</div>',
        unsafe_allow_html=True,
    )

    upper_bins = _get_bins_from_merged(mUe)
    lower_bins = _get_bins_from_merged(mLe)

    upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
    lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

    st.markdown(f"<div class='sidebar-card'><b>Upper Stack (binned)</b><br>{upper_text}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-card'><b>Lower Stack (binned)</b><br>{lower_text}</div>", unsafe_allow_html=True)

    with st.expander("Detailed error statistics (mm)"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("Upper stack")
            st.dataframe(error_describe(mUe), use_container_width=True)
        with c2:
            st.write("Lower stack")
            st.dataframe(error_describe(mLe), use_container_width=True)

    st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

    picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
    gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
    gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

    figU = plot_virtual_picket_fence(
        picket_centers_mm=picket_centers,
        gap_df=gapU,
        title="Upper stack — delivered gap pattern",
    )
    figL = plot_virtual_picket_fence(
        picket_centers_mm=picket_centers,
        gap_df=gapL,
        title="Lower stack — delivered gap pattern",
    )

    colU, colL = st.columns(2, gap="large")
    with colU:
        st.pyplot(figU, clear_figure=True, use_container_width=True)
    with colL:
        st.pyplot(figL, clear_figure=True, use_container_width=True)

    st.markdown('<div class="section-title">Per-Leaf Maximum Absolute Error</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Tolerance and action thresholds are shown (MaxAbs only).</div>',
        unsafe_allow_html=True,
    )

    comb, _, _ = make_combined_leaf_index(mUe, mLe)
    st.pyplot(
        plot_max_errors_by_leaf(
            max_by_leaf(comb),
            threshold_mm=tol_mm,
            fail_mm=act_mm,
            title="Max Abs Error by Leaf (Tolerance/Action)",
            show_bands=True,
        ),
        clear_figure=True,
        use_container_width=True,
    )

    with st.expander("Per-leaf tables"):
        st.write("Upper stack per-leaf metrics")
        st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
        st.write("Lower stack per-leaf metrics")
        st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# =============================================================================
# TAB 3: REPORTS & EXPORT
# =============================================================================
with tab_reports:
    st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)

    if st.session_state.get("qa_mode") == "PF Film analysis":
        st.warning("Film report export is not implemented yet. Switch to **PF Log-file analysis** to generate PDFs.")
        st.stop()

    st.markdown(
        '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
        st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
        st.stop()

    if "analysis_out" not in st.session_state:
        st.info("Run **Analysis** first to generate report inputs.")
        st.stop()

    tol_mm = float(st.session_state["tolerance_mm"])
    act_mm = float(st.session_state["action_mm"])

    mUe = st.session_state["analysis_out"]["mUe"]
    mLe = st.session_state["analysis_out"]["mLe"]

    pid = _safe_first(mUe, "Patient ID") or "ID"
    dts = _safe_first(mUe, "Date") or "Date"
    plan = _safe_first(mUe, "Plan Name") or "Plan"
    default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dts}.pdf".replace(" ", "_").replace("/", "-")

    with st.container(border=True):
        st.markdown("**Report metadata**")
        c1, c2, c3 = st.columns(3)
        with c1:
            report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
        with c2:
            report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
        with c3:
            report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

        st.session_state["site_name"] = report_site
        st.session_state["machine_name"] = report_machine
        st.session_state["reviewer_name"] = report_reviewer

        report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")

    tolerances = {"warn_max": tol_mm, "fail_max": act_mm}

    left, right = st.columns([0.62, 0.38], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Trend logging (optional)**")
            update_trend = st.toggle("Append this run to trends", value=False)
            trend_csv_path = None
            if update_trend:
                trend_csv_path = st.text_input(
                    "Trends file path (server-side)",
                    value=str(TREND_DIR / "trend_report_bank_summary.csv"),
                )
            st.markdown("---")

    with right:
        with st.container(border=True):
            st.markdown("**Generate**")
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            gen_btn = st.button("Generate PDF report", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.caption("Produces a downloadable PDF. Optional server trend append is supported.")

    if gen_btn:
        try:
            st.session_state["system_status"] = "parsing"
            with st.spinner("Generating PDF…"):
                pdf_bytes = generate_pdf_qa_report_bytes(
                    mU=mUe,
                    mL=mLe,
                    report_title=report_title,
                    tolerances=tolerances,
                    trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
                    site=report_site,
                    machine=report_machine,
                    reviewer=report_reviewer,
                )
            st.session_state["system_status"] = "ready"
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_name"] = default_pdf_name
            st.success("PDF report generated.")
        except Exception as e:
            st.session_state["system_status"] = "error"
            st.error(f"Report generation failed: {e}")

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "Download PDF",
            data=st.session_state["pdf_bytes"],
            file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
            mime="application/pdf",
        )

# =============================================================================
# TAB 4: LONGITUDINAL TRENDS
# =============================================================================
with tab_trends:
    st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)

    if st.session_state.get("qa_mode") == "PF Film analysis":
        st.warning("Film trend tracking is not implemented yet. Switch to **PF Log-file analysis** to use trends.")
        st.stop()

    st.markdown(
        '<div class="section-sub">Track stability and drift of overall maximum absolute leaf errors over time.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
        st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
        st.stop()

    if "analysis_out" not in st.session_state:
        st.info("Run **Analysis** first to enable trend updates and plotting.")
        st.stop()

    tol_mm = float(st.session_state["tolerance_mm"])
    act_mm = float(st.session_state["action_mm"])

    mUe = st.session_state["analysis_out"]["mUe"]
    mLe = st.session_state["analysis_out"]["mLe"]

    trend_path = TREND_DIR / "trend_overall_max.csv"
    trend_path.parent.mkdir(parents=True, exist_ok=True)

    with st.container(border=True):
        st.markdown("**Trend plot settings**")
        scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)

    with st.container(border=True):
        st.markdown("**Update trend history**")
        st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        append_btn = st.button("Append current run to trends", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if append_btn:
        rows = [
            summarize_overall_max_error(mUe, scope="upper"),
            summarize_overall_max_error(mLe, scope="lower"),
            summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
        ]
        trend_all = append_trending_csv(trend_path, rows)
        st.success(f"Trends updated: {trend_path.as_posix()}")
        st.session_state["trend_all"] = trend_all

    if "trend_all" in st.session_state:
        trend_all = st.session_state["trend_all"]
    elif trend_path.exists():
        trend_all = pd.read_csv(trend_path)
        st.session_state["trend_all"] = trend_all
    else:
        trend_all = None

    if trend_all is None or len(trend_all) == 0:
        st.info("No trend history found yet. Append a run above to begin tracking.")
        st.stop()

    st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

    fig_tr = plot_overall_max_trending(
        trend_all,
        scope=scope,
        gantry_bin=None,
        fail_mm=act_mm,
        warn_mm=tol_mm,
        title="Trending: Overall Max Absolute MLC Error",
    )
    st.pyplot(fig_tr, clear_figure=True, use_container_width=True)

    with st.expander("Trend history table"):
        st.dataframe(trend_all, use_container_width=True)

    st.download_button(
        "Download trend history (CSV)",
        data=trend_all.to_csv(index=False).encode("utf-8"),
        file_name="trend_overall_max.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")



# # app.py — MRIdian MLC QA Suite (Clean Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib
# import datetime as dt

# import numpy as np
# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "Radiotherapy QA Platform"
# APP_SHORT_NAME = "RT QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🎯",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",
#     "accent": "#C99700",
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#4b5563",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# header[data-testid="stHeader"] {{
#   visibility: visible !important;
#   height: 3.25rem !important;
#   background: transparent !important;
#   border-bottom: none !important;
# }}
# header[data-testid="stHeader"] > div {{ background: transparent !important; }}

# button[data-testid="collapsedControl"] {{
#   visibility: visible !important;
#   opacity: 1 !important;
#   display: flex !important;
#   pointer-events: auto !important;
#   z-index: 9999 !important;
# }}

# /* Accent icons */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# [data-testid="stSidebar"] [data-testid*="icon"],
# [data-testid="stToolbar"] [data-testid*="icon"],
# [data-testid="stHeader"] [data-testid*="icon"],
# [data-testid="stExpander"] [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* Sidebar */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
#   z-index: 9998 !important;
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* =========================
#    Sidebar Selectbox (dark control)
#    + Dropdown Menu (white list)
#    ========================= */

# /* Closed control (dark) */
# section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.14) !important;
#   border-radius: 14px !important;
#   box-shadow: none !important;
# }}

# /* Selected value text inside closed control */
# section[data-testid="stSidebar"] div[data-baseweb="select"] span,
# section[data-testid="stSidebar"] div[data-baseweb="select"] input {{
#   color: rgba(255,255,255,0.92) !important;
#   font-weight: 750 !important;
# }}

# /* Placeholder in closed control */
# section[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder {{
#   color: rgba(255,255,255,0.72) !important;
# }}

# /* Chevron/arrow in closed control */
# section[data-testid="stSidebar"] div[data-baseweb="select"] svg,
# section[data-testid="stSidebar"] div[data-baseweb="select"] svg * {{
#   stroke: var(--accent) !important;
#   fill: var(--accent) !important;
# }}

# /* Dropdown popover above everything */
# div[data-baseweb="popover"] {{
#   z-index: 100000 !important;
# }}

# /* Dropdown container */
# div[data-baseweb="popover"] div[data-baseweb="menu"] {{
#   background: #ffffff !important;
#   border: 1px solid rgba(17,24,39,0.12) !important;
#   border-radius: 14px !important;
#   box-shadow: 0 14px 30px rgba(17, 24, 39, 0.18) !important;
#   padding: 6px !important;
# }}

# /* Menu items */
# div[data-baseweb="popover"] div[data-baseweb="menu"] li {{
#   background: #ffffff !important;
#   color: #111827 !important;
#   border-radius: 12px !important;
#   padding: 10px 12px !important;
#   margin: 4px 2px !important;
# }}

# /* Hover */
# div[data-baseweb="popover"] div[data-baseweb="menu"] li:hover {{
#   background: rgba(31,42,68,0.06) !important;
# }}

# /* Selected item highlight */
# div[data-baseweb="popover"] div[data-baseweb="menu"] li[aria-selected="true"] {{
#   background: rgba(17,24,39,0.04) !important;
#   border: 1px solid rgba(17,24,39,0.08) !important;
# }}

# /* Force dropdown text to dark (prevents inheriting sidebar white) */
# div[data-baseweb="popover"] * {{
#   color: #111827 !important;
# }}

# /* Sidebar cards */
# .sidebar-card {{
#   background: rgba(255,255,255,0.06);
#   border: 1px solid rgba(255,255,255,0.10);
#   border-radius: 14px;
#   padding: 12px 12px;
#   margin: 10px 0 10px 0;
# }}
# .sidebar-card h4 {{
#   margin: 0 0 8px 0;
#   font-size: 0.95rem;
#   font-weight: 850;
#   color: rgba(255,255,255,0.96);
# }}
# .sidebar-muted {{
#   color: rgba(255,255,255,0.78) !important;
#   font-size: 0.86rem;
#   line-height: 1.35;
# }}

# /* ========= FIX #1: Sidebar expander header turning white =========
#    Streamlit expander uses <details><summary>. When you click it, summary background
#    can become white. Force it to keep the dark sidebar look. */
# section[data-testid="stSidebar"] details {{
#   background: rgba(255,255,255,0.06) !important;
#   border: 1px solid rgba(255,255,255,0.10) !important;
#   border-radius: 14px !important;
#   overflow: hidden !important;
# }}
# section[data-testid="stSidebar"] details > summary {{
#   background: rgba(255,255,255,0.06) !important;
#   padding: 12px 12px !important;
#   border-radius: 14px !important;
# }}
# section[data-testid="stSidebar"] details[open] > summary {{
#   background: rgba(255,255,255,0.06) !important; /* keep dark-ish */
#   border-bottom: 1px solid rgba(255,255,255,0.10) !important;
# }}
# section[data-testid="stSidebar"] details > summary * {{
#   color: rgba(255,255,255,0.92) !important;
# }}

# /* ========= FIX #2: Inline code pill (like `.txt`) looking like a white box =========
#    Make code in sidebar match the sidebar theme. */
# section[data-testid="stSidebar"] code {{
#   background: rgba(255,255,255,0.10) !important;
#   color: rgba(255,255,255,0.92) !important;
#   border: 1px solid rgba(255,255,255,0.14) !important;
#   border-radius: 8px !important;
#   padding: 0.05rem 0.35rem !important;
# }}

# /* Topbar */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{ display:flex; flex-direction:column; gap:2px; }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 900;
#   color: var(--text);
# }}
# .brand-sub {{
#   font-size: 0.90rem;
#   color: var(--muted);
#   line-height: 1.45;
# }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width:9px; height:9px; border-radius:50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# /* Main cards */
# .card {{
#   position: relative;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .card::before {{
#   content: "";
#   position: absolute;
#   left: 0;
#   top: 14px;
#   bottom: 14px;
#   width: 3px;
#   border-radius: 999px;
#   background: rgba(201,151,0,0.55);
# }}

# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 900;
#   font-size: 1.08rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
#   line-height: 1.45;
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 700 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 900;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); line-height: 1.4; }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* Tabs */
# div[data-testid="stTabs"] {{ width: 100%; }}
# div[data-baseweb="tab-list"] {{
#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
# }}
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;
#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 900 !important;
#   font-size: 0.96rem !important;
#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
#   border-bottom: none !important;
# }}
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Defaults / constants
# # =============================================================================
# PREVIEW_ROWS = 100
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",
#         "upload_complete": False,
#         "analysis_ready": False,
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#         "qa_mode": "PF Log-file analysis",
#         "site_name": "",
#         "machine_name": "MRIdian",
#         "reviewer_name": "",
#         # tolerances (dynamic)
#         "tol_profile": "MPPG (0.5/1.0)",
#         "tolerance_mm": 0.5,
#         "action_mm": 1.0,
#         # last run summary
#         "last_run": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all",
#             "df_upper",
#             "df_lower",
#             "df_errors",
#             "dfP_all",
#             "dfP_upper",
#             "dfP_lower",
#             "mU",
#             "mL",
#             "analysis_out",
#             "pdf_bytes",
#             "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#             "analysis_ready",
#             "last_run",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     out = preprocess_and_merge(
#         dfP_upper=st.session_state["dfP_upper"],
#         dfP_lower=st.session_state["dfP_lower"],
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Warning (between tolerance and action)"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Action level exceeded"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def combine_status(su: str, sl: str) -> str:
#     order = {"PASS": 0, "WARN": 1, "FAIL": 2}
#     su = (su or "FAIL").upper().strip()
#     sl = (sl or "FAIL").upper().strip()
#     return max([su, sl], key=lambda x: order.get(x, 2))


# def render_topbar(qa_mode: str, tol_mm: float, act_mm: float) -> None:
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

#     last_run = st.session_state.get("last_run")
#     if isinstance(last_run, dict) and last_run.get("status"):
#         lr = f"Last QA: {last_run['status']} • {last_run.get('timestamp','')}"
#     else:
#         lr = "Last QA: —"

#     tol_tag = f"Tol {tol_mm:.2f} / Act {act_mm:.2f} mm"

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{qa_mode}</span></div>
#     <div class="badge"><span>{tol_tag}</span></div>
#     <div class="badge"><span>{lr}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _clear_log_related_state() -> None:
#     for k in (
#         "df_all",
#         "df_upper",
#         "df_lower",
#         "df_errors",
#         "mU",
#         "mL",
#         "analysis_out",
#         "pdf_bytes",
#         "pdf_name",
#         "trend_all",
#         "last_parsed_sig",
#         "upload_complete",
#         "analysis_ready",
#         "last_upload_signature",
#         "last_run",
#     ):
#         st.session_state.pop(k, None)
#     _parse_uploaded_texts_cached.clear()


# def _get_bins_from_merged(df: pd.DataFrame) -> list[int]:
#     if df is None or df.empty:
#         return []
#     if "GantryBin" in df.columns:
#         b = pd.to_numeric(df["GantryBin"], errors="coerce").dropna()
#         if not b.empty:
#             return sorted({int(round(x)) for x in b.to_numpy(dtype=float)})
#     if "GantryDeg" in df.columns:
#         g = pd.to_numeric(df["GantryDeg"], errors="coerce").dropna()
#         if not g.empty:
#             bins = np.floor((g % 360.0) / 90.0) * 90.0
#             return sorted({int(round(x)) for x in bins.to_numpy(dtype=float)})
#     return []


# # =============================================================================
# # Sidebar (clean) + tolerance profiles
# # =============================================================================
# with st.sidebar:
#     st.markdown(f"### {APP_SHORT_NAME}")
#     st.caption("Platform for Radiotherapy QA")

#     st.markdown(
#         """
# <div class="sidebar-card">
#   <h4>QA mode</h4>
# </div>
# """,
#         unsafe_allow_html=True,
#     )

#     qa_mode = st.selectbox(
#         "Workflow",
#         options=["PF Log-file analysis", "PF Film analysis"],
#         index=0 if st.session_state.get("qa_mode") != "PF Film analysis" else 1,
#         help="PF Log-file analysis runs the delivery-log pipeline. PF Film analysis is a placeholder (to be implemented).",
#     )
#     if qa_mode != st.session_state.get("qa_mode"):
#         st.session_state["qa_mode"] = qa_mode
#         if qa_mode == "PF Film analysis":
#             _clear_log_related_state()

#     st.markdown(
#         """
# <div class="sidebar-card">
#   <h4>QA criteria</h4>
# </div>
# """,
#         unsafe_allow_html=True,
#     )

#     # --- Criteria profiles (as you requested: MPPG + TG-142) ---
#     profiles = {
#         "MPPG (0.5/1.0)": (0.5, 1.0),
#         "TG-142 (1.0/1.5)": (1.0, 1.5),
#         "Custom": None,
#     }

#     profile_keys = list(profiles.keys())
#     current_profile = st.session_state.get("tol_profile", profile_keys[0])
#     if current_profile not in profiles:
#         current_profile = profile_keys[0]

#     profile = st.selectbox(
#         "Tolerance profile",
#         profile_keys,
#         index=profile_keys.index(current_profile),
#     )
#     st.session_state["tol_profile"] = profile

#     if profile == "Custom":
#         tol_mm = st.number_input(
#             "Tolerance (mm)",
#             min_value=0.0,
#             max_value=5.0,
#             value=float(st.session_state.get("tolerance_mm", 0.5)),
#             step=0.1,
#         )
#         act_mm = st.number_input(
#             "Action (mm)",
#             min_value=0.0,
#             max_value=5.0,
#             value=float(st.session_state.get("action_mm", 1.0)),
#             step=0.1,
#         )
#     else:
#         tol_mm, act_mm = profiles[profile]

#     tol_mm = float(tol_mm)
#     act_mm = float(act_mm)
#     if act_mm < tol_mm:
#         st.warning("Action should be ≥ tolerance. Adjusting action to match tolerance.")
#         act_mm = tol_mm

#     st.session_state["tolerance_mm"] = tol_mm
#     st.session_state["action_mm"] = act_mm

#     st.markdown(
#         f"""
# <div class="sidebar-card">
#   <div class="sidebar-muted">
#     <b>Selected criteria</b>: {profile}<br/>
#     <b>Tolerance</b>: {tol_mm:.2f} mm<br/>
#     <b>Action</b>: {act_mm:.2f} mm<br/>
#     PASS ≤ {tol_mm:.2f} &nbsp;|&nbsp; WARN ({tol_mm:.2f}, {act_mm:.2f}] &nbsp;|&nbsp; FAIL &gt; {act_mm:.2f}
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )

#     st.markdown(
#         """
# <div class="sidebar-card">
#   <h4>Matching</h4>
# </div>
# """,
#         unsafe_allow_html=True,
#     )
#     st.session_state["drop_stack_by_plan_name"] = st.checkbox(
#         "Auto-select stacks by plan naming convention",
#         value=st.session_state.get("drop_stack_by_plan_name", True),
#         disabled=(st.session_state.get("qa_mode") == "PF Film analysis"),
#     )
#     st.caption("Merge runs in Analysis tab (log mode only).")

#     with st.expander("Help / SOP"):
#         # NOTE: intentionally NOT using backticks around .txt to avoid the white “pill” look.
#         st.markdown(
#             """
# - **Upload**: provide both **upper** and **lower** stack MRIdian log **.txt** files.  
# - **Analysis**: click **Run analysis** to match plan PKLs with delivery records and compute errors.  
# - **Reports**: enter Site/Machine/Reviewer (optional) and generate a PDF.  
# - **Trends**: append the overall max error to a CSV history and plot drift over time.  
# """
#         )

#     st.markdown("---")
#     st.caption(f"Version {APP_VERSION} • Research & QA use only • For technical issues: osmanaf@vcu.edu")


# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(
#     st.session_state.get("qa_mode", "PF Log-file analysis"),
#     tol_mm=float(st.session_state["tolerance_mm"]),
#     act_mm=float(st.session_state["action_mm"]),
# )

# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# if st.session_state.get("qa_mode") == "PF Film analysis":
#     st.markdown(
#         '<div class="section-sub">Film-based picket fence QA mode is selected (pipeline will be implemented next).</div>',
#         unsafe_allow_html=True,
#     )
# else:
#     st.markdown(
#         '<div class="section-sub">Log-file–based verification of MRIdian MLC positional accuracy with standardized reporting and longitudinal trending.</div>',
#         unsafe_allow_html=True,
#     )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["📤 Upload & Intake", "📊 Analysis", "📄 Reports & Export", "📈 Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "PF Film analysis":
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Film QA (Coming soon)**")
#         st.caption("Next: add film image uploader + picket detection + offsets table.")
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.info("Switch back to **PF Log-file analysis** in the sidebar to use the current log pipeline.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         clear_btn = st.button("Clear session", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             keep = {
#                 "qa_mode",
#                 "drop_stack_by_plan_name",
#                 "site_name",
#                 "machine_name",
#                 "reviewer_name",
#                 "tol_profile",
#                 "tolerance_mm",
#                 "action_mm",
#             }
#             for k in list(st.session_state.keys()):
#                 if k not in keep:
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.session_state["analysis_ready"] = False
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Parsing delivery logs…"):
#                 df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#             st.session_state["system_status"] = "ready"

#             if df_errors is not None and len(df_errors) > 0:
#                 st.warning("Some files could not be parsed.")
#                 with st.expander("Parsing details"):
#                     st.dataframe(df_errors, use_container_width=True)

#             n_upper = 0 if df_upper is None else int(len(df_upper))
#             n_lower = 0 if df_lower is None else int(len(df_lower))

#             m1, m2, m3 = st.columns(3)
#             m1.metric("Uploaded files", len(uploaded))
#             m2.metric("Upper stack records", n_upper)
#             m3.metric("Lower stack records", n_lower)

#             chips = []
#             chips.append("Upper ✅" if n_upper > 0 else "Upper ❌")
#             chips.append("Lower ✅" if n_lower > 0 else "Lower ❌")
#             chips.append("Plan PKLs ✅" if _plans_available(PLAN_FOLDER) else "Plan PKLs ❌")
#             st.caption(" • ".join(chips))

#             if n_upper == 0 or n_lower == 0:
#                 st.session_state["upload_complete"] = False
#                 st.session_state["analysis_ready"] = False
#                 st.session_state["system_status"] = "error"
#                 missing = "Upper" if n_upper == 0 else "Lower"
#                 present = "Lower" if n_upper == 0 else "Upper"
#                 st.error(
#                     f"**Incomplete upload — {missing} stack missing.** "
#                     f"Full QA requires at least one {present} and one {missing} stack log."
#                 )
#             else:
#                 st.session_state["upload_complete"] = True
#                 st.session_state["analysis_ready"] = True
#                 st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Run analysis**.")

#             with st.expander("Data preview (verification)"):
#                 st.write("All records")
#                 st.dataframe(df_all.head(PREVIEW_ROWS), use_container_width=True)
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("Upper stack")
#                     st.dataframe(df_upper.head(PREVIEW_ROWS), use_container_width=True)
#                 with c2:
#                     st.write("Lower stack")
#                     st.dataframe(df_lower.head(PREVIEW_ROWS), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Plan Reference (Local)**")

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Notes**")
#         st.caption("• Upload both upper & lower stack logs (auto-parsed).")
#         st.caption("• Click **Run analysis** in the Analysis tab to run matching + QA.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "PF Film analysis":
#         st.warning("Film picket fence analysis is not implemented yet. Switch to **PF Log-file analysis** to run QA.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Match delivery logs to nominal plan PKLs and compute MLC positional accuracy metrics.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     tol_mm = float(st.session_state["tolerance_mm"])
#     act_mm = float(st.session_state["action_mm"])

#     cA, cB, cC = st.columns([0.24, 0.26, 0.50], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         analyze_btn = st.button("Run analysis", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         reset_btn = st.button("Reset cache", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cC:
#         st.caption("Run after upload. Reset if plan PKLs or matching option changed.")

#     if reset_btn:
#         for k in ("mU", "mL", "analysis_out"):
#             st.session_state.pop(k, None)
#         st.toast("Matching cache cleared.", icon="🔁")

#     if not analyze_btn and "analysis_out" not in st.session_state:
#         st.info("Ready. Click **Run analysis** to compute QA metrics.")
#         st.stop()

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}")
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")

#     statusU, maxU = classify(metricsU, warn_mm=tol_mm, fail_mm=act_mm)
#     statusL, maxL = classify(metricsL, warn_mm=tol_mm, fail_mm=act_mm)

#     status_all = combine_status(statusU, statusL)
#     max_all = float(max(maxU, maxL))
#     st.session_state["last_run"] = {
#         "status": status_all,
#         "max_abs_mm": max_all,
#         "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
#         "tol_mm": tol_mm,
#         "act_mm": act_mm,
#     }

#     st.markdown('<div class="section-title">Overall QA Verdict</div>', unsafe_allow_html=True)
#     _status_banner("Overall", status_all, max_all)

#     st.markdown('<div class="section-title">Stack QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     st.markdown('<div class="section-title">Gantry Angles (Detected; binned)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Unique gantry bins present in the matched delivery records (90° bins).</div>',
#         unsafe_allow_html=True,
#     )

#     upper_bins = _get_bins_from_merged(mUe)
#     lower_bins = _get_bins_from_merged(mLe)

#     upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
#     lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

#     st.markdown(f"<div class='card'><b>Upper Stack (binned)</b><br>{upper_text}</div>", unsafe_allow_html=True)
#     st.markdown(f"<div class='card'><b>Lower Stack (binned)</b><br>{lower_text}</div>", unsafe_allow_html=True)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     # =====================================================================
#     # Virtual picket fence (SIDE-BY-SIDE)  ✅ as requested
#     # =====================================================================
#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     figU = plot_virtual_picket_fence(
#         picket_centers_mm=picket_centers,
#         gap_df=gapU,
#         title="Upper stack — delivered gap pattern",
#     )
#     figL = plot_virtual_picket_fence(
#         picket_centers_mm=picket_centers,
#         gap_df=gapL,
#         title="Lower stack — delivered gap pattern",
#     )

#     colU, colL = st.columns(2, gap="large")
#     with colU:
#         st.pyplot(figU, clear_figure=True, use_container_width=True)
#     with colL:
#         st.pyplot(figL, clear_figure=True, use_container_width=True)

#     st.markdown('<div class="section-title">Per-Leaf Maximum Absolute Error</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Tolerance and action thresholds are shown (MaxAbs only).</div>',
#         unsafe_allow_html=True,
#     )

#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(
#         plot_max_errors_by_leaf(
#             max_by_leaf(comb),
#             threshold_mm=tol_mm,
#             fail_mm=act_mm,
#             title="Max Abs Error by Leaf (Tolerance/Action)",
#             show_bands=True,
#         ),
#         clear_figure=True,
#         use_container_width=True,
#     )

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "PF Film analysis":
#         st.warning("Film report export is not implemented yet. Switch to **PF Log-file analysis** to generate PDFs.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate report inputs.")
#         st.stop()

#     tol_mm = float(st.session_state["tolerance_mm"])
#     act_mm = float(st.session_state["action_mm"])

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dts = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dts}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     st.session_state["site_name"] = report_site
#     st.session_state["machine_name"] = report_machine
#     st.session_state["reviewer_name"] = report_reviewer

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {"warn_max": tol_mm, "fail_max": act_mm}

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Produces a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "PF Film analysis":
#         st.warning("Film trend tracking is not implemented yet. Switch to **PF Log-file analysis** to use trends.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Track stability and drift of overall maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     tol_mm = float(st.session_state["tolerance_mm"])
#     act_mm = float(st.session_state["action_mm"])

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(
#         trend_all,
#         scope=scope,
#         gantry_bin=None,
#         fail_mm=act_mm,
#         warn_mm=tol_mm,
#         title="Trending: Overall Max Absolute MLC Error",
#     )
#     st.pyplot(fig_tr, clear_figure=True, use_container_width=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")



# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import numpy as np
# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MR-Guided Radiotherapy QA Platform"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",
#     "accent": "#C99700",
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# /* Keep header visible so the sidebar collapse control exists */
# header[data-testid="stHeader"] {{
#   visibility: visible !important;
#   height: 3.25rem !important;
#   background: transparent !important;
#   border-bottom: none !important;
# }}
# header[data-testid="stHeader"] > div {{
#   background: transparent !important;
# }}

# /* Ensure collapse/expand button stays clickable */
# button[data-testid="collapsedControl"] {{
#   visibility: visible !important;
#   opacity: 1 !important;
#   display: flex !important;
#   pointer-events: auto !important;
#   z-index: 9999 !important;
# }}

# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# [data-testid="stSidebar"] [data-testid*="icon"],
# [data-testid="stToolbar"] [data-testid*="icon"],
# [data-testid="stHeader"] [data-testid*="icon"],
# [data-testid="stExpander"] [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
#   z-index: 9998 !important;
# }}
# /* default sidebar text color */
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}

# section[data-testid="stSidebar"] .stTextInput input {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}

# /* ===== FIX: Sidebar selectbox readability ===== */
# /* Make selectbox control match sidebar (dark translucent) so white text is readable */
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# /* Selected value text inside the control */
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# /* Placeholder */
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}

# /* Dropdown menu (popover) should be white background with dark text */
# div[data-baseweb="popover"] {{
#   z-index: 100000 !important;
# }}
# div[data-baseweb="popover"] * {{
#   color: var(--text) !important;
# }}
# div[data-baseweb="menu"] {{
#   background: #ffffff !important;
# }}
# div[data-baseweb="menu"] li {{
#   background: #ffffff !important;
# }}
# div[data-baseweb="menu"] li:hover {{
#   background: rgba(31,42,68,0.06) !important;
# }}
# /* ===== end fix ===== */

# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{ display:flex; flex-direction:column; gap:2px; }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{ font-size: 0.88rem; color: var(--muted); }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{ width:9px; height:9px; border-radius:50%; background: var(--muted); }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# div[data-testid="stTabs"] {{ width: 100%; }}
# div[data-baseweb="tab-list"] {{
#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
# }}
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;
#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 850 !important;
#   font-size: 0.96rem !important;
#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
# }}
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # QA thresholds (MAX ERROR ONLY)
# # =============================================================================
# TOLERANCE_MM = 0.5   # PASS <= 0.5
# ACTION_MM = 1.0      # WARN (0.5, 1.0], FAIL > 1.0
# PREVIEW_ROWS = 100

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",
#         "upload_complete": False,
#         "analysis_ready": False,
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#         "qa_mode": "Log-file analysis",

#         # Stored report context (NO sidebar inputs)
#         "site_name": "",
#         "machine_name": "MRIdian",
#         "reviewer_name": "",
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#             "analysis_ready",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     out = preprocess_and_merge(
#         dfP_upper=st.session_state["dfP_upper"],
#         dfP_lower=st.session_state["dfP_lower"],
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Warning (between tolerance and action)"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Action level exceeded"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(qa_mode: str) -> None:
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]
#     ctx = qa_mode or "Workflow"

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _clear_log_related_state() -> None:
#     """Clear only log-related objects when switching to film mode."""
#     for k in (
#         "df_all", "df_upper", "df_lower", "df_errors",
#         "mU", "mL", "analysis_out",
#         "pdf_bytes", "pdf_name",
#         "trend_all",
#         "last_parsed_sig",
#         "upload_complete",
#         "analysis_ready",
#         "last_upload_signature",
#     ):
#         st.session_state.pop(k, None)
#     _parse_uploaded_texts_cached.clear()


# # =============================================================================
# # Sidebar (clean) + QA mode dropdown (ONLY)
# # =============================================================================
# with st.sidebar:
#     st.markdown(f"### {APP_SHORT_NAME}")
#     st.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
#     st.markdown("---")

#     st.subheader("QA mode")
#     qa_mode = st.selectbox(
#         "Workflow",
#         options=["Log-file analysis", "Film analysis"],
#         index=0 if st.session_state.get("qa_mode") != "Film analysis" else 1,
#         help="Log-file analysis runs the delivery-log pipeline. Film analysis is a placeholder (to be implemented).",
#     )

#     if qa_mode != st.session_state.get("qa_mode"):
#         st.session_state["qa_mode"] = qa_mode
#         if qa_mode == "Film analysis":
#             _clear_log_related_state()

#     st.markdown("---")
#     st.subheader("QA criteria (Max error, mm)")
#     st.markdown(
#         f"""
# - **Tolerance:** `{TOLERANCE_MM:.2f}` mm  
# - **Action:** `{ACTION_MM:.2f}` mm  
# - **PASS:** ≤ {TOLERANCE_MM:.2f}  
# - **WARN:** ({TOLERANCE_MM:.2f}, {ACTION_MM:.2f}]  
# - **FAIL:** > {ACTION_MM:.2f}
# """.strip()
#     )

#     st.markdown("---")
#     st.subheader("Matching options")
#     st.session_state["drop_stack_by_plan_name"] = st.checkbox(
#         "Auto-select stacks by plan naming convention",
#         value=st.session_state.get("drop_stack_by_plan_name", True),
#         disabled=(st.session_state.get("qa_mode") == "Film analysis"),
#     )
#     st.caption("Merge runs in Analysis tab (log mode only).")
#     st.markdown("---")
#     st.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")

# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(st.session_state.get("qa_mode", "Log-file analysis"))
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

# if st.session_state.get("qa_mode") == "Film analysis":
#     st.markdown(
#         '<div class="section-sub">Film-based picket fence QA mode is selected (pipeline will be implemented next).</div>',
#         unsafe_allow_html=True,
#     )
# else:
#     st.markdown(
#         '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#         unsafe_allow_html=True,
#     )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["📤 Upload & Intake", "📊 Analysis", "📄 Reports & Export", "📈 Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "Film analysis":
#         st.markdown(
#             '<div class="section-sub">Film QA selected. Upload UI and analysis pipeline will be added later.</div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Film QA (Coming soon)**")
#         st.caption("Next: add film image uploader + picket detection + offsets table.")
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.info("Switch back to **Log-file analysis** in the sidebar to use the current log pipeline.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         clear_btn = st.button("Clear session data", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("drop_stack_by_plan_name", "qa_mode", "site_name", "machine_name", "reviewer_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.session_state["analysis_ready"] = False
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Parsing delivery logs…"):
#                 df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#             st.session_state["system_status"] = "ready"

#             if df_errors is not None and len(df_errors) > 0:
#                 st.warning("Some files could not be parsed.")
#                 with st.expander("Parsing details"):
#                     st.dataframe(df_errors, use_container_width=True)

#             n_upper = 0 if df_upper is None else int(len(df_upper))
#             n_lower = 0 if df_lower is None else int(len(df_lower))

#             m1, m2, m3 = st.columns(3)
#             m1.metric("Uploaded files", len(uploaded))
#             m2.metric("Upper stack records", n_upper)
#             m3.metric("Lower stack records", n_lower)

#             if n_upper == 0 or n_lower == 0:
#                 st.session_state["upload_complete"] = False
#                 st.session_state["analysis_ready"] = False
#                 st.session_state["system_status"] = "error"
#                 missing = "Upper" if n_upper == 0 else "Lower"
#                 present = "Lower" if n_upper == 0 else "Upper"
#                 st.error(
#                     f"**Incomplete upload — {missing} stack missing.** "
#                     f"Full QA requires at least one {present} and one {missing} stack log."
#                 )
#             else:
#                 st.session_state["upload_complete"] = True
#                 st.session_state["analysis_ready"] = True
#                 st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Analyze**.")

#             with st.expander("Data preview (verification)"):
#                 st.write("All records")
#                 st.dataframe(df_all.head(PREVIEW_ROWS), use_container_width=True)
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("Upper stack")
#                     st.dataframe(df_upper.head(PREVIEW_ROWS), use_container_width=True)
#                 with c2:
#                     st.write("Lower stack")
#                     st.dataframe(df_lower.head(PREVIEW_ROWS), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Plan Reference (Local)**")

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Notes**")
#         st.caption("• Upload both upper & lower stack logs (auto-parsed).")
#         st.caption("• Click **Analyze** in the Analysis tab to run matching + QA.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "Film analysis":
#         st.markdown(
#             '<div class="section-sub">Film QA mode selected. Analysis pipeline will be implemented later.</div>',
#             unsafe_allow_html=True,
#         )
#         st.warning("Film picket fence analysis is not implemented yet. Switch to **Log-file analysis** to run QA.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Match log parsed data with nominal plan data and compute MLC positional accuracy metrics.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB, cC = st.columns([0.28, 0.36, 0.36], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         analyze_btn = st.button("Analyze", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         reset_btn = st.button("Reset matching cache", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cC:
#         st.caption("Click **Analyze** after upload. Use reset if plan PKLs or matching option changed.")

#     if reset_btn:
#         for k in ("mU", "mL", "analysis_out"):
#             st.session_state.pop(k, None)
#         st.toast("Matching cache cleared.", icon="🔁")

#     if not analyze_btn and "analysis_out" not in st.session_state:
#         st.info("Ready. Click **Analyze** to run matching + QA.")
#         st.stop()

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")

#     statusU, maxU = classify(metricsU, warn_mm=float(TOLERANCE_MM), fail_mm=float(ACTION_MM))
#     statusL, maxL = classify(metricsL, warn_mm=float(TOLERANCE_MM), fail_mm=float(ACTION_MM))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     st.markdown('<div class="section-title">Gantry Angles (Detected; binned)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Unique gantry bins present in the matched delivery records (90° bins).</div>',
#         unsafe_allow_html=True,
#     )

#     def _get_bins_from_merged(df: pd.DataFrame) -> list[int]:
#         if df is None or df.empty:
#             return []
#         if "GantryBin" in df.columns:
#             b = pd.to_numeric(df["GantryBin"], errors="coerce").dropna()
#             if not b.empty:
#                 return sorted({int(round(x)) for x in b.to_numpy(dtype=float)})
#         if "GantryDeg" in df.columns:
#             g = pd.to_numeric(df["GantryDeg"], errors="coerce").dropna()
#             if not g.empty:
#                 bins = np.floor((g % 360.0) / 90.0) * 90.0
#                 return sorted({int(round(x)) for x in bins.to_numpy(dtype=float)})
#         return []

#     upper_bins = _get_bins_from_merged(mUe)
#     lower_bins = _get_bins_from_merged(mLe)

#     upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
#     lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Upper Stack (binned)</b><br>{upper_text}</div>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Lower Stack (binned)</b><br>{lower_text}</div>",
#         unsafe_allow_html=True,
#     )

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Maximum Absolute Error</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Tolerance and action thresholds are shown (MaxAbs only).</div>',
#         unsafe_allow_html=True,
#     )

#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(
#         plot_max_errors_by_leaf(
#             max_by_leaf(comb),
#             threshold_mm=float(TOLERANCE_MM),
#             fail_mm=float(ACTION_MM),
#             title="Max Abs Error by Leaf (Tolerance/Action)",
#             show_bands=True,
#         ),
#         clear_figure=True,
#     )

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "Film analysis":
#         st.markdown(
#             '<div class="section-sub">Film QA mode selected. Reporting will be added later.</div>',
#             unsafe_allow_html=True,
#         )
#         st.warning("Film report export is not implemented yet. Switch to **Log-file analysis** to generate PDFs.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     # Persist these values as the ONLY source of truth (no sidebar inputs)
#     st.session_state["site_name"] = report_site
#     st.session_state["machine_name"] = report_machine
#     st.session_state["reviewer_name"] = report_reviewer

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_max": float(TOLERANCE_MM),
#         "fail_max": float(ACTION_MM),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)

#     if st.session_state.get("qa_mode") == "Film analysis":
#         st.markdown(
#             '<div class="section-sub">Film QA mode selected. Trending will be added later.</div>',
#             unsafe_allow_html=True,
#         )
#         st.warning("Film trend tracking is not implemented yet. Switch to **Log-file analysis** to use trends.")
#         st.stop()

#     st.markdown(
#         '<div class="section-sub">Track stability and drift of overall maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(
#         trend_all,
#         scope=scope,
#         gantry_bin=None,
#         fail_mm=float(ACTION_MM),
#         warn_mm=float(TOLERANCE_MM),
#         title="Trending: Overall Max Absolute MLC Error",
#     )
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")



# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import numpy as np
# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# /* IMPORTANT: Do NOT hide header; sidebar toggle lives there in recent Streamlit versions */
# header[data-testid="stHeader"] {{
#   visibility: visible !important;
#   height: 3.25rem !important;
#   background: transparent !important;
#   border-bottom: none !important;
# }}
# header[data-testid="stHeader"] > div {{
#   background: transparent !important;
# }}

# /* Ensure sidebar collapse/expand control is visible & clickable */
# button[data-testid="collapsedControl"] {{
#   visibility: visible !important;
#   opacity: 1 !important;
#   display: flex !important;
#   pointer-events: auto !important;
#   z-index: 9999 !important;
# }}

# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# /* UI icons only */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# [data-testid="stSidebar"] [data-testid*="icon"],
# [data-testid="stToolbar"] [data-testid*="icon"],
# [data-testid="stHeader"] [data-testid*="icon"],
# [data-testid="stExpander"] [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* Sidebar */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
#   z-index: 9998 !important; /* safety */
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important; 
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* Topbar + cards */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{ display:flex; flex-direction:column; gap:2px; }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{ font-size: 0.88rem; color: var(--muted); }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{ width:9px; height:9px; border-radius:50%; background: var(--muted); }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# /* Upload / DataFrame / Buttons */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# /* Status banner */
# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* Tabs */
# div[data-testid="stTabs"] {{ width: 100%; }}
# div[data-testid="stTabs"] > div {{
#   padding-left: 0 !important;
#   padding-right: 0 !important;
# }}
# div[data-baseweb="tab-list"] {{
#   position: relative !important;
#   display: flex !important;
#   width: 100% !important;
#   justify-content: space-between !important;
#   gap: 0 !important;

#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
#   box-shadow: none !important;
# }}
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;

#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 850 !important;
#   font-size: 0.96rem !important;

#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
#   border: 1px solid transparent !important;

#   transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease !important;
#   margin: 0 0.16rem !important;
#   position: relative !important;
# }}
# button[data-baseweb="tab"]:hover {{ background: rgba(31,42,68,0.06) !important; }}
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;

#   transform: translateY(-1px) !important;
#   box-shadow:
#     0 10px 18px rgba(17,24,39,0.06),
#     0 4px 10px rgba(201,151,0,0.10) !important;
# }}
# div[data-baseweb="tab-highlight"] {{
#   position: absolute !important;
#   left: 0 !important;
#   bottom: 6px !important;

#   height: 3px !important;
#   border-radius: 999px !important;

#   background: var(--accent) !important;
#   box-shadow: 0 8px 18px rgba(201,151,0,0.22) !important;

#   transition: transform 280ms cubic-bezier(.2,.9,.2,1), width 280ms cubic-bezier(.2,.9,.2,1) !important;
# }}
# div[data-baseweb="tab-border"] {{ display: none !important; }}
# div[data-testid="stTabs"] [role="tabpanel"] {{ animation: tabFade 200ms ease-in-out; }}
# @keyframes tabFade {{
#   from {{ opacity: 0.0; transform: translateY(3px); }}
#   to   {{ opacity: 1.0; transform: translateY(0px); }}
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "analysis_ready": False,   # logs parsed and complete
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#             "analysis_ready",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     dfP_upper = st.session_state["dfP_upper"]
#     dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def get_binned_gantry_angles(df: pd.DataFrame, bin_deg: int = 90) -> list[int]:
#     """
#     Report only detected binned angles, e.g., 0, 90, 180, 270
#     (bin_deg fixed at 90 by default; no UI slider)
#     """
#     if df is None or df.empty:
#         return []

#     gantry_col = None
#     for c in ["GantryAngle", "Gantry Angle", "Gantry", "Gantry (deg)", "gantry_angle"]:
#         if c in df.columns:
#             gantry_col = c
#             break
#     if gantry_col is None:
#         return []

#     g = pd.to_numeric(df[gantry_col], errors="coerce").dropna()
#     if g.empty:
#         return []

#     g = (g % 360.0 + 360.0) % 360.0
#     bins = np.floor(g / float(bin_deg)) * float(bin_deg)
#     return sorted({int(b) for b in bins})


# # =============================================================================
# # Sidebar
# # =============================================================================
# with st.sidebar:
#     st.markdown(f"### {APP_SHORT_NAME}")
#     st.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
#     st.markdown("---")
#     st.subheader("Site & machine")

#     site_name = st.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
#     machine_name = st.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
#     reviewer_name = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     st.session_state["site_name"] = site_name
#     st.session_state["machine_name"] = machine_name
#     st.session_state["reviewer_name"] = reviewer_name

#     st.markdown("---")
#     st.subheader("QA criteria (mm)")
#     action_level_mm = st.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
#     tolerance_level_mm = st.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

#     st.markdown("---")
#     st.subheader("Display")
#     preview_rows = st.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

#     st.markdown("---")
#     st.subheader("Report criteria (mm)")
#     rep_warn_rms = st.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
#     rep_fail_rms = st.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
#     rep_warn_max = st.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
#     rep_fail_max = st.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

#     st.markdown("---")
#     st.subheader("Matching options")
#     st.session_state["drop_stack_by_plan_name"] = st.checkbox(
#         "Auto-select stacks by plan naming convention (recommended)",
#         value=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.caption("Plan/log merge runs in Analysis tab.")
#     st.markdown("---")
#     st.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")

# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["📤 Upload & Intake", "📊 Analysis", "📄 Reports & Export", "📈 Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE (AUTO-PARSE ON UPLOAD)
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         clear_btn = st.button("Clear session data", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.session_state["analysis_ready"] = False
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Parsing delivery logs…"):
#                 df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#             st.session_state["system_status"] = "ready"

#             if df_errors is not None and len(df_errors) > 0:
#                 st.warning("Some files could not be parsed.")
#                 with st.expander("Parsing details"):
#                     st.dataframe(df_errors, use_container_width=True)

#             n_upper = 0 if df_upper is None else int(len(df_upper))
#             n_lower = 0 if df_lower is None else int(len(df_lower))

#             m1, m2, m3 = st.columns(3)
#             m1.metric("Uploaded files", len(uploaded))
#             m2.metric("Upper stack records", n_upper)
#             m3.metric("Lower stack records", n_lower)

#             if n_upper == 0 or n_lower == 0:
#                 st.session_state["upload_complete"] = False
#                 st.session_state["analysis_ready"] = False
#                 st.session_state["system_status"] = "error"
#                 missing = "Upper" if n_upper == 0 else "Lower"
#                 present = "Lower" if n_upper == 0 else "Upper"
#                 st.error(
#                     f"**Incomplete upload — {missing} stack missing.** "
#                     f"Full QA requires at least one {present} and one {missing} stack log."
#                 )
#             else:
#                 st.session_state["upload_complete"] = True
#                 st.session_state["analysis_ready"] = True
#                 st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Analyze**.")

#             with st.expander("Data preview (verification)"):
#                 st.write("All records")
#                 st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("Upper stack")
#                     st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                 with c2:
#                     st.write("Lower stack")
#                     st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Plan Reference (Local)**")

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Notes**")
#         st.caption("• Upload both upper & lower stack logs (auto-parsed).")
#         st.caption("• Click **Analyze** in the Analysis tab to run matching + QA.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS (BUTTON = ANALYZE)
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Match log parsed data with nominal plan data and compute MLC positional accuracy metrics.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB, cC = st.columns([0.28, 0.36, 0.36], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         analyze_btn = st.button("Analyze", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         reset_btn = st.button("Reset matching cache", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cC:
#         st.caption("Click **Analyze** after upload. Use reset if plan PKLs or matching option changed.")

#     if reset_btn:
#         for k in ("mU", "mL", "analysis_out"):
#             st.session_state.pop(k, None)
#         st.toast("Matching cache cleared.", icon="🔁")

#     if not analyze_btn and "analysis_out" not in st.session_state:
#         st.info("Ready. Click **Analyze** to run matching + QA.")
#         st.stop()

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     # ---- Gantry angles (binned) AFTER QA status; NO slider; report only detected bins (0/90/180/270) ----
#     st.markdown('<div class="section-title">Gantry Angles (Detected; binned)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Unique gantry bins present in the matched delivery records (90° bins).</div>',
#         unsafe_allow_html=True,
#     )

#     upper_bins = get_binned_gantry_angles(mUe, bin_deg=90)
#     lower_bins = get_binned_gantry_angles(mLe, bin_deg=90)

#     upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
#     lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Upper Stack (binned)</b><br>{upper_text}</div>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Lower Stack (binned)</b><br>{lower_text}</div>",
#         unsafe_allow_html=True,
#     )

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")

# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import numpy as np
# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# /* UI icons only */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# [data-testid="stSidebar"] [data-testid*="icon"],
# [data-testid="stToolbar"] [data-testid*="icon"],
# [data-testid="stHeader"] [data-testid*="icon"],
# [data-testid="stExpander"] [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* Sidebar */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important; 
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* Topbar + cards */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{ display:flex; flex-direction:column; gap:2px; }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{ font-size: 0.88rem; color: var(--muted); }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{ width:9px; height:9px; border-radius:50%; background: var(--muted); }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# /* Upload / DataFrame / Buttons */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# /* Status banner */
# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* Tabs */
# div[data-testid="stTabs"] {{ width: 100%; }}
# div[data-testid="stTabs"] > div {{
#   padding-left: 0 !important;
#   padding-right: 0 !important;
# }}
# div[data-baseweb="tab-list"] {{
#   position: relative !important;
#   display: flex !important;
#   width: 100% !important;
#   justify-content: space-between !important;
#   gap: 0 !important;

#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
#   box-shadow: none !important;
# }}
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;

#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 850 !important;
#   font-size: 0.96rem !important;

#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
#   border: 1px solid transparent !important;

#   transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease !important;
#   margin: 0 0.16rem !important;
#   position: relative !important;
# }}
# button[data-baseweb="tab"]:hover {{ background: rgba(31,42,68,0.06) !important; }}
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;

#   transform: translateY(-1px) !important;
#   box-shadow:
#     0 10px 18px rgba(17,24,39,0.06),
#     0 4px 10px rgba(201,151,0,0.10) !important;
# }}
# div[data-baseweb="tab-highlight"] {{
#   position: absolute !important;
#   left: 0 !important;
#   bottom: 6px !important;

#   height: 3px !important;
#   border-radius: 999px !important;

#   background: var(--accent) !important;
#   box-shadow: 0 8px 18px rgba(201,151,0,0.22) !important;

#   transition: transform 280ms cubic-bezier(.2,.9,.2,1), width 280ms cubic-bezier(.2,.9,.2,1) !important;
# }}
# div[data-baseweb="tab-border"] {{ display: none !important; }}
# div[data-testid="stTabs"] [role="tabpanel"] {{ animation: tabFade 200ms ease-in-out; }}
# @keyframes tabFade {{
#   from {{ opacity: 0.0; transform: translateY(3px); }}
#   to   {{ opacity: 1.0; transform: translateY(0px); }}
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "analysis_ready": False,   # logs parsed and complete
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#             "analysis_ready",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     dfP_upper = st.session_state["dfP_upper"]
#     dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def get_binned_gantry_angles(df: pd.DataFrame, bin_deg: int = 90) -> list[int]:
#     """
#     Report only detected binned angles, e.g., 0, 90, 180, 270
#     (bin_deg fixed at 90 by default; no UI slider)
#     """
#     if df is None or df.empty:
#         return []

#     gantry_col = None
#     for c in ["GantryAngle", "Gantry Angle", "Gantry", "Gantry (deg)", "gantry_angle"]:
#         if c in df.columns:
#             gantry_col = c
#             break
#     if gantry_col is None:
#         return []

#     g = pd.to_numeric(df[gantry_col], errors="coerce").dropna()
#     if g.empty:
#         return []

#     g = (g % 360.0 + 360.0) % 360.0
#     bins = np.floor(g / float(bin_deg)) * float(bin_deg)
#     return sorted({int(b) for b in bins})


# # =============================================================================
# # Sidebar
# # =============================================================================
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Matching options")
# st.session_state["drop_stack_by_plan_name"] = st.sidebar.checkbox(
#     "Auto-select stacks by plan naming convention (recommended)",
#     value=st.session_state.get("drop_stack_by_plan_name", True),
# )
# st.sidebar.caption("Plan/log merge runs in Analysis tab.")
# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")

# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["Upload & Intake", "Analysis", "Reports & Export", "Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE (AUTO-PARSE ON UPLOAD)
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         clear_btn = st.button("Clear session data", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.session_state["analysis_ready"] = False
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Parsing delivery logs…"):
#                 df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#             st.session_state["system_status"] = "ready"

#             if df_errors is not None and len(df_errors) > 0:
#                 st.warning("Some files could not be parsed.")
#                 with st.expander("Parsing details"):
#                     st.dataframe(df_errors, use_container_width=True)

#             n_upper = 0 if df_upper is None else int(len(df_upper))
#             n_lower = 0 if df_lower is None else int(len(df_lower))

#             m1, m2, m3 = st.columns(3)
#             m1.metric("Uploaded files", len(uploaded))
#             m2.metric("Upper stack records", n_upper)
#             m3.metric("Lower stack records", n_lower)

#             if n_upper == 0 or n_lower == 0:
#                 st.session_state["upload_complete"] = False
#                 st.session_state["analysis_ready"] = False
#                 st.session_state["system_status"] = "error"
#                 missing = "Upper" if n_upper == 0 else "Lower"
#                 present = "Lower" if n_upper == 0 else "Upper"
#                 st.error(
#                     f"**Incomplete upload — {missing} stack missing.** "
#                     f"Full QA requires at least one {present} and one {missing} stack log."
#                 )
#             else:
#                 st.session_state["upload_complete"] = True
#                 st.session_state["analysis_ready"] = True
#                 st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Analyze**.")

#             with st.expander("Data preview (verification)"):
#                 st.write("All records")
#                 st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("Upper stack")
#                     st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                 with c2:
#                     st.write("Lower stack")
#                     st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Plan Reference (Local)**")

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Notes**")
#         st.caption("• Upload both upper & lower stack logs (auto-parsed).")
#         st.caption("• Click **Analyze** in the Analysis tab to run matching + QA.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS (BUTTON = ANALYZE)
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Match log psrsed data with nominal plan data and compute MLC positional accuracy metrics.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB, cC = st.columns([0.28, 0.36, 0.36], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         analyze_btn = st.button("Analyze", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         reset_btn = st.button("Reset matching cache", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cC:
#         st.caption("Click **Analyze** after upload. Use reset if plan PKLs or matching option changed.")

#     if reset_btn:
#         for k in ("mU", "mL", "analysis_out"):
#             st.session_state.pop(k, None)
#         st.toast("Matching cache cleared.", icon="🔁")

#     if not analyze_btn and "analysis_out" not in st.session_state:
#         st.info("Ready. Click **Analyze** to run matching + QA.")
#         st.stop()

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     # ---- Gantry angles (binned) AFTER QA status; NO slider; report only detected bins (0/90/180/270) ----
#     st.markdown('<div class="section-title">Gantry Angles (Detected; binned)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Unique gantry bins present in the matched delivery records (90° bins).</div>',
#         unsafe_allow_html=True,
#     )

#     upper_bins = get_binned_gantry_angles(mUe, bin_deg=90)
#     lower_bins = get_binned_gantry_angles(mLe, bin_deg=90)

#     upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
#     lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Upper Stack (binned)</b><br>{upper_text}</div>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Lower Stack (binned)</b><br>{lower_text}</div>",
#         unsafe_allow_html=True,
#     )

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")


# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import numpy as np
# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     """
#     UI polish:
#     - Keeps near-commercial look
#     - Recolors UI icons to accent (yellow) WITHOUT tinting matplotlib charts
#     - Upgraded tabs (Option A):
#         * smooth sliding underline animation
#         * active tab slightly raised (soft depth)
#         * subtle fade transition when switching tabs
#         * tab container aligned with your topbar card width
#     """
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# /* ============================================================================
#    Icons: yellow (UI-only)
#    - SVG icons only (does NOT recolor button text)
#    ============================================================================ */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# [data-testid="stSidebar"] [data-testid*="icon"],
# [data-testid="stToolbar"] [data-testid*="icon"],
# [data-testid="stHeader"] [data-testid*="icon"],
# [data-testid="stExpander"] [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* ============================================================================
#    Sidebar
#    ============================================================================ */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* ============================================================================
#    Topbar + cards
#    ============================================================================ */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{
#   display:flex;
#   flex-direction:column;
#   gap:2px;
# }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{
#   font-size: 0.88rem;
#   color: var(--muted);
# }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width:9px; height:9px; border-radius:50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# /* ============================================================================
#    Upload / DataFrame / Buttons
#    ============================================================================ */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# /* ============================================================================
#    Status banner
#    ============================================================================ */
# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* ============================================================================
#    Tabs — Option A (Upgraded)
#    ============================================================================ */

# /* Align tabs container with your topbar card width */
# div[data-testid="stTabs"] {{
#   width: 100%;
# }}
# div[data-testid="stTabs"] > div {{
#   padding-left: 0 !important;
#   padding-right: 0 !important;
# }}

# /* Pill container */
# div[data-baseweb="tab-list"] {{
#   position: relative !important;
#   display: flex !important;
#   width: 100% !important;
#   justify-content: space-between !important;
#   gap: 0 !important;

#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
#   box-shadow: none !important;
# }}

# /* Tabs (equal width) */
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;

#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 850 !important;
#   font-size: 0.96rem !important;

#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
#   border: 1px solid transparent !important;

#   transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease !important;
#   margin: 0 0.16rem !important;
#   position: relative !important;
# }}

# button[data-baseweb="tab"]:hover {{
#   background: rgba(31,42,68,0.06) !important;
# }}

# /* Active = raised + soft depth */
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;

#   transform: translateY(-1px) !important;
#   box-shadow:
#     0 10px 18px rgba(17,24,39,0.06),
#     0 4px 10px rgba(201,151,0,0.10) !important;
# }}

# /* Sliding underline (BaseWeb highlight element moves; style it) */
# div[data-baseweb="tab-highlight"] {{
#   position: absolute !important;
#   left: 0 !important;
#   bottom: 6px !important;

#   height: 3px !important;
#   border-radius: 999px !important;

#   background: var(--accent) !important;
#   box-shadow: 0 8px 18px rgba(201,151,0,0.22) !important;

#   transition: transform 280ms cubic-bezier(.2,.9,.2,1), width 280ms cubic-bezier(.2,.9,.2,1) !important;
# }}

# div[data-baseweb="tab-border"] {{
#   display: none !important;
# }}

# /* Fade transition on content panel */
# div[data-testid="stTabs"] [role="tabpanel"] {{
#   animation: tabFade 200ms ease-in-out;
# }}
# @keyframes tabFade {{
#   from {{ opacity: 0.0; transform: translateY(3px); }}
#   to   {{ opacity: 1.0; transform: translateY(0px); }}
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "analysis_ready": False,   # logs parsed and complete
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers (performance + robustness)
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     # Cache by (text, name) tuples (hashable), not UploadedFile objects.
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#             "analysis_ready",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     dfP_upper = st.session_state["dfP_upper"]
#     dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Upload • Analysis • Reports • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def get_binned_gantry_angles(df: pd.DataFrame, bin_deg: int = 10) -> list[int]:
#     """
#     Returns sorted unique gantry bin start angles (deg).
#     Example: [0, 270]
#     """
#     if df is None or df.empty:
#         return []

#     gantry_col = None
#     for c in ["GantryAngle", "Gantry Angle", "Gantry", "Gantry (deg)", "gantry_angle"]:
#         if c in df.columns:
#             gantry_col = c
#             break
#     if gantry_col is None:
#         return []

#     g = pd.to_numeric(df[gantry_col], errors="coerce").dropna()
#     if g.empty:
#         return []

#     g = (g % 360.0 + 360.0) % 360.0
#     bins = np.floor(g / float(bin_deg)) * float(bin_deg)
#     return sorted({int(b) for b in bins})


# # =============================================================================
# # Sidebar
# # =============================================================================
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_rms = st.sidebar.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
# rep_fail_rms = st.sidebar.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Matching options")
# st.session_state["drop_stack_by_plan_name"] = st.sidebar.checkbox(
#     "Auto-select stacks by plan naming convention (recommended)",
#     value=st.session_state.get("drop_stack_by_plan_name", True),
# )
# st.sidebar.caption("Plan/log merge runs in Analysis tab.")
# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")

# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["Upload & Intake", "Analysis", "Reports & Export", "Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE (AUTO-PARSE ON UPLOAD)
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         clear_btn = st.button("Clear session data", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.session_state["analysis_ready"] = False
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Parsing delivery logs…"):
#                 df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#             st.session_state["system_status"] = "ready"

#             if df_errors is not None and len(df_errors) > 0:
#                 st.warning("Some files could not be parsed.")
#                 with st.expander("Parsing details"):
#                     st.dataframe(df_errors, use_container_width=True)

#             n_upper = 0 if df_upper is None else int(len(df_upper))
#             n_lower = 0 if df_lower is None else int(len(df_lower))

#             m1, m2, m3 = st.columns(3)
#             m1.metric("Uploaded files", len(uploaded))
#             m2.metric("Upper stack records", n_upper)
#             m3.metric("Lower stack records", n_lower)

#             if n_upper == 0 or n_lower == 0:
#                 st.session_state["upload_complete"] = False
#                 st.session_state["analysis_ready"] = False
#                 st.session_state["system_status"] = "error"
#                 missing = "Upper" if n_upper == 0 else "Lower"
#                 present = "Lower" if n_upper == 0 else "Upper"
#                 st.error(
#                     f"**Incomplete upload — {missing} stack missing.** "
#                     f"Full QA requires at least one {present} and one {missing} stack log."
#                 )
#             else:
#                 st.session_state["upload_complete"] = True
#                 st.session_state["analysis_ready"] = True
#                 st.success("Delivery logs parsed successfully. Proceed to **Analysis** and click **Analyze**.")

#             with st.expander("Data preview (verification)"):
#                 st.write("All records")
#                 st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     st.write("Upper stack")
#                     st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                 with c2:
#                     st.write("Lower stack")
#                     st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown(
#             "**Plan Reference (Local)**  \n"
#             "Expected: `data/dfP_all.pkl`, `data/dfP_upper.pkl`, `data/dfP_lower.pkl`"
#         )

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Intake Notes**")
#         st.caption("• Upload both upper & lower stack logs (auto-parsed).")
#         st.caption("• Click **Analyze** in the Analysis tab to run matching + QA.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS (BUTTON = ANALYZE)
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Run plan/log matching and compute MLC positional accuracy metrics.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     # Controls row: Analyze + optional reset merge
#     cA, cB, cC = st.columns([0.28, 0.36, 0.36], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         analyze_btn = st.button("Analyze", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         reset_btn = st.button("Reset matching cache", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cC:
#         st.caption("Click **Analyze** after upload. Use reset if plan PKLs or matching option changed.")

#     if reset_btn:
#         for k in ("mU", "mL", "analysis_out"):
#             st.session_state.pop(k, None)
#         st.toast("Matching cache cleared.", icon="🔁")

#     # Only run analysis when user clicks Analyze (prevents rerun on every widget change)
#     if not analyze_btn and "analysis_out" not in st.session_state:
#         st.info("Ready. Click **Analyze** to run matching + QA.")
#         st.stop()

#     # Run matching
#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     # Compute errors + store
#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     # --- Gantry angles (binned) like your screenshot ---
#     st.markdown('<div class="section-title">Gantry Angles (Binned)</div>', unsafe_allow_html=True)
#     gantry_bin_deg = st.slider("Gantry bin size (deg)", min_value=1, max_value=30, value=10, step=1)

#     upper_bins = get_binned_gantry_angles(mUe, bin_deg=int(gantry_bin_deg))
#     lower_bins = get_binned_gantry_angles(mLe, bin_deg=int(gantry_bin_deg))

#     upper_text = ", ".join(str(b) for b in upper_bins) if upper_bins else "None detected"
#     lower_text = ", ".join(str(b) for b in lower_bins) if lower_bins else "None detected"

#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Upper Stack (binned)</b><br>{upper_text}</div>",
#         unsafe_allow_html=True,
#     )
#     st.markdown(
#         f"<div class='card'><b>Gantry angles (°) — Lower Stack (binned)</b><br>{lower_text}</div>",
#         unsafe_allow_html=True,
#     )

#     # QA status
#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")


# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     """
#     UI polish:
#     - Keeps near-commercial look
#     - Recolors UI icons to accent (yellow) WITHOUT tinting matplotlib charts
#     - Upgraded tabs (Option A):
#         * smooth sliding underline animation
#         * active tab slightly raised (soft depth)
#         * subtle fade transition when switching tabs
#         * tab container aligned with your topbar card width
#     """
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# /* ============================================================================
#    Icons: yellow (UI-only)
#    Scope to UI containers so charts don't get recolored.
#    ============================================================================ */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}

# /* Some icons follow currentColor */
# button, [role="button"], i, [class*="icon"], [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* ============================================================================
#    Sidebar
#    ============================================================================ */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* ============================================================================
#    Topbar + cards
#    ============================================================================ */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{
#   display:flex;
#   flex-direction:column;
#   gap:2px;
# }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{
#   font-size: 0.88rem;
#   color: var(--muted);
# }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width:9px; height:9px; border-radius:50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}

# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# /* ============================================================================
#    Upload / DataFrame / Buttons
#    ============================================================================ */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# /* ============================================================================
#    Status banner
#    ============================================================================ */
# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* ============================================================================
#    Tabs — Option A (Upgraded)
#    - smooth sliding underline animation
#    - active tab slightly raised
#    - subtle fade on tab content
#    - aligned to topbar width
#    ============================================================================ */

# /* Align tabs container with your topbar card width */
# div[data-testid="stTabs"] {{
#   width: 100%;
# }}
# div[data-testid="stTabs"] > div {{
#   padding-left: 0 !important;
#   padding-right: 0 !important;
# }}

# /* Pill container */
# div[data-baseweb="tab-list"] {{
#   position: relative !important;
#   display: flex !important;
#   width: 100% !important;
#   justify-content: space-between !important;
#   gap: 0 !important;

#   padding: 0.30rem !important;
#   background: rgba(17,24,39,0.03) !important;
#   border: 1px solid var(--border) !important;
#   border-radius: 18px !important;
#   overflow: hidden !important;
#   box-shadow: none !important;
# }}

# /* Tabs (equal width) */
# button[data-baseweb="tab"] {{
#   flex: 1 !important;
#   justify-content: center !important;

#   border-radius: 16px !important;
#   padding: 0.62rem 1.10rem !important;
#   font-weight: 850 !important;
#   font-size: 0.96rem !important;

#   color: rgba(31,42,68,0.78) !important;
#   background: transparent !important;
#   border: 1px solid transparent !important;

#   transition: transform 180ms ease, box-shadow 180ms ease, background 180ms ease, color 180ms ease !important;
#   margin: 0 0.16rem !important;
#   position: relative !important;
# }}

# button[data-baseweb="tab"]:hover {{
#   background: rgba(31,42,68,0.06) !important;
# }}

# /* Active = raised + soft depth */
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: rgba(201,151,0,0.12) !important;
#   color: var(--primary) !important;
#   border: 1px solid rgba(201,151,0,0.25) !important;

#   transform: translateY(-1px) !important;
#   box-shadow:
#     0 10px 18px rgba(17,24,39,0.06),
#     0 4px 10px rgba(201,151,0,0.10) !important;
# }}

# /* Sliding underline (BaseWeb highlight element moves; style it) */
# div[data-baseweb="tab-highlight"] {{
#   position: absolute !important;
#   left: 0 !important;
#   bottom: 6px !important;

#   height: 3px !important;
#   border-radius: 999px !important;

#   background: var(--accent) !important;
#   box-shadow: 0 8px 18px rgba(201,151,0,0.22) !important;

#   transition: transform 280ms cubic-bezier(.2,.9,.2,1), width 280ms cubic-bezier(.2,.9,.2,1) !important;
# }}

# div[data-baseweb="tab-border"] {{
#   display: none !important;
# }}

# /* Fade transition on content panel */
# div[data-testid="stTabs"] [role="tabpanel"] {{
#   animation: tabFade 200ms ease-in-out;
# }}
# @keyframes tabFade {{
#   from {{ opacity: 0.0; transform: translateY(3px); }}
#   to   {{ opacity: 1.0; transform: translateY(0px); }}
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()

# # =============================================================================
# # Helpers (performance + robustness)
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     # Cache by (text, name) tuples (hashable), not UploadedFile objects.
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     dfP_upper = st.session_state["dfP_upper"]
#     dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {"success": "System Ready", "warn": "Attention", "danger": "Action Required"}[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Mechanical QA • Delivery Analytics • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# # =============================================================================
# # Sidebar
# # =============================================================================
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_rms = st.sidebar.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
# rep_fail_rms = st.sidebar.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Matching options")
# st.session_state["drop_stack_by_plan_name"] = st.sidebar.checkbox(
#     "Auto-select stacks by plan naming convention (recommended)",
#     value=st.session_state.get("drop_stack_by_plan_name", True),
# )
# st.sidebar.caption("Plan/log merge runs in Analysis tab.")
# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")

# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["Upload & Intake", "Mechanical QA", "Reports & Export", "Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         btnA, btnB = st.columns([0.38, 0.62], gap="small")
#         with btnA:
#             st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#             parse_btn = st.button("Parse logs", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with btnB:
#             st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#             clear_btn = st.button("Clear session data", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             if parse_btn or "df_all" in st.session_state:
#                 st.session_state["system_status"] = "parsing"
#                 with st.spinner("Parsing delivery logs…"):
#                     df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#                 st.session_state["system_status"] = "ready"

#                 if df_errors is not None and len(df_errors) > 0:
#                     st.warning("Some files could not be parsed.")
#                     with st.expander("Parsing details"):
#                         st.dataframe(df_errors, use_container_width=True)

#                 n_upper = 0 if df_upper is None else int(len(df_upper))
#                 n_lower = 0 if df_lower is None else int(len(df_lower))

#                 m1, m2, m3 = st.columns(3)
#                 m1.metric("Uploaded files", len(uploaded))
#                 m2.metric("Upper stack records", n_upper)
#                 m3.metric("Lower stack records", n_lower)

#                 if n_upper == 0 or n_lower == 0:
#                     st.session_state["upload_complete"] = False
#                     st.session_state["system_status"] = "error"
#                     missing = "Upper" if n_upper == 0 else "Lower"
#                     present = "Lower" if n_upper == 0 else "Upper"
#                     st.error(
#                         f"**Incomplete upload — {missing} stack missing.** "
#                         f"Full QA requires at least one {present} and one {missing} stack log."
#                     )
#                 else:
#                     st.session_state["upload_complete"] = True
#                     st.success("Delivery logs parsed successfully.")

#                 with st.expander("Data preview (verification)"):
#                     st.write("All records")
#                     st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                     c1, c2 = st.columns(2)
#                     with c1:
#                         st.write("Upper stack")
#                         st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                     with c2:
#                         st.write("Lower stack")
#                         st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown(
#             "**Plan Reference (Local)**  \n"
#             "Expected: `data/dfP_all.pkl`, `data/dfP_upper.pkl`, `data/dfP_lower.pkl`"
#         )

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Intake Notes**")
#         st.caption("• Upload both upper & lower stack logs.")
#         st.caption("• Matching runs in the Mechanical QA tab.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Mechanical QA (Log-Based)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Plan/log matching and MLC positional accuracy evaluation.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB = st.columns([0.34, 0.66], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         rerun_merge = st.button("Re-run plan/log matching", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.caption("Use if plan reference files or matching options changed.")

#     if rerun_merge:
#         st.session_state.pop("mU", None)
#         st.session_state.pop("mL", None)
#         st.toast("Will re-run matching.", icon="🔁")

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")


# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # =============================================================================
# # App identity
# # =============================================================================
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"

# st.set_page_config(
#     page_title=APP_SHORT_NAME,
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # =============================================================================
# # Theme + CSS
# # =============================================================================
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     """
#     - Keeps your original near-commercial look.
#     - Adds "UI icons only" yellow recolor (won't tint matplotlib charts).
#     - Adds a cleaner global tab style (segmented look).
#     """
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
#   --icon: var(--accent);
# }}

# .stApp {{ background: var(--bg); }}

# header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# /* =========================
#    Icons: yellow (UI-only)
#    Avoid recoloring charts by scoping to UI containers/buttons
#    ========================= */
# button svg, button svg *,
# [data-testid="stSidebar"] svg, [data-testid="stSidebar"] svg *,
# [data-testid="stExpander"] svg, [data-testid="stExpander"] svg *,
# [data-testid="stToolbar"] svg, [data-testid="stToolbar"] svg *,
# [data-testid="stHeader"] svg, [data-testid="stHeader"] svg * {{
#   fill: var(--icon) !important;
#   stroke: var(--icon) !important;
# }}
# /* Some icons follow currentColor */
# button, [role="button"], i, [class*="icon"], [data-testid*="icon"] {{
#   color: var(--icon) !important;
# }}

# /* =========================
#    Sidebar
#    ========================= */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# /* =========================
#    Topbar + cards
#    ========================= */
# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{
#   display:flex;
#   flex-direction:column;
#   gap:2px;
# }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{
#   font-size: 0.88rem;
#   color: var(--muted);
# }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width:9px; height:9px; border-radius:50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}
# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# /* =========================
#    Upload / DataFrame / Buttons
#    ========================= */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{ filter: brightness(1.06); }}
# .ghost-btn button {{ background: rgba(31,42,68,0.03) !important; }}

# /* =========================
#    Status banner
#    ========================= */
# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{ color: var(--muted); }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# /* =========================
#    Tabs: global segmented look
#    ========================= */
# div[data-baseweb="tab-list"] {{
#   gap: 0 !important;
#   padding: 0.22rem !important;
#   background: rgba(17,24,39,0.03);
#   border: 1px solid var(--border);
#   border-radius: 14px;
#   width: fit-content;
#   overflow: hidden;
# }}
# button[data-baseweb="tab"] {{
#   border-radius: 12px !important;
#   padding: 0.48rem 0.92rem !important;
#   font-weight: 800 !important;
#   font-size: 0.94rem !important;
#   color: rgba(17,24,39,0.72) !important;
#   background: transparent !important;
#   border: 1px solid transparent !important;
#   transition: all 120ms ease-in-out;
#   margin: 0 0.12rem;
# }}
# button[data-baseweb="tab"]:hover {{
#   background: rgba(31,42,68,0.06) !important;
# }}
# button[data-baseweb="tab"][aria-selected="true"] {{
#   background: white !important;
#   color: var(--text) !important;
#   border: 1px solid var(--border) !important;
#   box-shadow: 0 10px 18px rgba(17,24,39,0.06);
# }}
# div[data-baseweb="tab-highlight"],
# div[data-baseweb="tab-border"] {{
#   display: none !important;
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)

# # =============================================================================
# # Constants
# # =============================================================================
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"
# TREND_DIR.mkdir(parents=True, exist_ok=True)

# # =============================================================================
# # State management
# # =============================================================================
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         st.session_state.setdefault(k, v)


# ensure_state()


# # =============================================================================
# # Helpers (performance + robustness)
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     # Cache by (text, name) tuples (hashable), not UploadedFile objects.
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower

#     dfP_upper = st.session_state["dfP_upper"]
#     dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {
#         "success": "System Ready",
#         "warn": "Attention",
#         "danger": "Action Required",
#     }[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Mechanical QA • Delivery Analytics • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# # =============================================================================
# # Sidebar
# # =============================================================================
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_rms = st.sidebar.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
# rep_fail_rms = st.sidebar.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Matching options")
# st.session_state["drop_stack_by_plan_name"] = st.sidebar.checkbox(
#     "Auto-select stacks by plan naming convention (recommended)",
#     value=st.session_state.get("drop_stack_by_plan_name", True),
# )
# st.sidebar.caption("Plan/log merge runs in Analysis tab.")
# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")


# # =============================================================================
# # Header
# # =============================================================================
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["Upload & Intake", "Mechanical QA", "Reports & Export", "Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         btnA, btnB = st.columns([0.38, 0.62], gap="small")
#         with btnA:
#             st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#             parse_btn = st.button("Parse logs", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with btnB:
#             st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#             clear_btn = st.button("Clear session data", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             if parse_btn or "df_all" in st.session_state:
#                 st.session_state["system_status"] = "parsing"
#                 with st.spinner("Parsing delivery logs…"):
#                     df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#                 st.session_state["system_status"] = "ready"

#                 if df_errors is not None and len(df_errors) > 0:
#                     st.warning("Some files could not be parsed.")
#                     with st.expander("Parsing details"):
#                         st.dataframe(df_errors, use_container_width=True)

#                 n_upper = 0 if df_upper is None else int(len(df_upper))
#                 n_lower = 0 if df_lower is None else int(len(df_lower))

#                 m1, m2, m3 = st.columns(3)
#                 m1.metric("Uploaded files", len(uploaded))
#                 m2.metric("Upper stack records", n_upper)
#                 m3.metric("Lower stack records", n_lower)

#                 if n_upper == 0 or n_lower == 0:
#                     st.session_state["upload_complete"] = False
#                     st.session_state["system_status"] = "error"
#                     missing = "Upper" if n_upper == 0 else "Lower"
#                     present = "Lower" if n_upper == 0 else "Upper"
#                     st.error(
#                         f"**Incomplete upload — {missing} stack missing.** "
#                         f"Full QA requires at least one {present} and one {missing} stack log."
#                     )
#                 else:
#                     st.session_state["upload_complete"] = True
#                     st.success("Delivery logs parsed successfully.")

#                 with st.expander("Data preview (verification)"):
#                     st.write("All records")
#                     st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                     c1, c2 = st.columns(2)
#                     with c1:
#                         st.write("Upper stack")
#                         st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                     with c2:
#                         st.write("Lower stack")
#                         st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown(
#             "**Plan Reference (Local)**  \n"
#             "Expected: `data/dfP_all.pkl`, `data/dfP_upper.pkl`, `data/dfP_lower.pkl`"
#         )

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             if st.session_state.get("system_status") == "missing_plan":
#                 st.session_state["system_status"] = "ready"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Intake Notes**")
#         st.caption("• Upload both upper & lower stack logs.")
#         st.caption("• Matching runs in the Mechanical QA tab.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Mechanical QA (Log-Based)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Plan/log matching and MLC positional accuracy evaluation.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB = st.columns([0.34, 0.66], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         rerun_merge = st.button("Re-run plan/log matching", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.caption("Use if plan reference files or matching options changed.")

#     if rerun_merge:
#         st.session_state.pop("mU", None)
#         st.session_state.pop("mL", None)
#         st.toast("Will re-run matching.", icon="🔁")

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = TREND_DIR / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")
#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared")

# # app.py — MRIdian MLC QA Suite (Near-Commercial Clinical UI)
# # streamlit run app.py

# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Tuple
# import hashlib

# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # -----------------------------------------------------------------------------
# # App identity
# # -----------------------------------------------------------------------------
# APP_VERSION = "1.0.0"
# APP_NAME = "MRIdian Log-Based QA Suite"
# APP_SHORT_NAME = "MRIdian MLC QA"
# PAGE_TITLE = APP_SHORT_NAME

# st.set_page_config(page_title=PAGE_TITLE, page_icon="🧪", layout="wide", initial_sidebar_state="expanded")


# # -----------------------------------------------------------------------------
# # Visual theme (near-commercial)
# # -----------------------------------------------------------------------------
# THEME = {
#     "primary": "#1f2a44",   # deep navy
#     "accent": "#C99700",    # gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# def inject_css(t: dict) -> None:
#     st.markdown(
#         f"""
# <style>
# :root {{
#   --primary: {t["primary"]};
#   --accent: {t["accent"]};
#   --bg: {t["bg"]};
#   --panel: {t["panel"]};
#   --border: {t["border"]};
#   --text: {t["text"]};
#   --muted: {t["muted"]};
#   --success: {t["success"]};
#   --warn: {t["warn"]};
#   --danger: {t["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
# }}

# .stApp {{ background: var(--bg); }}

# header[data-testid="stHeader"] {{ visibility: hidden; height: 0px; }}
# footer {{ visibility: hidden; }}

# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.0rem !important;
#   max-width: 1280px;
# }}

# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stNumberInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}
# section[data-testid="stSidebar"] .stMarkdown small,
# section[data-testid="stSidebar"] .stCaptionContainer {{
#   color: rgba(255,255,255,0.70) !important;
# }}

# .topbar {{
#   display: flex;
#   align-items: center;
#   justify-content: space-between;
#   gap: 16px;
#   padding: 14px 16px;
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
# }}
# .brand {{
#   display:flex;
#   flex-direction:column;
#   gap:2px;
# }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 800;
#   color: var(--text);
# }}
# .brand-sub {{
#   font-size: 0.88rem;
#   color: var(--muted);
# }}
# .topbar-right {{
#   display:flex;
#   align-items:center;
#   gap:10px;
#   flex-wrap:wrap;
#   justify-content:flex-end;
# }}
# .badge {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:6px 10px;
#   border:1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size:0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width:9px;height:9px;border-radius:50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}
# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace;
#   font-size:0.82rem;
#   padding:2px 6px;
#   border:1px solid var(--border);
#   border-radius:8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px;
# }}
# .section-title {{
#   margin: 18px 0 6px 0;
#   font-weight: 850;
#   font-size: 1.05rem;
#   color: var(--text);
# }}
# .section-sub {{
#   margin: 0 0 12px 0;
#   color: var(--muted);
# }}

# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}

# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{
#   filter: brightness(1.06);
# }}
# .ghost-btn button {{
#   background: rgba(31,42,68,0.03) !important;
# }}

# .status-banner {{
#   padding: 0.85rem 1rem;
#   border-radius: 0.85rem;
#   border: 1px solid var(--border);
#   margin: 0.5rem 0 0.75rem 0;
#   background: var(--panel);
#   box-shadow: var(--shadow);
# }}
# .status-title {{
#   font-weight: 850;
#   font-size: 1.02rem;
#   margin-bottom: 0.15rem;
# }}
# .status-sub {{
#   color: var(--muted);
# }}
# .status-chip {{
#   display:inline-flex;
#   align-items:center;
#   gap:8px;
#   border-radius:999px;
#   padding:5px 10px;
#   border:1px solid var(--border);
#   font-size:0.82rem;
#   margin-left:8px;
# }}
# .status-chip.pass {{ background: rgba(15,118,110,0.10); color: var(--success); border-color: rgba(15,118,110,0.25); }}
# .status-chip.warn {{ background: rgba(180,83,9,0.10); color: var(--warn); border-color: rgba(180,83,9,0.25); }}
# .status-chip.fail {{ background: rgba(185,28,28,0.10); color: var(--danger); border-color: rgba(185,28,28,0.25); }}

# div[data-testid="stDataFrame"] {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   overflow: hidden;
#   box-shadow: var(--shadow);
# }}
# </style>
# """,
#         unsafe_allow_html=True,
#     )


# inject_css(THEME)


# # -----------------------------------------------------------------------------
# # Session state defaults
# # -----------------------------------------------------------------------------
# def ensure_state() -> None:
#     defaults = {
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "upload_complete": False,
#         "drop_stack_by_plan_name": True,
#         "last_upload_signature": None,
#         "last_parsed_sig": None,
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# ensure_state()


# # -----------------------------------------------------------------------------
# # Helpers (performance + robustness)
# # -----------------------------------------------------------------------------
# PLAN_FOLDER = Path("data")
# OUTPUTS_DIR = Path("outputs")
# TREND_DIR = Path("data") / "trending"


# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     # Cache by (text, name) tuples (hashable), not UploadedFile objects.
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     sig = _uploaded_signature(files)
#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors
#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files) -> None:
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         # clear derived state
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged() -> None:
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower
#     else:
#         dfP_upper = st.session_state["dfP_upper"]
#         dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _upload_is_complete() -> bool:
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# def _status_banner(scope_name: str, status: str, max_abs_mm: float) -> None:
#     s = (status or "").strip().upper()
#     if s == "PASS":
#         chip = '<span class="status-chip pass">PASS</span>'
#         title = f"{scope_name}: Within tolerance"
#     elif s in ("WARN", "WARNING"):
#         chip = '<span class="status-chip warn">WARN</span>'
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         chip = '<span class="status-chip fail">FAIL</span>'
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner">
#   <div class="status-title">{title}{chip}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{float(max_abs_mm):.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _status_dot_class(system_status: str) -> str:
#     s = (system_status or "").lower()
#     if s == "ready":
#         return "success"
#     if s in ("parsing", "missing_plan"):
#         return "warn"
#     return "danger"


# def render_topbar(site: str, machine: str, reviewer: str) -> None:
#     ctx = " • ".join([x for x in [site, machine, reviewer] if str(x).strip()]) or "No context set"
#     badge_class = _status_dot_class(st.session_state.get("system_status", "ready"))
#     badge_text = {
#         "success": "System Ready",
#         "warn": "Attention",
#         "danger": "Action Required",
#     }[badge_class]

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">{APP_NAME} <span class="kbd">v{APP_VERSION}</span></div>
#     <div class="brand-sub">Mechanical QA • Delivery Analytics • Trend Tracking</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{ctx}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# # -----------------------------------------------------------------------------
# # Sidebar (clinical-oriented)
# # -----------------------------------------------------------------------------
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log–driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_rms = st.sidebar.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
# rep_fail_rms = st.sidebar.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Matching options")
# st.session_state["drop_stack_by_plan_name"] = st.sidebar.checkbox(
#     "Auto-select stacks by plan naming convention (recommended)",
#     value=st.session_state.get("drop_stack_by_plan_name", True),
# )
# st.sidebar.caption("Plan/log merge runs in Analysis tab.")

# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION} • Research & QA use only • Not FDA cleared")


# # -----------------------------------------------------------------------------
# # Topbar + intro
# # -----------------------------------------------------------------------------
# render_topbar(site_name, machine_name, reviewer_name)
# st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="section-sub">Automated analysis of MLC delivery accuracy from MRIdian log records.</div>',
#     unsafe_allow_html=True,
# )

# # -----------------------------------------------------------------------------
# # Tabs
# # -----------------------------------------------------------------------------
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(
#     ["Upload & Intake", "Mechanical QA", "Reports & Export", "Longitudinal Trends"]
# )

# # =============================================================================
# # TAB 1: UPLOAD & INTAKE
# # =============================================================================
# with tab_upload:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Upload MRIdian delivery logs and confirm plan reference availability.</div>',
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns([1.12, 0.88], gap="large")

#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Delivery Log Files (.txt)**  \nSupported: ViewRay MRIdian • Upload both upper & lower stacks")
#         uploaded = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             key="log_uploader",
#             label_visibility="collapsed",
#         )
#         st.markdown("</div>", unsafe_allow_html=True)

#         _reset_results_on_new_upload(uploaded)

#         btnA, btnB = st.columns([0.38, 0.62], gap="small")
#         with btnA:
#             st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#             parse_btn = st.button("Parse logs", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with btnB:
#             st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#             clear_btn = st.button("Clear session data", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#         if clear_btn:
#             for k in list(st.session_state.keys()):
#                 if k not in ("site_name", "machine_name", "reviewer_name", "drop_stack_by_plan_name"):
#                     st.session_state.pop(k, None)
#             _parse_uploaded_texts_cached.clear()
#             st.toast("Session cleared.", icon="🧹")

#         if not uploaded:
#             st.session_state["system_status"] = "ready"
#             st.info("No delivery logs detected. Upload at least one MRIdian delivery log file to continue.")
#         else:
#             if parse_btn or "df_all" in st.session_state:
#                 st.session_state["system_status"] = "parsing"
#                 with st.spinner("Parsing delivery logs…"):
#                     df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)
#                 st.session_state["system_status"] = "ready"

#                 if df_errors is not None and len(df_errors) > 0:
#                     st.warning("Some files could not be parsed.")
#                     with st.expander("Parsing details"):
#                         st.dataframe(df_errors, use_container_width=True)

#                 n_upper = 0 if df_upper is None else int(len(df_upper))
#                 n_lower = 0 if df_lower is None else int(len(df_lower))

#                 m1, m2, m3 = st.columns(3)
#                 m1.metric("Uploaded files", len(uploaded))
#                 m2.metric("Upper stack records", n_upper)
#                 m3.metric("Lower stack records", n_lower)

#                 if n_upper == 0 or n_lower == 0:
#                     st.session_state["upload_complete"] = False
#                     st.session_state["system_status"] = "error"
#                     missing = "Upper" if n_upper == 0 else "Lower"
#                     present = "Lower" if n_upper == 0 else "Upper"
#                     st.error(
#                         f"**Incomplete upload — {missing} stack missing.** "
#                         f"Full QA requires at least one {present} and one {missing} stack log."
#                     )
#                 else:
#                     st.session_state["upload_complete"] = True
#                     st.success("Delivery logs parsed successfully.")

#                 with st.expander("Data preview (verification)"):
#                     st.write("All records")
#                     st.dataframe(df_all.head(preview_rows), use_container_width=True)
#                     c1, c2 = st.columns(2)
#                     with c1:
#                         st.write("Upper stack")
#                         st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#                     with c2:
#                         st.write("Lower stack")
#                         st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Plan Reference (Local)**  \nExpected: `data/dfP_all.pkl`, `data/dfP_upper.pkl`, `data/dfP_lower.pkl`")

#         if _plans_available(PLAN_FOLDER):
#             st.success("Plan reference detected in `./data` ✅")
#             st.session_state["system_status"] = "ready" if st.session_state["system_status"] != "error" else "error"
#         else:
#             st.session_state["system_status"] = "missing_plan"
#             st.error("Plan reference files not found in `./data` ❌")
#             st.caption("Place the three PKLs in `./data` to enable matching.")

#         st.markdown("---")
#         st.markdown("**Intake Notes**")
#         st.caption("• Upload both upper & lower stack logs.")
#         st.caption("• Matching runs in the Mechanical QA tab.")
#         st.caption("• Reports require completed analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)


# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.markdown('<div class="section-title">Mechanical QA (Log-Based)</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Plan/log matching and MLC positional accuracy evaluation.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable analysis.")
#         st.stop()

#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in `./data`. Add PKLs then re-run.")
#         st.stop()

#     cA, cB = st.columns([0.34, 0.66], gap="small")
#     with cA:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         rerun_merge = st.button("Re-run plan/log matching", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with cB:
#         st.caption("Use if plan reference files or matching options changed.")

#     if rerun_merge:
#         st.session_state.pop("mU", None)
#         st.session_state.pop("mL", None)
#         st.toast("Will re-run matching.", icon="🔁")

#     try:
#         st.session_state["system_status"] = "parsing"
#         with st.spinner("Matching plan to delivery logs…"):
#             _ensure_merged()
#         st.session_state["system_status"] = "ready"
#         st.success(
#             f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}"
#         )
#     except Exception as e:
#         st.session_state["system_status"] = "error"
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     # Compute errors
#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     # Per-leaf metrics + status
#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")
#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.markdown('<div class="section-title">QA Status</div>', unsafe_allow_html=True)
#     c1, c2 = st.columns(2, gap="large")
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.markdown('<div class="section-title">Virtual Picket Fence</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Delivered gap pattern derived from log records.</div>', unsafe_allow_html=True)

#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapU,
#             title="Upper stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(
#             picket_centers_mm=picket_centers,
#             gap_df=gapL,
#             title="Lower stack — delivered gap pattern",
#         ),
#         clear_figure=True,
#     )

#     st.markdown('<div class="section-title">Per-Leaf Summary</div>', unsafe_allow_html=True)
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot reference (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)


# # =============================================================================
# # TAB 3: REPORTS & EXPORT
# # =============================================================================
# with tab_reports:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Generate standardized PDF reports and optional trend logging.</div>', unsafe_allow_html=True)

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to generate report inputs.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Report metadata**")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")
#     st.markdown("</div>", unsafe_allow_html=True)

#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     left, right = st.columns([0.62, 0.38], gap="large")
#     with left:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Trend logging (optional)**")
#         update_trend = st.toggle("Append this run to trends", value=False)
#         trend_csv_path = None
#         if update_trend:
#             trend_csv_path = st.text_input(
#                 "Trends file path (server-side)",
#                 value=str(TREND_DIR / "trend_report_bank_summary.csv"),
#             )
#         st.markdown("---")
#         keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown('<div class="card">', unsafe_allow_html=True)
#         st.markdown("**Generate**")
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         gen_btn = st.button("Generate PDF report", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#         st.caption("Outputs a downloadable PDF. Optional server copy and trend append are supported.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     if gen_btn:
#         try:
#             st.session_state["system_status"] = "parsing"
#             with st.spinner("Generating PDF…"):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                     # logo_path=Path("assets/logo.png")  # optional
#                 )
#             st.session_state["system_status"] = "ready"
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.session_state["system_status"] = "error"
#             st.error(f"Report generation failed: {e}")

#     if st.session_state.get("pdf_bytes"):
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
#             out_path = OUTPUTS_DIR / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")


# # =============================================================================
# # TAB 4: LONGITUDINAL TRENDS
# # =============================================================================
# with tab_trends:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-sub">Track stability and drift of maximum absolute leaf errors over time.</div>',
#         unsafe_allow_html=True,
#     )

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload and parse both Upper and Lower stack logs in **Upload & Intake** to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Mechanical QA** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = Path("data") / "trending" / "trend_overall_max.csv"
#     trend_path.parent.mkdir(parents=True, exist_ok=True)

#     # --- Trend plot settings (CARD) ---
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Trend plot settings**")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for visualization only; does not change stored data.",
#         )
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

#     # --- Update trend history (CARD) ---
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.markdown("**Update trend history**")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")

#     st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#     append_btn = st.button("Append current run to trends", use_container_width=True)
#     st.markdown("</div>", unsafe_allow_html=True)

#     st.markdown("</div>", unsafe_allow_html=True)

#     if append_btn:
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     # Load history
#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Trend plot</div>', unsafe_allow_html=True)

#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # -----------------------------------------------------------------------------
# # Footer
# # -----------------------------------------------------------------------------
# st.markdown("---")
# st.caption(
#     f"{APP_NAME} • Version {APP_VERSION} • Research and QA use only • Not FDA cleared"
# )


# # app.py — MRIdian MLC QA Platform (Clinical UI)
# # streamlit run app.py

# from pathlib import Path
# from typing import Tuple
# import hashlib

# import pandas as pd
# import streamlit as st

# from core.io_log import extract_logs_texts
# from core.preprocess import preprocess_and_merge
# from core.report import generate_pdf_qa_report_bytes

# from core.analysis import (
#     add_errors,
#     error_describe,
#     leaf_metrics,
#     classify,
#     compute_gap_df_from_merged,
#     plot_virtual_picket_fence,
#     make_combined_leaf_index,
#     rms_by_leaf,
#     plot_rms_errors_by_leaf,
#     max_by_leaf,
#     plot_max_errors_by_leaf,
#     summarize_overall_max_error,
#     append_trending_csv,
#     plot_overall_max_trending,
# )

# # -----------------------------------------------------------------------------
# # App identity (clinical-facing)
# # -----------------------------------------------------------------------------
# APP_VERSION = "1.0.0"  # major.minor.patch
# APP_SHORT_NAME = "MRIdian MLC QA Platform"
# PAGE_TITLE = "MRIdian MLC QA"

# # IMPORTANT: must be the first Streamlit command
# st.set_page_config(page_title=PAGE_TITLE, layout="wide")

# # =============================================================================
# # Clinical UI polish (calm, compact, consistent)
# # =============================================================================
# st.markdown(
#     """
# <style>
# /* Layout */
# .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }

# /* Tabs */
# div[data-baseweb="tab-list"] button { font-weight: 650; }
# div[data-baseweb="tab-list"] { gap: 0.25rem; }

# /* Sidebar */
# section[data-testid="stSidebar"] { background: #fafafa; }
# section[data-testid="stSidebar"] .stMarkdown p { margin-bottom: 0.25rem; }

# /* Status banner */
# .status-banner {
#   padding: 0.85rem 1rem;
#   border-radius: 0.8rem;
#   border: 1px solid rgba(0,0,0,0.08);
#   margin: 0.5rem 0 1rem 0;
# }
# .status-pass { background: rgba(0, 128, 0, 0.08); }
# .status-warn { background: rgba(255, 165, 0, 0.10); }
# .status-fail { background: rgba(255, 0, 0, 0.08); }
# .status-title { font-weight: 750; font-size: 1.05rem; margin-bottom: 0.15rem; }
# .status-sub { color: rgba(0,0,0,0.70); }

# /* Section headers */
# h1, h2, h3 { letter-spacing: -0.01em; }
# </style>
# """,
#     unsafe_allow_html=True,
# )

# # =============================================================================
# # Helpers (performance + robustness)
# # =============================================================================
# def _plans_available(plan_folder: Path) -> bool:
#     return all((plan_folder / n).exists() for n in ("dfP_all.pkl", "dfP_upper.pkl", "dfP_lower.pkl"))


# @st.cache_data(show_spinner=False)
# def load_plan_data(folder: Path):
#     """Load plan PKLs from local folder -> (dfP_all, dfP_upper, dfP_lower)."""
#     dfP_all = pd.read_pickle(folder / "dfP_all.pkl")
#     dfP_upper = pd.read_pickle(folder / "dfP_upper.pkl")
#     dfP_lower = pd.read_pickle(folder / "dfP_lower.pkl")
#     return dfP_all, dfP_upper, dfP_lower


# def _uploaded_signature(files) -> Tuple[Tuple[str, str, int], ...]:
#     """Stable signature: (name, md5, size)."""
#     sig = []
#     for f in files:
#         b = f.getvalue()
#         sig.append((f.name, hashlib.md5(b).hexdigest(), len(b)))
#     return tuple(sorted(sig))


# @st.cache_data(show_spinner=False)
# def _parse_uploaded_texts_cached(texts_and_names: Tuple[Tuple[str, str], ...]):
#     """
#     Cache by (text, name) tuples (hashable), not UploadedFile objects.
#     Returns (df_all, df_upper, df_lower) or (df_all, df_upper, df_lower, df_errors).
#     """
#     return extract_logs_texts(list(texts_and_names))


# def parse_uploaded(files):
#     """
#     Parse uploaded .txt logs -> (df_all, df_upper, df_lower, df_errors)
#     Uses session_state + cached-by-text to avoid hangs.
#     """
#     sig = _uploaded_signature(files)

#     if st.session_state.get("last_parsed_sig") == sig and "df_all" in st.session_state:
#         return (
#             st.session_state["df_all"],
#             st.session_state["df_upper"],
#             st.session_state["df_lower"],
#             st.session_state.get("df_errors", pd.DataFrame(columns=["SourceFile", "Error"])),
#         )

#     texts_and_names = []
#     for f in files:
#         raw = f.getvalue()
#         text = raw.decode("utf-8", errors="ignore")
#         texts_and_names.append((text, f.name))

#     out = _parse_uploaded_texts_cached(tuple(texts_and_names))

#     if isinstance(out, tuple) and len(out) == 4:
#         df_all, df_upper, df_lower, df_errors = out
#     else:
#         df_all, df_upper, df_lower = out
#         df_errors = pd.DataFrame(columns=["SourceFile", "Error"])

#     st.session_state["last_parsed_sig"] = sig
#     st.session_state["df_all"] = df_all
#     st.session_state["df_upper"] = df_upper
#     st.session_state["df_lower"] = df_lower
#     st.session_state["df_errors"] = df_errors

#     return df_all, df_upper, df_lower, df_errors


# def _reset_results_on_new_upload(uploaded_files):
#     """Clear stale state if the uploaded set changes."""
#     if not uploaded_files:
#         return

#     current_sig = _uploaded_signature(uploaded_files)
#     last_sig = st.session_state.get("last_upload_signature")

#     if current_sig != last_sig:
#         for k in (
#             "df_all", "df_upper", "df_lower", "df_errors",
#             "dfP_all", "dfP_upper", "dfP_lower",
#             "mU", "mL",
#             "analysis_out",
#             "pdf_bytes", "pdf_name",
#             "trend_all",
#             "last_parsed_sig",
#             "upload_complete",
#         ):
#             st.session_state.pop(k, None)

#         _parse_uploaded_texts_cached.clear()
#         st.session_state["last_upload_signature"] = current_sig


# def _safe_first(df: pd.DataFrame, col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _ensure_merged():
#     """Ensure mU/mL exist by running preprocess_and_merge once when needed."""
#     if "mU" in st.session_state and "mL" in st.session_state:
#         return

#     if "df_upper" not in st.session_state or "df_lower" not in st.session_state:
#         raise RuntimeError("No parsed logs found. Please upload delivery log files first.")

#     PLAN_FOLDER = Path("data")
#     if not _plans_available(PLAN_FOLDER):
#         raise RuntimeError(
#             "Plan PKL files not found in ./data. "
#             "Place dfP_all.pkl / dfP_upper.pkl / dfP_lower.pkl in ./data."
#         )

#     if "dfP_upper" not in st.session_state or "dfP_lower" not in st.session_state:
#         dfP_all, dfP_upper, dfP_lower = load_plan_data(PLAN_FOLDER)
#         st.session_state["dfP_all"] = dfP_all
#         st.session_state["dfP_upper"] = dfP_upper
#         st.session_state["dfP_lower"] = dfP_lower
#     else:
#         dfP_upper = st.session_state["dfP_upper"]
#         dfP_lower = st.session_state["dfP_lower"]

#     out = preprocess_and_merge(
#         dfP_upper=dfP_upper,
#         dfP_lower=dfP_lower,
#         dfL_upper=st.session_state["df_upper"],
#         dfL_lower=st.session_state["df_lower"],
#         drop_stack_by_plan_name=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.session_state["mU"] = out["mU"]
#     st.session_state["mL"] = out["mL"]


# def _status_banner(scope_name: str, status: str, max_abs_mm: float):
#     status_norm = (status or "").strip().upper()
#     if status_norm == "PASS":
#         cls = "status-pass"
#         title = f"{scope_name}: Within tolerance"
#     elif status_norm in ("WARN", "WARNING"):
#         cls = "status-warn"
#         title = f"{scope_name}: Action level exceeded"
#     else:
#         cls = "status-fail"
#         title = f"{scope_name}: Out of tolerance"

#     st.markdown(
#         f"""
# <div class="status-banner {cls}">
#   <div class="status-title">{title}</div>
#   <div class="status-sub">Max absolute leaf position error: <b>{max_abs_mm:.3f} mm</b></div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def _upload_is_complete() -> bool:
#     """True only if we have both upper and lower records parsed."""
#     df_u = st.session_state.get("df_upper", None)
#     df_l = st.session_state.get("df_lower", None)
#     n_u = 0 if df_u is None else len(df_u)
#     n_l = 0 if df_l is None else len(df_l)
#     return (n_u > 0) and (n_l > 0)


# # =============================================================================
# # Sidebar (clinical-oriented)
# # =============================================================================
# st.sidebar.markdown(f"### {APP_SHORT_NAME}")
# st.sidebar.caption("Clinical workflow • Delivery-log driven • MLC positional QA")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Site & machine")

# site_name = st.sidebar.text_input("Institution / Site", value=st.session_state.get("site_name", ""))
# machine_name = st.sidebar.text_input("Machine name", value=st.session_state.get("machine_name", "MRIdian"))
# reviewer_name = st.sidebar.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

# st.session_state["site_name"] = site_name
# st.session_state["machine_name"] = machine_name
# st.session_state["reviewer_name"] = reviewer_name

# st.sidebar.markdown("---")
# st.sidebar.subheader("QA criteria (mm)")
# action_level_mm = st.sidebar.number_input("Action level", min_value=0.0, value=0.5, step=0.1)
# tolerance_level_mm = st.sidebar.number_input("Tolerance level", min_value=0.0, value=1.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Display")
# preview_rows = st.sidebar.slider("Preview rows", min_value=10, max_value=500, value=100, step=10)

# st.sidebar.markdown("---")
# st.sidebar.subheader("Report criteria (mm)")
# rep_warn_rms = st.sidebar.number_input("RMS action level", min_value=0.0, value=0.8, step=0.1)
# rep_fail_rms = st.sidebar.number_input("RMS tolerance level", min_value=0.0, value=1.2, step=0.1)
# rep_warn_max = st.sidebar.number_input("MaxAbs action level", min_value=0.0, value=1.5, step=0.1)
# rep_fail_max = st.sidebar.number_input("MaxAbs tolerance level", min_value=0.0, value=2.0, step=0.1)

# st.sidebar.markdown("---")
# st.sidebar.caption(f"Version {APP_VERSION}")

# # =============================================================================
# # Title / intro
# # =============================================================================
# st.title(APP_SHORT_NAME)
# st.write("Automated analysis of MLC delivery accuracy from treatment log records.")

# # =============================================================================
# # Tabs
# # =============================================================================
# tab_upload, tab_analysis, tab_reports, tab_trends = st.tabs(["Upload", "Analysis", "Reports", "Trends"])

# # =============================================================================
# # TAB 1: UPLOAD
# # =============================================================================
# with tab_upload:
#     st.header("Upload delivery logs")

#     st.subheader("Step 1 — Upload ViewRay delivery log file(s) (.txt)")
#     uploaded = st.file_uploader(
#         "Select one or more delivery log files",
#         type=["txt"],
#         accept_multiple_files=True,
#         key="log_uploader",
#     )

#     _reset_results_on_new_upload(uploaded)

#     if not uploaded:
#         st.info("Upload at least one delivery log file to continue.")
#     else:
#         with st.spinner("Parsing delivery logs..."):
#             df_all, df_upper, df_lower, df_errors = parse_uploaded(uploaded)

#         if df_errors is not None and len(df_errors) > 0:
#             st.warning("Some files could not be parsed.")
#             with st.expander("View parsing details"):
#                 st.dataframe(df_errors, use_container_width=True)

#         st.success("Delivery logs parsed successfully.")

#         n_upper = 0 if df_upper is None else int(len(df_upper))
#         n_lower = 0 if df_lower is None else int(len(df_lower))

#         cA, cB = st.columns(2)
#         with cA:
#             st.metric("Upper stack records", n_upper)
#         with cB:
#             st.metric("Lower stack records", n_lower)

#         if n_upper == 0 or n_lower == 0:
#             missing = "Upper" if n_upper == 0 else "Lower"
#             present = "Lower" if n_upper == 0 else "Upper"
#             st.error(
#                 f"**Incomplete upload — {missing} stack is missing.**\n\n"
#                 f"Full MLC QA analysis requires **at least one {present}** and **one {missing}** stack log.\n"
#                 "Please upload both stacks to continue."
#             )
#             st.session_state["upload_complete"] = False
#         else:
#             st.session_state["upload_complete"] = True

#         with st.expander("Data preview (for verification)"):
#             st.write("Parsed: all records")
#             st.dataframe(df_all.head(preview_rows), use_container_width=True)
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.write("Upper stack")
#                 st.dataframe(df_upper.head(preview_rows), use_container_width=True)
#             with c2:
#                 st.write("Lower stack")
#                 st.dataframe(df_lower.head(preview_rows), use_container_width=True)

#     st.subheader("Step 2 — Plan reference (local)")
#     PLAN_FOLDER = Path("data")
#     if _plans_available(PLAN_FOLDER):
#         st.success("Plan reference files detected in ./data ✅")
#         st.caption("Expected: data/dfP_all.pkl, data/dfP_upper.pkl, data/dfP_lower.pkl")
#     else:
#         st.error("Plan reference files not found in ./data ❌")
#         st.caption("Place: dfP_all.pkl, dfP_upper.pkl, dfP_lower.pkl inside ./data")

#     st.subheader("Step 3 — Matching options")
#     st.session_state["drop_stack_by_plan_name"] = st.checkbox(
#         "Auto-select stacks by plan naming convention (recommended)",
#         value=st.session_state.get("drop_stack_by_plan_name", True),
#     )
#     st.caption("Plan/log merge is performed automatically in the Analysis tab.")

# # =============================================================================
# # TAB 2: ANALYSIS
# # =============================================================================
# with tab_analysis:
#     st.header("Analysis")

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload both Upper and Lower stack logs in the **Upload** tab to enable analysis.")
#         st.stop()

#     PLAN_FOLDER = Path("data")
#     if not _plans_available(PLAN_FOLDER):
#         st.error("Plan reference files not found in ./data. Add PKLs then re-run.")
#         st.stop()

#     cA, cB = st.columns([1, 2])
#     with cA:
#         rerun_merge = st.button("Re-run plan/log matching", type="primary")
#     with cB:
#         st.caption("Use if plan reference files or matching options changed.")

#     if rerun_merge:
#         st.session_state.pop("mU", None)
#         st.session_state.pop("mL", None)

#     try:
#         with st.spinner("Matching plan to delivery logs..."):
#             _ensure_merged()
#         st.success(f"Matching complete. Upper: {st.session_state['mU'].shape} | Lower: {st.session_state['mL'].shape}")
#     except Exception as e:
#         st.error(str(e))
#         st.stop()

#     mU = st.session_state["mU"]
#     mL = st.session_state["mL"]

#     mUe = add_errors(mU)
#     mLe = add_errors(mL)
#     st.session_state["analysis_out"] = {"mUe": mUe, "mLe": mLe}

#     metricsU = leaf_metrics(mUe, "upper")
#     metricsL = leaf_metrics(mLe, "lower")

#     statusU, maxU = classify(metricsU, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))
#     statusL, maxL = classify(metricsL, warn_mm=float(action_level_mm), fail_mm=float(tolerance_level_mm))

#     st.subheader("QA status (quick view)")
#     c1, c2 = st.columns(2)
#     with c1:
#         _status_banner("Upper stack", statusU, maxU)
#     with c2:
#         _status_banner("Lower stack", statusL, maxL)

#     with st.expander("Detailed error statistics (mm)"):
#         c1, c2 = st.columns(2)
#         with c1:
#             st.write("Upper stack")
#             st.dataframe(error_describe(mUe), use_container_width=True)
#         with c2:
#             st.write("Lower stack")
#             st.dataframe(error_describe(mLe), use_container_width=True)

#     st.subheader("Virtual picket fence (delivered gap)")
#     picket_centers = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
#     gapU = compute_gap_df_from_merged(mUe, use_nominal=False)
#     gapL = compute_gap_df_from_merged(mLe, use_nominal=False)

#     st.pyplot(
#         plot_virtual_picket_fence(picket_centers_mm=picket_centers, gap_df=gapU, title="Upper stack — delivered gap pattern"),
#         clear_figure=True,
#     )
#     st.pyplot(
#         plot_virtual_picket_fence(picket_centers_mm=picket_centers, gap_df=gapL, title="Lower stack — delivered gap pattern"),
#         clear_figure=True,
#     )

#     st.subheader("Per-leaf summary")
#     comb, _, _ = make_combined_leaf_index(mUe, mLe)
#     st.pyplot(plot_rms_errors_by_leaf(rms_by_leaf(comb)), clear_figure=True)

#     th_plot = st.number_input(
#         "MaxAbs plot threshold (mm)",
#         min_value=0.0,
#         value=float(action_level_mm),
#         step=0.1,
#         help="Reference line for visualization only; clinical decision is based on QA status above.",
#     )
#     st.pyplot(plot_max_errors_by_leaf(max_by_leaf(comb), threshold_mm=float(th_plot)), clear_figure=True)

#     with st.expander("Per-leaf tables"):
#         st.write("Upper stack per-leaf metrics")
#         st.dataframe(metricsU.sort_values("leaf_pair"), use_container_width=True)
#         st.write("Lower stack per-leaf metrics")
#         st.dataframe(metricsL.sort_values("leaf_pair"), use_container_width=True)

# # =============================================================================
# # TAB 3: REPORTS
# # =============================================================================
# with tab_reports:
#     st.header("Reports")

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload both Upper and Lower stack logs in the **Upload** tab to enable reporting.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to generate the inputs needed for reporting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     pid = _safe_first(mUe, "Patient ID") or "ID"
#     dt = _safe_first(mUe, "Date") or "Date"
#     plan = _safe_first(mUe, "Plan Name") or "Plan"
#     default_pdf_name = f"MLC_QA_Report__{pid}__{plan}__{dt}.pdf".replace(" ", "_").replace("/", "-")

#     st.subheader("Report metadata")
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         report_site = st.text_input("Site", value=st.session_state.get("site_name", ""))
#     with c2:
#         report_machine = st.text_input("Machine", value=st.session_state.get("machine_name", "MRIdian"))
#     with c3:
#         report_reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer_name", ""))

#     report_title = st.text_input("Report title", value="MRIdian MLC Positional QA Report")

#     st.subheader("Report criteria")
#     tolerances = {
#         "warn_rms": float(rep_warn_rms),
#         "fail_rms": float(rep_fail_rms),
#         "warn_max": float(rep_warn_max),
#         "fail_max": float(rep_fail_max),
#     }

#     st.subheader("Trend logging (optional)")
#     update_trend = st.toggle("Append this run to trends", value=False)
#     trend_csv_path = None
#     if update_trend:
#         trend_csv_path = st.text_input(
#             "Trends file path (server-side)",
#             value=str(Path("data") / "trending" / "trend_report_bank_summary.csv"),
#         )

#     keep_server_copy = st.toggle("Save a server copy under ./outputs", value=False)

#     if st.button("Generate PDF report", type="primary"):
#         try:
#             with st.spinner("Generating PDF..."):
#                 pdf_bytes = generate_pdf_qa_report_bytes(
#                     mU=mUe,
#                     mL=mLe,
#                     report_title=report_title,
#                     tolerances=tolerances,
#                     trend_csv=Path(trend_csv_path) if update_trend and trend_csv_path else None,
#                     site=report_site,
#                     machine=report_machine,
#                     reviewer=report_reviewer,
#                 )
#             st.session_state["pdf_bytes"] = pdf_bytes
#             st.session_state["pdf_name"] = default_pdf_name
#             st.success("PDF report generated.")
#         except Exception as e:
#             st.error(f"Report generation failed: {e}")

#     if "pdf_bytes" in st.session_state and st.session_state["pdf_bytes"]:
#         st.download_button(
#             "Download PDF",
#             data=st.session_state["pdf_bytes"],
#             file_name=st.session_state.get("pdf_name", "MLC_QA_Report.pdf"),
#             mime="application/pdf",
#         )

#         if keep_server_copy:
#             out_dir = Path("outputs")
#             out_dir.mkdir(parents=True, exist_ok=True)
#             out_path = out_dir / st.session_state.get("pdf_name", "MLC_QA_Report.pdf")
#             try:
#                 out_path.write_bytes(st.session_state["pdf_bytes"])
#                 st.info(f"Saved server copy to: {out_path.as_posix()}")
#             except Exception as e:
#                 st.warning(f"Could not save server copy: {e}")

# # =============================================================================
# # TAB 4: TRENDS
# # =============================================================================
# with tab_trends:
#     st.header("MLC performance trends")

#     if not st.session_state.get("upload_complete", False) or not _upload_is_complete():
#         st.info("Upload both Upper and Lower stack logs in the **Upload** tab to enable trends.")
#         st.stop()

#     if "analysis_out" not in st.session_state:
#         st.info("Run **Analysis** first to enable trend updates and plotting.")
#         st.stop()

#     mUe = st.session_state["analysis_out"]["mUe"]
#     mLe = st.session_state["analysis_out"]["mLe"]

#     trend_path = Path("data") / "trending" / "trend_overall_max.csv"

#     st.subheader("Trend plot settings")
#     c1, c2 = st.columns(2)
#     with c1:
#         scope = st.selectbox("Scope", ["combined", "upper", "lower"], index=0)
#     with c2:
#         tol_for_plot = st.number_input(
#             "Tolerance reference (mm)",
#             min_value=0.0,
#             value=float(tolerance_level_mm),
#             step=0.1,
#             help="Reference line for the plot; does not change stored data.",
#         )

#     st.subheader("Update trend history")
#     st.caption("Appends the current run summary (overall max absolute error) to the trend history file.")

#     if st.button("Append current run to trends"):
#         rows = [
#             summarize_overall_max_error(mUe, scope="upper"),
#             summarize_overall_max_error(mLe, scope="lower"),
#             summarize_overall_max_error(pd.concat([mUe, mLe], ignore_index=True), scope="combined"),
#         ]
#         trend_all = append_trending_csv(trend_path, rows)
#         st.success(f"Trends updated: {trend_path.as_posix()}")
#         st.session_state["trend_all"] = trend_all

#     if "trend_all" in st.session_state:
#         trend_all = st.session_state["trend_all"]
#     elif trend_path.exists():
#         trend_all = pd.read_csv(trend_path)
#         st.session_state["trend_all"] = trend_all
#     else:
#         trend_all = None

#     if trend_all is None or len(trend_all) == 0:
#         st.info("No trend history found yet. Append a run above to begin tracking.")
#         st.stop()

#     st.subheader("Trend plot")
#     fig_tr = plot_overall_max_trending(trend_all, scope=scope, gantry_bin=None, fail_mm=float(tol_for_plot))
#     st.pyplot(fig_tr, clear_figure=True)

#     with st.expander("Trend history table"):
#         st.dataframe(trend_all, use_container_width=True)

#     st.download_button(
#         "Download trend history (CSV)",
#         data=trend_all.to_csv(index=False).encode("utf-8"),
#         file_name="trend_overall_max.csv",
#         mime="text/csv",
#     )

# # =============================================================================
# # Footer
# # =============================================================================
# st.markdown("---")
# st.caption(
#     f"{APP_SHORT_NAME} • Version {APP_VERSION} • "
#     "For support, contact your department system administrator."
# )
