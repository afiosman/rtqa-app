# core/report.py
from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.analysis import (
    plot_max_errors_by_leaf,
    top_worst_leaves,
)

# =============================================================================
# Branding
# =============================================================================
BRAND = {
    "navy": HexColor("#1f2a44"),
    "gold": HexColor("#C99700"),
    "bg": HexColor("#F5F7FA"),
    "panel": HexColor("#FFFFFF"),
    "border": HexColor("#E5E7EB"),
    "text": HexColor("#111827"),
    "muted": HexColor("#6B7280"),
    "success": HexColor("#0F766E"),
    "warn": HexColor("#B45309"),
    "danger": HexColor("#B91C1C"),
}


def _status_color(status: str):
    s = (status or "").upper().strip()
    if s == "PASS":
        return BRAND["success"]
    if s == "WARN":
        return BRAND["warn"]
    if s == "FAIL":
        return BRAND["danger"]
    return BRAND["muted"]


def _rl_color_to_hex(c: HexColor) -> str:
    hv = c.hexval()
    if isinstance(hv, str) and hv.startswith("0x") and len(hv) == 8:
        return "#" + hv[2:]
    return "#000000"


def _safe_str(x: Any, default: str = "N/A") -> str:
    if x is None:
        return default
    if isinstance(x, float) and np.isnan(x):
        return default
    s = str(x).strip()
    return s if s else default


def parse_log_date_to_ymd(date_str: object) -> Optional[str]:
    """
    Parses strings like: "Feb 12 2026 ..." -> "2026-02-12"
    Returns None if not parseable.
    """
    if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
        return None

    s = str(date_str).strip()
    m = re.match(r"^([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})", s)
    if not m:
        return None
    try:
        dt_ = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y")
        return dt_.strftime("%Y-%m-%d")
    except Exception:
        return None


def _get_first(df: Optional[pd.DataFrame], col: str):
    if df is None or df.empty or col not in df.columns:
        return None
    try:
        return df[col].iloc[0]
    except Exception:
        return None


def _require_columns(df: Optional[pd.DataFrame], required: List[str], df_name: str) -> None:
    if df is None or df.empty:
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _unique_join(df: Optional[pd.DataFrame], col: str, max_items: int = 12) -> str:
    """
    Collect unique non-empty values in a column and join them.
    Truncates to max_items with a "+N more" suffix.
    """
    if df is None or df.empty or col not in df.columns:
        return "N/A"

    s = df[col].dropna().astype(str).map(str.strip)
    s = s[s != ""]
    if s.empty:
        return "N/A"

    uniq = pd.unique(s)
    uniq = [v for v in uniq if v]
    if not uniq:
        return "N/A"

    if len(uniq) <= int(max_items):
        return ", ".join(uniq)

    head = ", ".join(uniq[: int(max_items)])
    return f"{head}, … (+{len(uniq) - int(max_items)} more)"


def _unique_join_two(
    df1: Optional[pd.DataFrame],
    df2: Optional[pd.DataFrame],
    col: str,
    max_items: int = 30,
) -> str:
    """
    Union of unique non-empty values from BOTH dataframes (preserve order).
    Truncates to max_items with a "+N more" suffix.
    """
    parts: List[str] = []

    for df in (df1, df2):
        if df is None or df.empty or col not in df.columns:
            continue
        s = df[col].dropna().astype(str).map(str.strip)
        s = s[s != ""]
        if not s.empty:
            parts.extend(pd.unique(s).tolist())

    seen = set()
    uniq: List[str] = []
    for p in parts:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    if not uniq:
        return "N/A"

    if len(uniq) <= int(max_items):
        return ", ".join(uniq)

    head = ", ".join(uniq[: int(max_items)])
    return f"{head}, … (+{len(uniq) - int(max_items)} more)"


# =============================================================================
# Gantry utilities (BINNED)
# =============================================================================
def gantry_to_bin(deg: object, step: float = 90.0) -> float:
    if pd.isna(deg):
        return np.nan
    d = float(deg) % 360.0
    b = (np.round(d / step) * step) % 360.0
    if np.isclose(b, 360.0) or np.isclose(b, 0.0):
        return 0.0
    return float(b)


def _extract_binned_gantry_angles_for_bank(
    m_df: Optional[pd.DataFrame],
    step: float = 90.0,
    gantry_col_candidates: Tuple[str, ...] = ("GantryDeg", "GantryAngle", "Gantry", "GantryBin"),
) -> Tuple[List[float], str]:
    if m_df is None or m_df.empty:
        return ([], "No data")

    gantry_col = None
    for c in gantry_col_candidates:
        if c in m_df.columns:
            gantry_col = c
            break
    if gantry_col is None:
        return ([], "Gantry column not found")

    vals = pd.to_numeric(m_df[gantry_col], errors="coerce").dropna().to_numpy()
    if vals.size == 0:
        return ([], f"{gantry_col} present but empty")

    if gantry_col == "GantryBin":
        binned = np.mod(vals.astype(float), 360.0)
    else:
        vals = np.mod(vals.astype(float), 360.0)
        binned = np.array([gantry_to_bin(v, step=step) for v in vals], dtype=float)

    binned = binned[~np.isnan(binned)]
    if binned.size == 0:
        return ([], f"{gantry_col} produced no valid bins")

    uniq = np.unique(np.round(binned, 3))
    return (np.sort(uniq).tolist(), gantry_col)


def _format_bins(angles: List[float]) -> str:
    if not angles:
        return "N/A"
    parts = []
    for a in angles:
        parts.append(str(int(a)) if float(a).is_integer() else f"{a:g}")
    return ", ".join(parts)


# =============================================================================
# Metrics computation (MaxAbs-focused)
# =============================================================================
def compute_bank_summary(m_df: Optional[pd.DataFrame], bank: str) -> dict:
    if m_df is None or m_df.empty:
        return {"bank": bank, "n": 0, "p95_abs_mm": np.nan, "max_abs_mm": np.nan}

    _require_columns(m_df, ["abs_err_left_mm", "abs_err_right_mm"], f"{bank} merged df")

    abs_left = pd.to_numeric(m_df["abs_err_left_mm"], errors="coerce").to_numpy(dtype=float)
    abs_right = pd.to_numeric(m_df["abs_err_right_mm"], errors="coerce").to_numpy(dtype=float)
    abs_errs = _finite(np.r_[abs_left, abs_right])

    p95 = float(np.percentile(abs_errs, 95)) if abs_errs.size else np.nan
    mx = float(np.max(abs_errs)) if abs_errs.size else np.nan

    return {"bank": bank, "n": int(len(m_df)), "p95_abs_mm": p95, "max_abs_mm": mx}


def compute_per_leaf_maxabs(m_df: Optional[pd.DataFrame], leaf_col: str = "leaf_pair") -> pd.DataFrame:
    if m_df is None or m_df.empty:
        return pd.DataFrame(columns=["mlc_leaf", "max_abs_left", "max_abs_right"])

    _require_columns(m_df, [leaf_col, "abs_err_left_mm", "abs_err_right_mm"], "compute_per_leaf_maxabs(m_df)")

    out = (
        m_df.groupby(leaf_col, as_index=False)
        .agg(
            max_abs_left=("abs_err_left_mm", "max"),
            max_abs_right=("abs_err_right_mm", "max"),
        )
        .rename(columns={leaf_col: "mlc_leaf"})
    )
    out["mlc_leaf"] = pd.to_numeric(out["mlc_leaf"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["mlc_leaf"])
    out["mlc_leaf"] = out["mlc_leaf"].astype(int)
    return out.sort_values("mlc_leaf").reset_index(drop=True)


def classify_status_maxabs(
    max_abs_mm: float,
    warn_max: float = 0.5,  # tolerance
    fail_max: float = 1.0,  # action
) -> str:
    if max_abs_mm is None or np.isnan(max_abs_mm):
        return "UNKNOWN"
    if float(max_abs_mm) > float(fail_max):
        return "FAIL"
    if float(max_abs_mm) > float(warn_max):
        return "WARN"
    return "PASS"


# =============================================================================
# Plot helpers
# =============================================================================
def _fig_to_png_bytes(fig: plt.Figure, dpi: int = 220) -> BytesIO:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _plot_maxabs_bytes(maxabs_df: pd.DataFrame, title: str, *, warn_mm: float, fail_mm: float) -> BytesIO:
    if maxabs_df is None or maxabs_df.empty:
        fig = plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        return _fig_to_png_bytes(fig)

    fig = plot_max_errors_by_leaf(
        maxabs_df,
        threshold_mm=float(warn_mm),  # tolerance
        fail_mm=float(fail_mm),       # action
        title=title,
        show_bands=True,
    )
    return _fig_to_png_bytes(fig)


# =============================================================================
# Trending CSV (no gantry stored)
# =============================================================================
def update_trending_csv(trend_csv: Path, summary_rows: List[dict]) -> pd.DataFrame:
    trend_csv = Path(trend_csv)
    trend_csv.parent.mkdir(parents=True, exist_ok=True)

    today_df = pd.DataFrame(summary_rows)

    if trend_csv.exists():
        all_df = pd.read_csv(trend_csv)
        all_df = pd.concat([all_df, today_df], ignore_index=True)
    else:
        all_df = today_df.copy()

    key_cols = [c for c in ["Date", "Machine", "bank"] if c in all_df.columns]
    if key_cols:
        all_df = all_df.drop_duplicates(subset=key_cols, keep="last")

    all_df.to_csv(trend_csv, index=False)
    return all_df


# =============================================================================
# PDF layout helpers
# =============================================================================
def _draw_header_footer(
    canvas,
    doc,
    *,
    title: str,
    subtitle: str,
    status: str,
    left_note: str,
    logo_path: Optional[Path] = None,
) -> None:
    canvas.saveState()
    page_w, page_h = letter

    canvas.setFillColor(BRAND["navy"])
    canvas.rect(0, page_h - 0.85 * inch, page_w, 0.85 * inch, stroke=0, fill=1)

    canvas.setFillColor(BRAND["gold"])
    canvas.rect(0, page_h - 0.85 * inch, page_w, 0.06 * inch, stroke=0, fill=1)

    x_left = 0.75 * inch
    if logo_path is not None:
        try:
            lp = Path(logo_path)
            if lp.exists():
                logo_w = 0.55 * inch
                logo_h = 0.55 * inch
                canvas.drawImage(str(lp), x_left, page_h - 0.78 * inch, width=logo_w, height=logo_h, mask="auto")
                x_left += 0.65 * inch
        except Exception:
            pass

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(x_left, page_h - 0.52 * inch, title)

    canvas.setFont("Helvetica", 9.5)
    canvas.setFillColor(HexColor("#E5E7EB"))
    canvas.drawString(x_left, page_h - 0.70 * inch, subtitle)

    badge_w = 1.35 * inch
    badge_h = 0.34 * inch
    x0 = page_w - 0.75 * inch - badge_w
    y0 = page_h - 0.62 * inch
    canvas.setFillColor(_status_color(status))
    canvas.roundRect(x0, y0, badge_w, badge_h, 8, stroke=0, fill=1)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawCentredString(x0 + badge_w / 2, y0 + 0.11 * inch, (status or "UNKNOWN").upper())

    canvas.setFillColor(BRAND["muted"])
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(0.75 * inch, 0.55 * inch, left_note)
    canvas.drawRightString(page_w - 0.75 * inch, 0.55 * inch, f"Page {doc.page}")

    canvas.restoreState()


def _table_style_key_value() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, -1), BRAND["panel"]),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#F3F4F6")),
            ("TEXTCOLOR", (0, 0), (-1, -1), BRAND["text"]),
            ("TEXTCOLOR", (0, 0), (0, -1), BRAND["muted"]),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.2),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, BRAND["border"]),
            ("BOX", (0, 0), (-1, -1), 0.8, BRAND["border"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )


def _table_style_summary(data: List[list]) -> TableStyle:
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EEF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), BRAND["text"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 1), (-2, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.8, BRAND["border"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, BRAND["border"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    for r in range(1, len(data)):
        if r % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, r), (-1, r), HexColor("#FAFBFC")))

    status_col = len(data[0]) - 1
    for r in range(1, len(data)):
        s = str(data[r][status_col]).upper()
        style_cmds.append(("TEXTCOLOR", (status_col, r), (status_col, r), _status_color(s)))
        style_cmds.append(("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold"))

    return TableStyle(style_cmds)


def _table_style_compact() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EEF2F7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.8),
            ("TEXTCOLOR", (0, 0), (-1, -1), BRAND["text"]),
            ("BOX", (0, 0), (-1, -1), 0.8, BRAND["border"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, BRAND["border"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


# =============================================================================
# Public API
# =============================================================================
def generate_pdf_qa_report_bytes(
    mU: pd.DataFrame,
    mL: pd.DataFrame,
    report_title: str = "MRIdian MLC Positional QA Report",
    tolerances: Optional[Dict[str, float]] = None,
    trend_csv: Optional[Path] = None,
    site: Optional[str] = None,
    machine: Optional[str] = None,
    reviewer: Optional[str] = None,
    gantry_bin_step_deg: float = 90.0,
    logo_path: Optional[Path] = None,
) -> bytes:
    """
    PDF report based on MaxAbs errors ONLY.

    Criteria:
      - PASS: max_abs <= warn_max (tolerance)
      - WARN: warn_max < max_abs <= fail_max (action)
      - FAIL: max_abs > fail_max

    NOTE (Option A):
      - This module does NOT write any PDF to disk.
      - It returns PDF bytes for Streamlit download.
      - Trend CSV update is optional (trend_csv argument).
    """
    if tolerances is None:
        tolerances = {"warn_max": 0.5, "fail_max": 1.0}
    warn_max = float(tolerances.get("warn_max", 0.5))
    fail_max = float(tolerances.get("fail_max", 1.0))

    required_cols = ["abs_err_left_mm", "abs_err_right_mm", "leaf_pair"]
    _require_columns(mU, required_cols, "mU (upper)")
    _require_columns(mL, required_cols, "mL (lower)")

    patient_name = _get_first(mU, "Patient Name") or _get_first(mL, "Patient Name")
    patient_id = _get_first(mU, "Patient ID") or _get_first(mL, "Patient ID")

    # FIX: union of BOTH banks (includes MLC1 + MLC2 if present)
    plan_name = _unique_join_two(mU, mL, "Plan Name", max_items=30)

    date_raw = _get_first(mU, "Date") or _get_first(mL, "Date")
    date_ymd = parse_log_date_to_ymd(date_raw) or _safe_str(date_raw, "UNKNOWN")

    if machine is None:
        machine = _get_first(mU, "Machine") or _get_first(mL, "Machine")

    binsU, _ = _extract_binned_gantry_angles_for_bank(mU, step=gantry_bin_step_deg)
    binsL, _ = _extract_binned_gantry_angles_for_bank(mL, step=gantry_bin_step_deg)
    binsU_str = _format_bins(binsU)
    binsL_str = _format_bins(binsL)

    sumU = compute_bank_summary(mU, "upper")
    sumL = compute_bank_summary(mL, "lower")
    sumU["status"] = classify_status_maxabs(sumU["max_abs_mm"], warn_max=warn_max, fail_max=fail_max)
    sumL["status"] = classify_status_maxabs(sumL["max_abs_mm"], warn_max=warn_max, fail_max=fail_max)

    rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "UNKNOWN": 0}
    overall_status = max([sumU["status"], sumL["status"]], key=lambda s: rank.get(str(s).upper(), 0))

    maxU = compute_per_leaf_maxabs(mU)
    maxL = compute_per_leaf_maxabs(mL)

    # Optional trend update (CSV)
    if trend_csv is not None:
        machine_for_csv = _safe_str(machine, default="MRIdian")
        rows = [
            {
                "Date": date_ymd,
                "Machine": machine_for_csv,
                "bank": "upper",
                "p95_abs_mm": sumU["p95_abs_mm"],
                "max_abs_mm": sumU["max_abs_mm"],
                "status": sumU["status"],
            },
            {
                "Date": date_ymd,
                "Machine": machine_for_csv,
                "bank": "lower",
                "p95_abs_mm": sumL["p95_abs_mm"],
                "max_abs_mm": sumL["max_abs_mm"],
                "status": sumL["status"],
            },
        ]
        update_trending_csv(Path(trend_csv), rows)

    maxU_png = _plot_maxabs_bytes(
        maxU,
        "Max Abs MLC Position Error — Upper Stack",
        warn_mm=warn_max,
        fail_mm=fail_max,
    )
    maxL_png = _plot_maxabs_bytes(
        maxL,
        "Max Abs MLC Position Error — Lower Stack",
        warn_mm=warn_max,
        fail_mm=fail_max,
    )

    worst_max_U = top_worst_leaves(maxU, n=5, mode="max")
    worst_max_L = top_worst_leaves(maxL, n=5, mode="max")

    base_styles = getSampleStyleSheet()
    styleTitle = ParagraphStyle(
        "TitleBrand",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=BRAND["text"],
        spaceAfter=10,
    )
    styleH = ParagraphStyle(
        "HBrand",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        textColor=BRAND["text"],
        spaceBefore=10,
        spaceAfter=6,
    )
    styleN = ParagraphStyle(
        "NBrand",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=BRAND["text"],
    )
    styleMuted = ParagraphStyle("Muted", parent=styleN, textColor=BRAND["muted"])
    styleKey = ParagraphStyle(
        "KeyCell",
        parent=styleN,
        fontName="Helvetica-Bold",
        fontSize=9.2,
        leading=11.2,
        textColor=BRAND["muted"],
    )
    styleVal = ParagraphStyle(
        "ValCell",
        parent=styleN,
        fontName="Helvetica",
        fontSize=9.2,
        leading=11.2,
        textColor=BRAND["text"],
    )

    def Pk(s: str) -> Paragraph:
        return Paragraph(_safe_str(s), styleKey)

    def Pv(s: Any) -> Paragraph:
        return Paragraph(_safe_str(s), styleVal)

    pdf_buf = BytesIO()

    header_subtitle = "Log file-based MLC positional QA from delivery records"
    footer_note = "Research and QA use only. Ensure local commissioning prior to clinical reliance."
    generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def on_page(canvas, doc):
        _draw_header_footer(
            canvas,
            doc,
            title=report_title,
            subtitle=header_subtitle,
            status=overall_status,
            left_note=f"{footer_note} • Generated {generated_ts}",
            logo_path=logo_path,
        )

    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=1.05 * inch,
        bottomMargin=0.85 * inch,
    )

    story: List[Any] = []

    story.append(Paragraph("Picket Fence QA Report", styleTitle))
    overall_hex = _rl_color_to_hex(_status_color(overall_status))
    story.append(
        Paragraph(
            f"Overall Status: <b><font color='{overall_hex}'>{_safe_str(overall_status, 'UNKNOWN')}</font></b>",
            styleN,
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    header_data = [
        [Pk("Site"), Pv(site)],
        [Pk("Machine"), Pv(_safe_str(machine, default="MRIdian"))],
        [Pk("Reviewer"), Pv(reviewer)],
        [Pk("Patient Name"), Pv(patient_name)],
        [Pk("Patient ID"), Pv(patient_id)],
        [Pk("Plan Name(s)"), Pv(plan_name)],
        [Pk("Date"), Pv(date_ymd)],
        [Pk("Gantry angles (°) — Upper Stack"), Pv(binsU_str)],
        [Pk("Gantry angles (°) — Lower Stack"), Pv(binsL_str)],
    ]
    t = Table(header_data, colWidths=[3.15 * inch, 3.85 * inch])
    t.setStyle(_table_style_key_value())
    story.append(t)
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Summary Metrics", styleH))
    story.append(
        Paragraph(
            "Bank-level absolute error statistics with PASS/WARN/FAIL classification (MaxAbs only).",
            styleMuted,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    summary_tbl = (
        pd.DataFrame([sumU, sumL])
        .rename(
            columns={
                "bank": "Bank",
                "p95_abs_mm": "p95 Abs. Error (mm)",
                "max_abs_mm": "Max Abs. Error (mm)",
            }
        )[["Bank", "p95 Abs. Error (mm)", "Max Abs. Error (mm)", "Status"]]
    )

    summary_tbl = summary_tbl.round({"p95 Abs. Error (mm)": 3, "Max Abs. Error (mm)": 3})
    data = [summary_tbl.columns.tolist()] + summary_tbl.values.tolist()

    tt = Table(data, colWidths=[1.1 * inch, 2.1 * inch, 2.1 * inch, 1.25 * inch])
    tt.setStyle(_table_style_summary(data))
    story.append(tt)
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Tolerance Criteria (MaxAbs)", styleH))
    story.append(
        Paragraph(
            f"Tolerance: {warn_max:.2f} mm • Action: {fail_max:.2f} mm "
            f"(PASS ≤ {warn_max:.2f}, WARN {warn_max:.2f}–{fail_max:.2f}, FAIL > {fail_max:.2f})",
            styleN,
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Worst-Case Leaves (Top 5)", styleH))
    story.append(Paragraph("Most deviating leaves per bank (MaxAbs).", styleMuted))
    story.append(Spacer(1, 0.06 * inch))

    worst_rows = [["Bank", "Leaf", "Max Abs. Error (mm)"]]
    for bank_label, df_ in [("Upper", worst_max_U), ("Lower", worst_max_L)]:
        for _, r in df_.iterrows():
            worst_rows.append([bank_label, int(r["mlc_leaf"]), f"{float(r['worst_mm']):.3f}"])

    worst_tbl = Table(worst_rows, colWidths=[1.2 * inch, 1.0 * inch, 1.8 * inch])
    worst_tbl.setStyle(_table_style_compact())
    story.append(worst_tbl)
    story.append(Spacer(1, 0.16 * inch))

    story.append(Paragraph("Per-leaf Maximum Absolute Error", styleH))
    story.append(
        Paragraph(
            "Shaded bands indicate PASS/WARN/FAIL regions under the configured thresholds.",
            styleMuted,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    img_w = 6.8 * inch
    img_h = 3.55 * inch
    story.append(Image(maxU_png, width=img_w, height=img_h))
    story.append(Spacer(1, 0.10 * inch))
    story.append(Image(maxL_png, width=img_w, height=img_h))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    pdf_buf.seek(0)
    return pdf_buf.getvalue()


# NOTE (Option A): intentionally removed generate_pdf_qa_report(...) that writes to disk.



# # core/report.py
# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from datetime import datetime
# from io import BytesIO
# from pathlib import Path
# from typing import Dict, Optional, Tuple, List, Any

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from reportlab.lib import colors
# from reportlab.lib.colors import HexColor
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
# from reportlab.lib.units import inch
# from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# # =============================================================================
# # Branding / UI tokens (near-commercial feel)
# # =============================================================================
# BRAND = {
#     "navy": HexColor("#1f2a44"),
#     "gold": HexColor("#C99700"),
#     "bg": HexColor("#F5F7FA"),
#     "panel": HexColor("#FFFFFF"),
#     "border": HexColor("#E5E7EB"),
#     "text": HexColor("#111827"),
#     "muted": HexColor("#6B7280"),
#     "success": HexColor("#0F766E"),
#     "warn": HexColor("#B45309"),
#     "danger": HexColor("#B91C1C"),
# }


# def _status_color(status: str):
#     s = (status or "").upper()
#     if s == "PASS":
#         return BRAND["success"]
#     if s == "WARN":
#         return BRAND["warn"]
#     if s == "FAIL":
#         return BRAND["danger"]
#     return BRAND["muted"]


# # =============================================================================
# # Small utilities
# # =============================================================================
# def _safe_str(x: Any, default: str = "N/A") -> str:
#     if x is None:
#         return default
#     if isinstance(x, float) and np.isnan(x):
#         return default
#     s = str(x).strip()
#     return s if s else default


# def parse_log_date_to_ymd(date_str: object) -> Optional[str]:
#     """
#     Examples:
#       'Jan 13 2026 17:52' -> '2026-01-13'
#       'Jan 9 2026'        -> '2026-01-09'
#     """
#     if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
#         return None

#     s = str(date_str).strip()
#     m = re.match(r"^([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})", s)
#     if not m:
#         return None

#     try:
#         dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y")
#         return dt.strftime("%Y-%m-%d")
#     except Exception:
#         return None


# def _get_first(df: Optional[pd.DataFrame], col: str):
#     if df is None or df.empty or col not in df.columns:
#         return None
#     try:
#         return df[col].iloc[0]
#     except Exception:
#         return None


# def _require_columns(df: Optional[pd.DataFrame], required: List[str], df_name: str) -> None:
#     """Raise a clear error if required columns are missing."""
#     if df is None or df.empty:
#         return
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"{df_name} is missing required columns: {missing}")


# # =============================================================================
# # Gantry utilities (BINNED, per bank)
# # =============================================================================
# def gantry_to_bin(deg: object, step: float = 90.0) -> float:
#     """Map gantry angle to nearest bin (default: 0/90/180/270)."""
#     if pd.isna(deg):
#         return np.nan
#     d = float(deg) % 360.0
#     b = (np.round(d / step) * step) % 360.0
#     if np.isclose(b, 360.0) or np.isclose(b, 0.0):
#         return 0.0
#     return float(b)


# def _extract_binned_gantry_angles_for_bank(
#     m_df: Optional[pd.DataFrame],
#     step: float = 90.0,
#     gantry_col_candidates: Tuple[str, ...] = ("GantryDeg", "GantryAngle", "Gantry"),
# ) -> Tuple[List[float], str]:
#     """
#     Return (sorted_unique_binned_angles_deg, gantry_col_used_or_reason).
#     - Uses only the provided bank dataframe (Upper or Lower).
#     - Drops NaNs, normalizes, bins, then uniques.
#     """
#     if m_df is None or m_df.empty:
#         return ([], "No data")

#     gantry_col = None
#     for c in gantry_col_candidates:
#         if c in m_df.columns:
#             gantry_col = c
#             break
#     if gantry_col is None:
#         return ([], "Gantry column not found")

#     vals = pd.to_numeric(m_df[gantry_col], errors="coerce").dropna().to_numpy()
#     if vals.size == 0:
#         return ([], f"{gantry_col} present but empty")

#     vals = np.mod(vals.astype(float), 360.0)
#     binned = np.array([gantry_to_bin(v, step=step) for v in vals], dtype=float)
#     binned = binned[~np.isnan(binned)]
#     if binned.size == 0:
#         return ([], f"{gantry_col} produced no valid bins")

#     uniq = np.unique(np.round(binned, 3))
#     return (np.sort(uniq).tolist(), gantry_col)


# def _format_bins(angles: List[float]) -> str:
#     """Format binned angles: '0, 90, 180, 270'."""
#     if not angles:
#         return "N/A"
#     parts = []
#     for a in angles:
#         parts.append(str(int(a)) if float(a).is_integer() else f"{a:g}")
#     return ", ".join(parts)


# # =============================================================================
# # Metrics computation
# # Expected input schema (mU/mL):
# #   leaf_pair (int)
# #   err_left_mm, err_right_mm
# #   abs_err_left_mm, abs_err_right_mm
# # =============================================================================
# def compute_bank_summary(m_df: Optional[pd.DataFrame], bank: str) -> dict:
#     if m_df is None or m_df.empty:
#         return {
#             "bank": bank,
#             "n": 0,
#             "rms_left_mm": np.nan,
#             "rms_right_mm": np.nan,
#             "mean_rms_mm": np.nan,
#             "p95_abs_mm": np.nan,
#             "max_abs_mm": np.nan,
#         }

#     abs_errs = np.concatenate(
#         [m_df["abs_err_left_mm"].to_numpy(), m_df["abs_err_right_mm"].to_numpy()]
#     )

#     rms_left = float(np.sqrt(np.mean(m_df["err_left_mm"].to_numpy() ** 2)))
#     rms_right = float(np.sqrt(np.mean(m_df["err_right_mm"].to_numpy() ** 2)))
#     mean_rms = float((rms_left + rms_right) / 2.0)

#     return {
#         "bank": bank,
#         "n": int(len(m_df)),
#         "rms_left_mm": rms_left,
#         "rms_right_mm": rms_right,
#         "mean_rms_mm": mean_rms,
#         "p95_abs_mm": float(np.percentile(abs_errs, 95)),
#         "max_abs_mm": float(np.max(abs_errs)),
#     }


# def compute_per_leaf_rms(m_df: Optional[pd.DataFrame], leaf_col: str = "leaf_pair") -> pd.DataFrame:
#     if m_df is None or m_df.empty:
#         return pd.DataFrame(columns=["mlc_leaf", "rms_left_mm", "rms_right_mm"])

#     g = m_df.groupby(leaf_col, as_index=True)
#     out = pd.DataFrame(
#         {
#             "mlc_leaf": g.size().index.astype(int),
#             "rms_left_mm": g["err_left_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).values,
#             "rms_right_mm": g["err_right_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).values,
#         }
#     ).sort_values("mlc_leaf").reset_index(drop=True)

#     return out


# def compute_per_leaf_maxabs(m_df: Optional[pd.DataFrame], leaf_col: str = "leaf_pair") -> pd.DataFrame:
#     if m_df is None or m_df.empty:
#         return pd.DataFrame(columns=["mlc_leaf", "maxabs_left_mm", "maxabs_right_mm"])

#     out = (
#         m_df.groupby(leaf_col, as_index=False)
#         .agg(
#             maxabs_left_mm=("abs_err_left_mm", "max"),
#             maxabs_right_mm=("abs_err_right_mm", "max"),
#         )
#         .rename(columns={leaf_col: "mlc_leaf"})
#     )
#     out["mlc_leaf"] = out["mlc_leaf"].astype(int)
#     return out.sort_values("mlc_leaf").reset_index(drop=True)


# def classify_status(
#     mean_rms_mm: float,
#     max_abs_mm: float,
#     warn_max: float = 0.5,
#     fail_max: float = 1.0,
# ) -> str:
#     if np.isnan(mean_rms_mm) or np.isnan(max_abs_mm):
#         return "UNKNOWN"
#     if (max_abs_mm >= fail_max):
#         return "FAIL"
#     if (max_abs_mm >= warn_max):
#         return "WARN"
#     return "PASS"


# # =============================================================================
# # Plot helpers (in-memory PNG for PDF embedding)
# # =============================================================================
# def _plot_rms_bytes(rms_df: pd.DataFrame, title: str) -> BytesIO:
#     buf = BytesIO()
#     fig = plt.figure(figsize=(7, 4))

#     if rms_df is None or rms_df.empty:
#         plt.text(0.5, 0.5, "No data", ha="center", va="center")
#         plt.axis("off")
#     else:
#         plt.plot(
#             rms_df["mlc_leaf"],
#             rms_df["rms_left_mm"],
#             marker="o",
#             markersize=3,
#             linewidth=1,
#             label="Y1 (Left)",
#         )
#         plt.plot(
#             rms_df["mlc_leaf"],
#             rms_df["rms_right_mm"],
#             marker="x",
#             markersize=3,
#             linewidth=1,
#             label="Y2 (Right)",
#         )
#         plt.xlabel("MLC Leaf")
#         plt.ylabel("RMS of MLC Position Error (mm)")
#         ymax = rms_df[["rms_left_mm", "rms_right_mm"]].to_numpy().max()
#         plt.ylim(0, max(1.0, float(ymax) * 1.2))
#         plt.grid(True, alpha=0.3)
#         plt.legend()

#     plt.title(title)
#     plt.tight_layout()
#     fig.savefig(buf, format="png", dpi=220)
#     plt.close(fig)
#     buf.seek(0)
#     return buf


# def _plot_maxabs_bytes(maxabs_df: pd.DataFrame, title: str) -> BytesIO:
#     buf = BytesIO()
#     fig = plt.figure(figsize=(7, 4))

#     if maxabs_df is None or maxabs_df.empty:
#         plt.text(0.5, 0.5, "No data", ha="center", va="center")
#         plt.axis("off")
#     else:
#         plt.plot(
#             maxabs_df["mlc_leaf"],
#             maxabs_df["maxabs_left_mm"],
#             marker="o",
#             markersize=3,
#             linewidth=1,
#             label="Y1 (Left)",
#         )
#         plt.plot(
#             maxabs_df["mlc_leaf"],
#             maxabs_df["maxabs_right_mm"],
#             marker="x",
#             markersize=3,
#             linewidth=1,
#             label="Y2 (Right)",
#         )
#         plt.xlabel("MLC Leaf")
#         plt.ylabel("Max abs MLC Position Error (mm)")
#         ymax = maxabs_df[["maxabs_left_mm", "maxabs_right_mm"]].to_numpy().max()
#         plt.ylim(0, max(2.0, float(ymax) * 1.2))
#         plt.grid(True, alpha=0.3)
#         plt.legend()

#     plt.title(title)
#     plt.tight_layout()
#     fig.savefig(buf, format="png", dpi=220)
#     plt.close(fig)
#     buf.seek(0)
#     return buf


# # =============================================================================
# # Trending CSV (NO gantry raw/bin)
# # =============================================================================
# def update_trending_csv(trend_csv: Path, summary_rows: List[dict]) -> pd.DataFrame:
#     trend_csv = Path(trend_csv)
#     trend_csv.parent.mkdir(parents=True, exist_ok=True)

#     today_df = pd.DataFrame(summary_rows)

#     if trend_csv.exists():
#         all_df = pd.read_csv(trend_csv)
#         all_df = pd.concat([all_df, today_df], ignore_index=True)
#     else:
#         all_df = today_df.copy()

#     key_cols = [c for c in ["Date", "Machine", "bank"] if c in all_df.columns]
#     if key_cols:
#         all_df = all_df.drop_duplicates(subset=key_cols, keep="last")

#     all_df.to_csv(trend_csv, index=False)
#     return all_df


# # =============================================================================
# # PDF layout helpers (header/footer drawing)
# # =============================================================================
# def _draw_header_footer(
#     canvas,
#     doc,
#     *,
#     title: str,
#     subtitle: str,
#     status: str,
#     left_note: str,
#     logo_path: Optional[Path] = None,
# ) -> None:
#     """
#     Draw a near-commercial header band + footer disclaimer.

#     - Header: navy band + gold accent line, title/subtitle, PASS/WARN/FAIL badge
#     - Footer: note + page number
#     """
#     canvas.saveState()

#     page_w, page_h = letter

#     # Header band
#     canvas.setFillColor(BRAND["navy"])
#     canvas.rect(0, page_h - 0.85 * inch, page_w, 0.85 * inch, stroke=0, fill=1)

#     # Gold accent line
#     canvas.setFillColor(BRAND["gold"])
#     canvas.rect(0, page_h - 0.85 * inch, page_w, 0.06 * inch, stroke=0, fill=1)

#     # Optional logo (kept safe: draw only if exists)
#     x_left = 0.75 * inch
#     if logo_path is not None:
#         try:
#             lp = Path(logo_path)
#             if lp.exists():
#                 # drawImage expects path string; keep small and aligned
#                 logo_w = 0.55 * inch
#                 logo_h = 0.55 * inch
#                 canvas.drawImage(str(lp), x_left, page_h - 0.78 * inch, width=logo_w, height=logo_h, mask="auto")
#                 x_left += 0.65 * inch
#         except Exception:
#             # never crash report generation due to logo
#             pass

#     # Title text
#     canvas.setFillColor(colors.white)
#     canvas.setFont("Helvetica-Bold", 14)
#     canvas.drawString(x_left, page_h - 0.52 * inch, title)

#     canvas.setFont("Helvetica", 9.5)
#     canvas.setFillColor(HexColor("#E5E7EB"))
#     canvas.drawString(x_left, page_h - 0.70 * inch, subtitle)

#     # Status badge (right)
#     badge_w = 1.35 * inch
#     badge_h = 0.34 * inch
#     x0 = page_w - 0.75 * inch - badge_w
#     y0 = page_h - 0.62 * inch
#     canvas.setFillColor(_status_color(status))
#     canvas.roundRect(x0, y0, badge_w, badge_h, 8, stroke=0, fill=1)

#     canvas.setFillColor(colors.white)
#     canvas.setFont("Helvetica-Bold", 10)
#     canvas.drawCentredString(x0 + badge_w / 2, y0 + 0.11 * inch, (status or "UNKNOWN").upper())

#     # Footer
#     canvas.setFillColor(BRAND["muted"])
#     canvas.setFont("Helvetica", 8.5)
#     canvas.drawString(0.75 * inch, 0.55 * inch, left_note)
#     canvas.drawRightString(page_w - 0.75 * inch, 0.55 * inch, f"Page {doc.page}")

#     canvas.restoreState()


# def _table_style_key_value() -> TableStyle:
#     return TableStyle(
#         [
#             ("BACKGROUND", (0, 0), (-1, -1), BRAND["panel"]),
#             ("BACKGROUND", (0, 0), (0, -1), HexColor("#F3F4F6")),
#             ("TEXTCOLOR", (0, 0), (-1, -1), BRAND["text"]),
#             ("TEXTCOLOR", (0, 0), (0, -1), BRAND["muted"]),
#             ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
#             ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
#             ("FONTSIZE", (0, 0), (-1, -1), 9.5),
#             ("LINEBELOW", (0, 0), (-1, -1), 0.25, BRAND["border"]),
#             ("BOX", (0, 0), (-1, -1), 0.8, BRAND["border"]),
#             ("LEFTPADDING", (0, 0), (-1, -1), 8),
#             ("RIGHTPADDING", (0, 0), (-1, -1), 8),
#             ("TOPPADDING", (0, 0), (-1, -1), 6),
#             ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
#             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#         ]
#     )


# def _table_style_summary(data: List[list]) -> TableStyle:
#     # Base summary style
#     style_cmds = [
#         ("BACKGROUND", (0, 0), (-1, 0), HexColor("#EEF2F7")),
#         ("TEXTCOLOR", (0, 0), (-1, 0), BRAND["text"]),
#         ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
#         ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
#         ("FONTSIZE", (0, 0), (-1, -1), 9),
#         ("ALIGN", (1, 1), (-2, -1), "CENTER"),
#         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#         ("BOX", (0, 0), (-1, -1), 0.8, BRAND["border"]),
#         ("INNERGRID", (0, 0), (-1, -1), 0.25, BRAND["border"]),
#         ("LEFTPADDING", (0, 0), (-1, -1), 6),
#         ("RIGHTPADDING", (0, 0), (-1, -1), 6),
#         ("TOPPADDING", (0, 0), (-1, -1), 5),
#         ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
#     ]

#     # Zebra rows
#     for r in range(1, len(data)):
#         if r % 2 == 0:
#             style_cmds.append(("BACKGROUND", (0, r), (-1, r), HexColor("#FAFBFC")))

#     # Color status column (last)
#     status_col = len(data[0]) - 1
#     for r in range(1, len(data)):
#         s = str(data[r][status_col]).upper()
#         style_cmds.append(("TEXTCOLOR", (status_col, r), (status_col, r), _status_color(s)))
#         style_cmds.append(("FONTNAME", (status_col, r), (status_col, r), "Helvetica-Bold"))

#     return TableStyle(style_cmds)


# # =============================================================================
# # Public API
# # =============================================================================
# def generate_pdf_qa_report_bytes(
#     mU: pd.DataFrame,
#     mL: pd.DataFrame,
#     report_title: str = "MRIdian MLC Positional QA Report",
#     tolerances: Optional[Dict[str, float]] = None,
#     trend_csv: Optional[Path] = None,
#     site: Optional[str] = None,
#     machine: Optional[str] = None,
#     reviewer: Optional[str] = None,
#     gantry_bin_step_deg: float = 90.0,
#     logo_path: Optional[Path] = None,
# ) -> bytes:
#     """
#     Generate a multi-page PDF QA report and return it as bytes.

#     Header includes:
#       - Site / Machine / Reviewer (passed from app)
#       - Patient Name / ID / Plan / Date (from merged dfs)
#       - Gantry angles (°) – Upper Stack (binned)
#       - Gantry angles (°) – Lower Stack (binned)

#     Trending CSV update (optional) uses Machine but does NOT store gantry raw/bin.
#     """
#     if tolerances is None:
#         tolerances = {"warn_rms": 0.8, "fail_rms": 1.2, "warn_max": 1.5, "fail_max": 2.0}

#     # Validate required columns (fail early with a clear message)
#     required_cols = ["err_left_mm", "err_right_mm", "abs_err_left_mm", "abs_err_right_mm", "leaf_pair"]
#     _require_columns(mU, required_cols, "mU (upper)")
#     _require_columns(mL, required_cols, "mL (lower)")

#     # --- metadata (from merged dataframes) ---
#     patient_name = _get_first(mU, "Patient Name") or _get_first(mL, "Patient Name")
#     patient_id = _get_first(mU, "Patient ID") or _get_first(mL, "Patient ID")
#     plan_name = _get_first(mU, "Plan Name") or _get_first(mL, "Plan Name")
#     date_raw = _get_first(mU, "Date") or _get_first(mL, "Date")
#     date_ymd = parse_log_date_to_ymd(date_raw) or _safe_str(date_raw, "UNKNOWN")

#     # fallback machine from data if not passed
#     if machine is None:
#         machine = _get_first(mU, "Machine") or _get_first(mL, "Machine")

#     # --- gantry bins per bank ---
#     binsU, gantryU_col = _extract_binned_gantry_angles_for_bank(mU, step=gantry_bin_step_deg)
#     binsL, gantryL_col = _extract_binned_gantry_angles_for_bank(mL, step=gantry_bin_step_deg)
#     binsU_str = _format_bins(binsU)
#     binsL_str = _format_bins(binsL)

#     # --- summaries ---
#     sumU = compute_bank_summary(mU, "upper")
#     sumL = compute_bank_summary(mL, "lower")
#     sumU["status"] = classify_status(sumU["mean_rms_mm"], sumU["max_abs_mm"], **tolerances)
#     sumL["status"] = classify_status(sumL["mean_rms_mm"], sumL["max_abs_mm"], **tolerances)

#     # overall status = worst of (upper, lower)
#     rank = {"FAIL": 3, "WARN": 2, "PASS": 1, "UNKNOWN": 0}
#     overall_status = max([sumU["status"], sumL["status"]], key=lambda s: rank.get(str(s).upper(), 0))

#     # --- per-leaf metrics (for plots) ---
#     rmsU = compute_per_leaf_rms(mU)
#     rmsL = compute_per_leaf_rms(mL)
#     maxU = compute_per_leaf_maxabs(mU)
#     maxL = compute_per_leaf_maxabs(mL)

#     # --- optional trending update ---
#     if trend_csv is not None:
#         machine_for_csv = _safe_str(machine, default="MRIdian")
#         rows = [
#             {
#                 "Date": date_ymd,
#                 "Machine": machine_for_csv,
#                 "bank": "upper",
#                 "mean_rms_mm": sumU["mean_rms_mm"],
#                 "p95_abs_mm": sumU["p95_abs_mm"],
#                 "max_abs_mm": sumU["max_abs_mm"],
#                 "status": sumU["status"],
#             },
#             {
#                 "Date": date_ymd,
#                 "Machine": machine_for_csv,
#                 "bank": "lower",
#                 "mean_rms_mm": sumL["mean_rms_mm"],
#                 "p95_abs_mm": sumL["p95_abs_mm"],
#                 "max_abs_mm": sumL["max_abs_mm"],
#                 "status": sumL["status"],
#             },
#         ]
#         update_trending_csv(Path(trend_csv), rows)

#     # --- plots (PNG in memory) ---
#     rmsU_png = _plot_rms_bytes(rmsU, "RMS of MLC Position Error — Upper Stack")
#     rmsL_png = _plot_rms_bytes(rmsL, "RMS of MLC Position Error — Lower Stack")
#     maxU_png = _plot_maxabs_bytes(maxU, "Max Abs MLC Position Error — Upper Stack")
#     maxL_png = _plot_maxabs_bytes(maxL, "Max Abs MLC Position Error — Lower Stack")

#     # --- styles ---
#     base_styles = getSampleStyleSheet()
#     styleTitle = ParagraphStyle(
#         "TitleBrand",
#         parent=base_styles["Title"],
#         fontName="Helvetica-Bold",
#         fontSize=18,
#         textColor=BRAND["text"],
#         spaceAfter=10,
#     )
#     styleH = ParagraphStyle(
#         "HBrand",
#         parent=base_styles["Heading2"],
#         fontName="Helvetica-Bold",
#         fontSize=12.5,
#         textColor=BRAND["text"],
#         spaceBefore=10,
#         spaceAfter=6,
#     )
#     styleN = ParagraphStyle(
#         "NBrand",
#         parent=base_styles["Normal"],
#         fontName="Helvetica",
#         fontSize=9.5,
#         leading=12,
#         textColor=BRAND["text"],
#     )
#     styleMuted = ParagraphStyle(
#         "Muted",
#         parent=styleN,
#         textColor=BRAND["muted"],
#     )

#     # --- build PDF ---
#     pdf_buf = BytesIO()

#     header_subtitle = "Automated log-file–based MLC positional QA from delivery records"
#     footer_note = "Research and QA use only. Not FDA cleared. Ensure local commissioning prior to clinical reliance."
#     generated_ts = datetime.now().strftime("%Y-%m-%d %H:%M")

#     def on_page(canvas, doc):
#         _draw_header_footer(
#             canvas,
#             doc,
#             title=report_title,
#             subtitle=header_subtitle,
#             status=overall_status,
#             left_note=f"{footer_note} • Generated {generated_ts}",
#             logo_path=logo_path,
#         )

#     doc = SimpleDocTemplate(
#         pdf_buf,
#         pagesize=letter,
#         leftMargin=0.75 * inch,
#         rightMargin=0.75 * inch,
#         topMargin=1.05 * inch,   # leave room for header band
#         bottomMargin=0.85 * inch,
#     )

#     story: List[Any] = []

#     # Report title area (inside content, below header band)
#     story.append(Paragraph("QA Report", styleTitle))
#     story.append(Paragraph(f"Overall Status: <b><font color='{_status_color(overall_status).hexval()}'>{overall_status}</font></b>", styleN))
#     story.append(Spacer(1, 0.12 * inch))

#     # --- Header key-value table ---
#     header_data = [
#         ["Site", _safe_str(site)],
#         ["Machine", _safe_str(machine, default="MRIdian")],
#         ["Reviewer", _safe_str(reviewer)],
#         ["Patient Name", _safe_str(patient_name)],
#         ["Patient ID", _safe_str(patient_id)],
#         ["Plan Name", _safe_str(plan_name)],
#         ["Date", _safe_str(date_ymd)],
#         ["Gantry angles (°) — Upper Stack (binned)", binsU_str],
#         ["Gantry angles (°) — Lower Stack (binned)", binsL_str],
#         ["Gantry source — Upper", _safe_str(gantryU_col)],
#         ["Gantry source — Lower", _safe_str(gantryL_col)],
#     ]
#     t = Table(header_data, colWidths=[2.5 * inch, 4.5 * inch])
#     t.setStyle(_table_style_key_value())
#     story.append(t)
#     story.append(Spacer(1, 0.18 * inch))

#     # --- Summary table ---
#     story.append(Paragraph("Summary Metrics", styleH))
#     story.append(Paragraph("Bank-level RMS and absolute error statistics with PASS/WARN/FAIL classification.", styleMuted))
#     story.append(Spacer(1, 0.08 * inch))

#     summary_tbl = pd.DataFrame([sumU, sumL])[["bank", "n", "mean_rms_mm", "p95_abs_mm", "max_abs_mm", "status"]]
#     summary_tbl = summary_tbl.round({"mean_rms_mm": 3, "p95_abs_mm": 3, "max_abs_mm": 3})
#     data = [summary_tbl.columns.tolist()] + summary_tbl.values.tolist()

#     tt = Table(
#         data,
#         colWidths=[1.0 * inch, 0.7 * inch, 1.25 * inch, 1.15 * inch, 1.15 * inch, 0.95 * inch],
#     )
#     tt.setStyle(_table_style_summary(data))
#     story.append(tt)
#     story.append(Spacer(1, 0.18 * inch))

#     # --- Tolerance notes ---
#     story.append(Paragraph("Tolerance Criteria", styleH))
#     tol_txt = (
#         f"RMS thresholds (WARN/FAIL): {tolerances['warn_rms']:.2f} / {tolerances['fail_rms']:.2f} mm • "
#         f"Max abs thresholds (WARN/FAIL): {tolerances['warn_max']:.2f} / {tolerances['fail_max']:.2f} mm"
#     )
#     story.append(Paragraph(tol_txt, styleN))
#     story.append(Spacer(1, 0.12 * inch))

#     # --- Plots ---
#     story.append(Paragraph("Per-leaf Performance", styleH))
#     story.append(Paragraph("RMS and maximum absolute error per leaf are shown for upper and lower stacks.", styleMuted))
#     story.append(Spacer(1, 0.10 * inch))

#     img_w = 6.8 * inch
#     img_h = 3.55 * inch

#     story.append(Image(rmsU_png, width=img_w, height=img_h))
#     story.append(Spacer(1, 0.10 * inch))
#     story.append(Image(rmsL_png, width=img_w, height=img_h))
#     story.append(PageBreak())

#     story.append(Paragraph("Per-leaf Maximum Absolute Error", styleH))
#     story.append(Spacer(1, 0.08 * inch))
#     story.append(Image(maxU_png, width=img_w, height=img_h))
#     story.append(Spacer(1, 0.10 * inch))
#     story.append(Image(maxL_png, width=img_w, height=img_h))

#     doc.build(story, onFirstPage=on_page, onLaterPages=on_page)

#     pdf_buf.seek(0)
#     return pdf_buf.getvalue()


# def generate_pdf_qa_report(
#     output_pdf: Path,
#     mU: pd.DataFrame,
#     mL: pd.DataFrame,
#     trend_csv: Optional[Path] = None,
#     report_title: str = "MRIdian MLC Positional QA Report",
#     tolerances: Optional[Dict[str, float]] = None,
#     site: Optional[str] = None,
#     machine: Optional[str] = None,
#     reviewer: Optional[str] = None,
#     gantry_bin_step_deg: float = 90.0,
#     logo_path: Optional[Path] = None,
# ) -> Path:
#     """
#     Convenience wrapper: writes the PDF to disk (and optionally updates trending CSV).
#     Returns output_pdf.
#     """
#     output_pdf = Path(output_pdf)
#     output_pdf.parent.mkdir(parents=True, exist_ok=True)

#     pdf_bytes = generate_pdf_qa_report_bytes(
#         mU=mU,
#         mL=mL,
#         report_title=report_title,
#         tolerances=tolerances,
#         trend_csv=trend_csv,
#         site=site,
#         machine=machine,
#         reviewer=reviewer,
#         gantry_bin_step_deg=gantry_bin_step_deg,
#         logo_path=logo_path,
#     )
#     output_pdf.write_bytes(pdf_bytes)
#     return output_pdf

# # app.py
# # Near-commercial-grade Streamlit UI template (CSS + layout scaffolding)
# # Run: streamlit run app.py

# from __future__ import annotations

# import time
# from dataclasses import dataclass
# from datetime import datetime
# from typing import List, Dict, Any

# import pandas as pd
# import streamlit as st


# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(
#     page_title="MRIdian Log-Based QA Suite",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )


# # -----------------------------
# # Theme tokens (edit these)
# # -----------------------------
# THEME = {
#     "primary": "#1f2a44",      # deep navy
#     "accent": "#C99700",       # VCU gold
#     "bg": "#f5f7fa",
#     "panel": "#ffffff",
#     "border": "#e5e7eb",
#     "text": "#111827",
#     "muted": "#6b7280",
#     "success": "#0f766e",
#     "warn": "#b45309",
#     "danger": "#b91c1c",
# }


# # -----------------------------
# # CSS Injection
# # -----------------------------
# def inject_css(theme: Dict[str, str]) -> None:
#     st.markdown(
#         f"""
# <style>
# /* --- Base --- */
# :root {{
#   --primary: {theme["primary"]};
#   --accent: {theme["accent"]};
#   --bg: {theme["bg"]};
#   --panel: {theme["panel"]};
#   --border: {theme["border"]};
#   --text: {theme["text"]};
#   --muted: {theme["muted"]};
#   --success: {theme["success"]};
#   --warn: {theme["warn"]};
#   --danger: {theme["danger"]};
#   --radius: 16px;
#   --shadow: 0 10px 30px rgba(17, 24, 39, 0.08);
# }}

# html, body, [class*="css"]  {{
#   font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
#   color: var(--text);
# }}

# .stApp {{
#   background: var(--bg);
# }}

# /* Hide Streamlit header & footer for a clean product feel */
# header[data-testid="stHeader"] {{
#   visibility: hidden;
#   height: 0px;
# }}
# footer {{
#   visibility: hidden;
# }}

# /* Reduce top padding */
# .block-container {{
#   padding-top: 1.0rem !important;
#   padding-bottom: 2.5rem !important;
#   max-width: 1280px;
# }}

# /* --- Sidebar polish --- */
# section[data-testid="stSidebar"] {{
#   background: linear-gradient(180deg, rgba(31,42,68,0.98), rgba(31,42,68,0.93));
#   border-right: 1px solid rgba(255,255,255,0.06);
# }}
# section[data-testid="stSidebar"] * {{
#   color: rgba(255,255,255,0.92) !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input,
# section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
#   background: rgba(255,255,255,0.08) !important;
#   border: 1px solid rgba(255,255,255,0.12) !important;
#   border-radius: 12px !important;
# }}
# section[data-testid="stSidebar"] .stTextInput input::placeholder {{
#   color: rgba(255,255,255,0.55) !important;
# }}
# section[data-testid="stSidebar"] .stMarkdown small {{
#   color: rgba(255,255,255,0.65) !important;
# }}

# /* --- Top bar --- */
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
#   display: flex;
#   flex-direction: column;
#   gap: 2px;
# }}
# .brand-title {{
#   font-size: 1.05rem;
#   font-weight: 700;
#   letter-spacing: 0.2px;
# }}
# .brand-sub {{
#   font-size: 0.88rem;
#   color: var(--muted);
# }}
# .topbar-right {{
#   display: flex;
#   align-items: center;
#   gap: 10px;
#   flex-wrap: wrap;
#   justify-content: flex-end;
# }}
# .badge {{
#   display: inline-flex;
#   align-items: center;
#   gap: 8px;
#   border-radius: 999px;
#   padding: 6px 10px;
#   border: 1px solid var(--border);
#   background: rgba(31,42,68,0.03);
#   font-size: 0.85rem;
#   color: var(--text);
# }}
# .badge-dot {{
#   width: 9px;
#   height: 9px;
#   border-radius: 50%;
#   background: var(--muted);
# }}
# .badge.success .badge-dot {{ background: var(--success); }}
# .badge.warn .badge-dot {{ background: var(--warn); }}
# .badge.danger .badge-dot {{ background: var(--danger); }}
# .kbd {{
#   font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
#   font-size: 0.82rem;
#   padding: 2px 6px;
#   border: 1px solid var(--border);
#   border-radius: 8px;
#   background: rgba(17,24,39,0.03);
#   color: var(--text);
# }}

# /* --- Cards --- */
# .card {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 16px 16px;
# }}
# .card h3 {{
#   margin: 0 0 6px 0;
#   font-size: 1.05rem;
# }}
# .card p {{
#   margin: 0;
#   color: var(--muted);
# }}

# /* --- Section headers --- */
# .section-title {{
#   margin: 18px 0 8px 0;
#   font-weight: 800;
#   font-size: 1.05rem;
#   letter-spacing: 0.2px;
# }}
# .section-sub {{
#   margin: -2px 0 10px 0;
#   color: var(--muted);
# }}

# /* --- File uploader polish --- */
# div[data-testid="stFileUploader"] {{
#   background: var(--panel);
#   border: 1px dashed rgba(17,24,39,0.25);
#   border-radius: var(--radius);
#   padding: 10px 10px 4px 10px;
# }}
# div[data-testid="stFileUploader"] section {{
#   padding: 6px 8px 10px 8px;
# }}
# div[data-testid="stFileUploader"] small {{
#   color: var(--muted) !important;
# }}

# /* --- Buttons --- */
# .stButton button {{
#   border-radius: 12px !important;
#   border: 1px solid var(--border) !important;
#   padding: 0.55rem 0.9rem !important;
#   font-weight: 650 !important;
# }}
# .stButton button:hover {{
#   border-color: rgba(31,42,68,0.35) !important;
# }}
# .primary-btn button {{
#   background: var(--primary) !important;
#   color: white !important;
#   border: 1px solid rgba(255,255,255,0.18) !important;
# }}
# .primary-btn button:hover {{
#   filter: brightness(1.05);
# }}
# .ghost-btn button {{
#   background: rgba(31,42,68,0.03) !important;
# }}

# /* --- Metrics-like tiles --- */
# .tiles {{
#   display: grid;
#   grid-template-columns: repeat(4, minmax(0, 1fr));
#   gap: 12px;
# }}
# @media (max-width: 1100px) {{
#   .tiles {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
# }}
# .tile {{
#   background: var(--panel);
#   border: 1px solid var(--border);
#   border-radius: var(--radius);
#   box-shadow: var(--shadow);
#   padding: 12px 14px;
# }}
# .tile .label {{
#   color: var(--muted);
#   font-size: 0.82rem;
# }}
# .tile .value {{
#   font-size: 1.25rem;
#   font-weight: 800;
#   margin-top: 2px;
# }}
# .tile .hint {{
#   color: var(--muted);
#   font-size: 0.82rem;
#   margin-top: 4px;
# }}

# /* --- Dataframe container polish --- */
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


# # -----------------------------
# # Simple state helpers
# # -----------------------------
# def ensure_state() -> None:
#     defaults = {
#         "nav": "Upload & Intake",
#         "system_status": "ready",  # ready | parsing | missing_plan | error
#         "context": {
#             "institution": "",
#             "site": "",
#             "machine": "MRIdian",
#             "reviewer": "",
#             "qa_type": "Routine",
#         },
#         "uploaded_meta": [],
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v


# ensure_state()


# # -----------------------------
# # Components
# # -----------------------------
# def topbar() -> None:
#     status = st.session_state["system_status"]
#     if status == "ready":
#         badge_class, badge_text = "success", "System Ready"
#     elif status == "parsing":
#         badge_class, badge_text = "warn", "Parsing Logs"
#     elif status == "missing_plan":
#         badge_class, badge_text = "warn", "Plan Reference Missing"
#     else:
#         badge_class, badge_text = "danger", "Action Required"

#     ctx = st.session_state["context"]
#     context_str = " • ".join([s for s in [ctx.get("institution"), ctx.get("site"), ctx.get("machine")] if s]) or "No context set"

#     st.markdown(
#         f"""
# <div class="topbar">
#   <div class="brand">
#     <div class="brand-title">MRIdian Log-Based QA Suite <span class="kbd">v1.0 (Research Build)</span></div>
#     <div class="brand-sub">Mechanical QA • Delivery Analytics • Predictive PSQA</div>
#   </div>
#   <div class="topbar-right">
#     <div class="badge {badge_class}"><span class="badge-dot"></span><span>{badge_text}</span></div>
#     <div class="badge"><span>{context_str}</span></div>
#   </div>
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# def sidebar() -> None:
#     st.sidebar.markdown("### MRIdian QA Suite")
#     st.sidebar.caption("Delivery-log–driven quality assurance framework")

#     nav = st.sidebar.radio(
#         "Navigation",
#         [
#             "Dashboard",
#             "Upload & Intake",
#             "Mechanical QA (Log-Based)",
#             "Longitudinal Trends",
#             "Reports & Export",
#             "Settings",
#         ],
#         index=[
#             "Dashboard",
#             "Upload & Intake",
#             "Mechanical QA (Log-Based)",
#             "Longitudinal Trends",
#             "Reports & Export",
#             "Settings",
#         ].index(st.session_state["nav"]),
#     )
#     st.session_state["nav"] = nav

#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Site & Machine Context")

#     ctx = st.session_state["context"]

#     ctx["institution"] = st.sidebar.text_input("Institution", value=ctx.get("institution", ""), placeholder="e.g., VCU Massey")
#     ctx["site"] = st.sidebar.text_input("Satellite Center / Site", value=ctx.get("site", ""), placeholder="e.g., Main Campus")
#     ctx["machine"] = st.sidebar.selectbox("Machine", options=["MRIdian", "MRIdian (TB-A)", "MRIdian (TB-B)"], index=0)
#     ctx["reviewer"] = st.sidebar.text_input("Reviewer", value=ctx.get("reviewer", ""), placeholder="Your name / initials")
#     ctx["qa_type"] = st.sidebar.selectbox("QA Type", options=["Routine", "Commissioning", "Investigation"], index=["Routine", "Commissioning", "Investigation"].index(ctx.get("qa_type", "Routine")))

#     st.sidebar.caption("Used for report labeling only.")
#     st.sidebar.markdown("---")
#     st.sidebar.caption("Research and QA use only • Not FDA cleared")


# @dataclass
# class UploadMeta:
#     file: str
#     size_mb: float
#     beams: int
#     total_mu: float
#     delivery_time: str


# def fake_parse_files(files: List[Any]) -> List[UploadMeta]:
#     # Replace this with your real log parsing later
#     st.session_state["system_status"] = "parsing"
#     time.sleep(0.35)
#     out: List[UploadMeta] = []
#     for f in files:
#         size_mb = round(len(f.getbuffer()) / (1024 * 1024), 2)
#         beams = max(1, int(size_mb) % 4 + 1)
#         total_mu = round(120 * beams + (size_mb * 5), 1)
#         delivery_time = f"{1 + beams:02d}:{(12 + int(size_mb * 3)) % 60:02d}"
#         out.append(UploadMeta(f.name, size_mb, beams, total_mu, delivery_time))
#     st.session_state["system_status"] = "ready"
#     return out


# def tiles_row(items: List[Dict[str, str]]) -> None:
#     html = '<div class="tiles">'
#     for it in items:
#         html += f"""
#         <div class="tile">
#           <div class="label">{it["label"]}</div>
#           <div class="value">{it["value"]}</div>
#           <div class="hint">{it.get("hint","")}</div>
#         </div>
#         """
#     html += "</div>"
#     st.markdown(html, unsafe_allow_html=True)


# def card(title: str, subtitle: str = "", body_html: str = "") -> None:
#     st.markdown(
#         f"""
# <div class="card">
#   <h3>{title}</h3>
#   <p>{subtitle}</p>
#   {body_html}
# </div>
# """,
#         unsafe_allow_html=True,
#     )


# # -----------------------------
# # Pages
# # -----------------------------
# def page_dashboard() -> None:
#     st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Operational status and recent activity</div>', unsafe_allow_html=True)

#     tiles_row(
#         [
#             {"label": "QA Runs (30d)", "value": "—", "hint": "Connect to your database later"},
#             {"label": "Latest Run", "value": datetime.utcnow().strftime("%Y-%m-%d"), "hint": "UTC time"},
#             {"label": "Mean Leaf Deviation", "value": "— mm", "hint": "Awaiting analysis"},
#             {"label": "QA Status", "value": "—", "hint": "PASS / Monitor / FAIL"},
#         ]
#     )

#     c1, c2 = st.columns([1.1, 0.9], gap="large")
#     with c1:
#         card(
#             "Modules",
#             "Mechanical QA, delivery analytics, and longitudinal trending (framework-ready).",
#             body_html="""
#             <div style="margin-top:12px; display:flex; gap:10px; flex-wrap:wrap;">
#               <span class="badge"><span class="badge-dot"></span><span>Log Intake</span></span>
#               <span class="badge"><span class="badge-dot"></span><span>MLC Positional QA</span></span>
#               <span class="badge"><span class="badge-dot"></span><span>Trend Analytics</span></span>
#               <span class="badge"><span class="badge-dot"></span><span>Report Export</span></span>
#             </div>
#             """,
#         )
#     with c2:
#         card(
#             "Compliance Notes",
#             "For research and QA use only. Not FDA cleared. Ensure local commissioning prior to clinical reliance.",
#         )


# def page_upload() -> None:
#     st.markdown('<div class="section-title">Upload & Intake</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Provide delivery logs and plan reference to initiate analysis</div>', unsafe_allow_html=True)

#     left, right = st.columns([1.1, 0.9], gap="large")

#     with left:
#         card(
#             "Delivery Log Files (.txt)",
#             "Supported: ViewRay MRIdian delivery logs • Max 200 MB per file",
#         )
#         files = st.file_uploader(
#             "Drag files here or browse",
#             type=["txt"],
#             accept_multiple_files=True,
#             label_visibility="collapsed",
#         )

#         colA, colB, colC = st.columns([1, 1, 1], gap="small")
#         with colA:
#             st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#             run_parse = st.button("Parse & Validate", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with colB:
#             st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#             clear = st.button("Clear", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)
#         with colC:
#             st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#             demo = st.button("Load Demo", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#         if clear:
#             st.session_state["uploaded_meta"] = []
#             st.session_state["system_status"] = "ready"
#             st.toast("Cleared.", icon="🧹")

#         if demo:
#             # Demo without files — just mock meta
#             st.session_state["uploaded_meta"] = [
#                 UploadMeta("DEMO_BeamA.txt", 4.20, 3, 520.0, "02:31").__dict__,
#                 UploadMeta("DEMO_BeamB.txt", 3.10, 2, 250.5, "03:08").__dict__,
#             ]
#             st.toast("Demo loaded.", icon="🧪")

#         if run_parse:
#             if not files:
#                 st.session_state["system_status"] = "error"
#                 st.error("No delivery logs detected. Please upload at least one MRIdian delivery log file to initiate analysis.")
#             else:
#                 with st.spinner("Parsing logs…"):
#                     meta = fake_parse_files(files)
#                 st.session_state["uploaded_meta"] = [m.__dict__ for m in meta]
#                 st.success("Parsing complete. Proceed to Mechanical QA module.")

#         # Render table if available
#         if st.session_state["uploaded_meta"]:
#             df = pd.DataFrame(st.session_state["uploaded_meta"])
#             df = df.rename(
#                 columns={
#                     "file": "File",
#                     "size_mb": "Size (MB)",
#                     "beams": "Beams",
#                     "total_mu": "Total MU",
#                     "delivery_time": "Delivery Time (mm:ss)",
#                 }
#             )
#             st.markdown('<div class="section-title">Detected Deliveries</div>', unsafe_allow_html=True)
#             st.dataframe(df, use_container_width=True, hide_index=True)

#     with right:
#         card(
#             "Plan Reference (Local)",
#             "Optional in this template. Connect to your RTPLAN selection/matching pipeline here.",
#         )

#         plan_mode = st.radio("Reference Type", ["RTPLAN (DICOM)", "Plan Summary PDF", "Manual Entry"], horizontal=True)
#         if plan_mode == "RTPLAN (DICOM)":
#             _ = st.file_uploader("Upload RTPLAN", type=["dcm"], accept_multiple_files=False)
#         elif plan_mode == "Plan Summary PDF":
#             _ = st.file_uploader("Upload Plan Summary PDF", type=["pdf"], accept_multiple_files=False)
#         else:
#             st.text_input("Plan ID / UID", placeholder="e.g., SOPInstanceUID")
#             st.text_input("Beam Names (comma-separated)", placeholder="Beam1, Beam2, Beam3")

#         st.markdown('<div class="section-title">Delivery Summary</div>', unsafe_allow_html=True)
#         meta = st.session_state["uploaded_meta"]
#         if meta:
#             total_beams = sum(m["beams"] for m in meta)
#             total_mu = sum(m["total_mu"] for m in meta)
#             tiles_row(
#                 [
#                     {"label": "Files", "value": str(len(meta)), "hint": "Parsed delivery logs"},
#                     {"label": "Total Beams", "value": str(total_beams), "hint": "Detected across logs"},
#                     {"label": "Total MU", "value": f"{total_mu:.1f}", "hint": "Aggregate MU"},
#                     {"label": "QA Status", "value": "Ready", "hint": "Proceed to analysis"},
#                 ]
#             )
#         else:
#             st.info("Upload and parse logs to populate the delivery summary.")


# def page_mechanical_qa() -> None:
#     st.markdown('<div class="section-title">Mechanical QA (Log-Based)</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">MLC positional accuracy evaluation from delivery records</div>', unsafe_allow_html=True)

#     left, right = st.columns([0.95, 1.05], gap="large")

#     with left:
#         card("Analysis Controls", "Configure tolerance, filtering, and weighting prior to analysis.")
#         tol = st.select_slider("Tolerance (mm)", options=[0.25, 0.5, 1.0, 1.5, 2.0], value=1.0)
#         weighting = st.toggle("MU-weighted statistics", value=True)
#         leaf_range = st.slider("Leaf range (index)", 1, 138, (1, 138))
#         seg_filter = st.multiselect("Segment filter", options=["All", "Static", "Dynamic"], default=["All"])

#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         run = st.button("Run Mechanical QA", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#         if run:
#             if not st.session_state["uploaded_meta"]:
#                 st.error("No parsed logs available. Go to Upload & Intake first.")
#                 return
#             with st.spinner("Computing MLC deviation metrics…"):
#                 time.sleep(0.45)
#             st.success(f"Analysis complete using ±{tol} mm tolerance.")

#         st.caption("Tip: wire this to your real parsing + deviation computation functions.")

#     with right:
#         # Placeholder visuals (replace with your plots/heatmaps)
#         card("Results Preview", "Replace with your deviation heatmap, histogram, and bank-wise metrics.")
#         tiles_row(
#             [
#                 {"label": "Mean |Δ|", "value": "0.21 mm", "hint": "Example output"},
#                 {"label": "Max |Δ|", "value": "0.78 mm", "hint": "Example output"},
#                 {"label": "95th pct |Δ|", "value": "0.42 mm", "hint": "Example output"},
#                 {"label": "QA Status", "value": "PASS", "hint": f"±{tol} mm"},
#             ]
#         )
#         st.info("Connect your actual plots here (matplotlib/plotly). Keep scales consistent and publication-ready.")


# def page_trends() -> None:
#     st.markdown('<div class="section-title">Longitudinal Trends</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Track drift, stability, and control limits over time</div>', unsafe_allow_html=True)

#     card(
#         "Trend Analytics",
#         "Add control charts, leaf drift heatmaps, and per-bank stability indices. This is where your tool outshines many commercial systems.",
#     )
#     st.info("Template page. Connect to your saved run database (CSV/SQL) and display time-series metrics.")


# def page_reports() -> None:
#     st.markdown('<div class="section-title">Reports & Export</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Generate standardized QA reports for documentation</div>', unsafe_allow_html=True)

#     card("Report Generator", "Export PDF/HTML reports with institutional header, summary, plots, and sign-off fields.")
#     c1, c2, c3 = st.columns([1, 1, 1], gap="small")
#     with c1:
#         st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
#         _ = st.button("Generate QA Report (PDF)", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with c2:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         _ = st.button("Export CSV", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)
#     with c3:
#         st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
#         _ = st.button("Export Figures", use_container_width=True)
#         st.markdown("</div>", unsafe_allow_html=True)

#     st.caption("Wire these buttons to your reportlab / matplotlib export functions when ready.")


# def page_settings() -> None:
#     st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-sub">Platform configuration and appearance</div>', unsafe_allow_html=True)

#     card("Appearance", "Edit theme colors and branding tokens (for production, store in config).")
#     st.code(f"THEME = {THEME}", language="python")
#     st.caption("For production: move theme/config to a YAML file and load at startup.")


# # -----------------------------
# # App render
# # -----------------------------
# sidebar()
# topbar()

# nav = st.session_state["nav"]
# if nav == "Dashboard":
#     page_dashboard()
# elif nav == "Upload & Intake":
#     page_upload()
# elif nav == "Mechanical QA (Log-Based)":
#     page_mechanical_qa()
# elif nav == "Longitudinal Trends":
#     page_trends()
# elif nav == "Reports & Export":
#     page_reports()
# else:
#     page_settings()


# # -----------------------------
# # Footer (subtle)
# # -----------------------------
# st.markdown(
#     """
# <div style="margin-top:20px; text-align:center; color:rgba(107,114,128,0.9); font-size:0.85rem;">
#   MRIdian Log-Based QA Suite • Research and QA use only • Not FDA cleared
# </div>
# """,
#     unsafe_allow_html=True,
# )


# # # core/report.py
# # from __future__ import annotations

# # import re
# # from datetime import datetime
# # from io import BytesIO
# # from pathlib import Path
# # from typing import Dict, Optional, Tuple, List

# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # from reportlab.lib import colors
# # from reportlab.lib.pagesizes import letter
# # from reportlab.lib.styles import getSampleStyleSheet
# # from reportlab.lib.units import inch
# # from reportlab.platypus import (
# #     Image,
# #     PageBreak,
# #     Paragraph,
# #     SimpleDocTemplate,
# #     Spacer,
# #     Table,
# #     TableStyle,
# # )

# # # =============================================================================
# # # Small utilities
# # # =============================================================================
# # def _safe_str(x, default: str = "N/A") -> str:
# #     if x is None:
# #         return default
# #     if isinstance(x, float) and np.isnan(x):
# #         return default
# #     s = str(x).strip()
# #     return s if s else default


# # def parse_log_date_to_ymd(date_str: object) -> Optional[str]:
# #     """
# #     Examples:
# #       'Jan 13 2026 17:52' -> '2026-01-13'
# #       'Jan 9 2026'        -> '2026-01-09'
# #     """
# #     if date_str is None or (isinstance(date_str, float) and np.isnan(date_str)):
# #         return None

# #     s = str(date_str).strip()
# #     m = re.match(r"^([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})", s)
# #     if not m:
# #         return None

# #     try:
# #         dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y")
# #         return dt.strftime("%Y-%m-%d")
# #     except Exception:
# #         return None


# # def _get_first(df: Optional[pd.DataFrame], col: str):
# #     if df is None or df.empty or col not in df.columns:
# #         return None
# #     try:
# #         return df[col].iloc[0]
# #     except Exception:
# #         return None


# # def _require_columns(df: Optional[pd.DataFrame], required: List[str], df_name: str) -> None:
# #     """Raise a clear error if required columns are missing."""
# #     if df is None or df.empty:
# #         return
# #     missing = [c for c in required if c not in df.columns]
# #     if missing:
# #         raise ValueError(f"{df_name} is missing required columns: {missing}")


# # # =============================================================================
# # # Gantry utilities (BINNED, per bank)
# # # =============================================================================
# # def gantry_to_bin(deg: object, step: float = 90.0) -> float:
# #     """Map gantry angle to nearest bin (default: 0/90/180/270)."""
# #     if pd.isna(deg):
# #         return np.nan
# #     d = float(deg) % 360.0
# #     b = (np.round(d / step) * step) % 360.0
# #     # normalize 360 -> 0
# #     if np.isclose(b, 360.0) or np.isclose(b, 0.0):
# #         return 0.0
# #     return float(b)


# # def _extract_binned_gantry_angles_for_bank(
# #     m_df: Optional[pd.DataFrame],
# #     step: float = 90.0,
# #     gantry_col_candidates: Tuple[str, ...] = ("GantryDeg", "GantryAngle", "Gantry"),
# # ) -> Tuple[List[float], str]:
# #     """
# #     Return (sorted_unique_binned_angles_deg, gantry_col_used_or_reason).
# #     - Uses only the provided bank dataframe (Upper or Lower).
# #     - Drops NaNs, normalizes, bins, then uniques.
# #     """
# #     if m_df is None or m_df.empty:
# #         return ([], "No data")

# #     gantry_col = None
# #     for c in gantry_col_candidates:
# #         if c in m_df.columns:
# #             gantry_col = c
# #             break
# #     if gantry_col is None:
# #         return ([], "Gantry column not found")

# #     vals = pd.to_numeric(m_df[gantry_col], errors="coerce").dropna().to_numpy()
# #     if vals.size == 0:
# #         return ([], f"{gantry_col} present but empty")

# #     vals = np.mod(vals.astype(float), 360.0)
# #     binned = np.array([gantry_to_bin(v, step=step) for v in vals], dtype=float)
# #     binned = binned[~np.isnan(binned)]
# #     if binned.size == 0:
# #         return ([], f"{gantry_col} produced no valid bins")

# #     uniq = np.unique(np.round(binned, 3))
# #     return (np.sort(uniq).tolist(), gantry_col)


# # def _format_bins(angles: List[float]) -> str:
# #     """Format binned angles: '0, 90, 180, 270'."""
# #     if not angles:
# #         return "N/A"
# #     parts = []
# #     for a in angles:
# #         parts.append(str(int(a)) if float(a).is_integer() else f"{a:g}")
# #     return ", ".join(parts)


# # # =============================================================================
# # # Metrics computation
# # # Expected input schema (mU/mL):
# # #   leaf_pair (int)
# # #   err_left_mm, err_right_mm
# # #   abs_err_left_mm, abs_err_right_mm
# # # =============================================================================
# # def compute_bank_summary(m_df: Optional[pd.DataFrame], bank: str) -> dict:
# #     if m_df is None or m_df.empty:
# #         return {
# #             "bank": bank,
# #             "n": 0,
# #             "rms_left_mm": np.nan,
# #             "rms_right_mm": np.nan,
# #             "mean_rms_mm": np.nan,
# #             "p95_abs_mm": np.nan,
# #             "max_abs_mm": np.nan,
# #         }

# #     abs_errs = np.concatenate(
# #         [m_df["abs_err_left_mm"].to_numpy(), m_df["abs_err_right_mm"].to_numpy()]
# #     )

# #     rms_left = float(np.sqrt(np.mean(m_df["err_left_mm"].to_numpy() ** 2)))
# #     rms_right = float(np.sqrt(np.mean(m_df["err_right_mm"].to_numpy() ** 2)))
# #     mean_rms = float((rms_left + rms_right) / 2.0)

# #     return {
# #         "bank": bank,
# #         "n": int(len(m_df)),
# #         "rms_left_mm": rms_left,
# #         "rms_right_mm": rms_right,
# #         "mean_rms_mm": mean_rms,
# #         "p95_abs_mm": float(np.percentile(abs_errs, 95)),
# #         "max_abs_mm": float(np.max(abs_errs)),
# #     }


# # def compute_per_leaf_rms(m_df: Optional[pd.DataFrame], leaf_col: str = "leaf_pair") -> pd.DataFrame:
# #     if m_df is None or m_df.empty:
# #         return pd.DataFrame(columns=["mlc_leaf", "rms_left_mm", "rms_right_mm"])

# #     g = m_df.groupby(leaf_col, as_index=True)
# #     out = pd.DataFrame(
# #         {
# #             "mlc_leaf": g.size().index.astype(int),
# #             "rms_left_mm": g["err_left_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).values,
# #             "rms_right_mm": g["err_right_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2)))).values,
# #         }
# #     ).sort_values("mlc_leaf").reset_index(drop=True)

# #     return out


# # def compute_per_leaf_maxabs(m_df: Optional[pd.DataFrame], leaf_col: str = "leaf_pair") -> pd.DataFrame:
# #     if m_df is None or m_df.empty:
# #         return pd.DataFrame(columns=["mlc_leaf", "maxabs_left_mm", "maxabs_right_mm"])

# #     out = (
# #         m_df.groupby(leaf_col, as_index=False)
# #         .agg(
# #             maxabs_left_mm=("abs_err_left_mm", "max"),
# #             maxabs_right_mm=("abs_err_right_mm", "max"),
# #         )
# #         .rename(columns={leaf_col: "mlc_leaf"})
# #     )
# #     out["mlc_leaf"] = out["mlc_leaf"].astype(int)
# #     return out.sort_values("mlc_leaf").reset_index(drop=True)


# # def classify_status(
# #     mean_rms_mm: float,
# #     max_abs_mm: float,
# #     warn_rms: float = 0.8,
# #     fail_rms: float = 1.2,
# #     warn_max: float = 1.5,
# #     fail_max: float = 2.0,
# # ) -> str:
# #     if np.isnan(mean_rms_mm) or np.isnan(max_abs_mm):
# #         return "UNKNOWN"
# #     if (max_abs_mm >= fail_max) or (mean_rms_mm >= fail_rms):
# #         return "FAIL"
# #     if (max_abs_mm >= warn_max) or (mean_rms_mm >= warn_rms):
# #         return "WARN"
# #     return "PASS"


# # # =============================================================================
# # # Plot helpers (in-memory PNG for PDF embedding)
# # # =============================================================================
# # def _plot_rms_bytes(rms_df: pd.DataFrame, title: str) -> BytesIO:
# #     buf = BytesIO()
# #     fig = plt.figure(figsize=(7, 4))

# #     if rms_df is None or rms_df.empty:
# #         plt.text(0.5, 0.5, "No data", ha="center", va="center")
# #         plt.axis("off")
# #     else:
# #         plt.plot(
# #             rms_df["mlc_leaf"],
# #             rms_df["rms_left_mm"],
# #             marker="o",
# #             markersize=3,
# #             linewidth=1,
# #             label="Y1 (Left)",
# #         )
# #         plt.plot(
# #             rms_df["mlc_leaf"],
# #             rms_df["rms_right_mm"],
# #             marker="x",
# #             markersize=3,
# #             linewidth=1,
# #             label="Y2 (Right)",
# #         )
# #         plt.xlabel("MLC Leaf")
# #         plt.ylabel("RMS of MLC Position Error (mm)")
# #         ymax = rms_df[["rms_left_mm", "rms_right_mm"]].to_numpy().max()
# #         plt.ylim(0, max(1.0, float(ymax) * 1.2))
# #         plt.grid(True, alpha=0.3)
# #         plt.legend()

# #     plt.title(title)
# #     plt.tight_layout()
# #     fig.savefig(buf, format="png", dpi=200)
# #     plt.close(fig)
# #     buf.seek(0)
# #     return buf


# # def _plot_maxabs_bytes(maxabs_df: pd.DataFrame, title: str) -> BytesIO:
# #     buf = BytesIO()
# #     fig = plt.figure(figsize=(7, 4))

# #     if maxabs_df is None or maxabs_df.empty:
# #         plt.text(0.5, 0.5, "No data", ha="center", va="center")
# #         plt.axis("off")
# #     else:
# #         plt.plot(
# #             maxabs_df["mlc_leaf"],
# #             maxabs_df["maxabs_left_mm"],
# #             marker="o",
# #             markersize=3,
# #             linewidth=1,
# #             label="Y1 (Left)",
# #         )
# #         plt.plot(
# #             maxabs_df["mlc_leaf"],
# #             maxabs_df["maxabs_right_mm"],
# #             marker="x",
# #             markersize=3,
# #             linewidth=1,
# #             label="Y2 (Right)",
# #         )
# #         plt.xlabel("MLC Leaf")
# #         plt.ylabel("Max abs MLC Position Error (mm)")
# #         ymax = maxabs_df[["maxabs_left_mm", "maxabs_right_mm"]].to_numpy().max()
# #         plt.ylim(0, max(2.0, float(ymax) * 1.2))
# #         plt.grid(True, alpha=0.3)
# #         plt.legend()

# #     plt.title(title)
# #     plt.tight_layout()
# #     fig.savefig(buf, format="png", dpi=200)
# #     plt.close(fig)
# #     buf.seek(0)
# #     return buf


# # # =============================================================================
# # # Trending CSV (NO gantry raw/bin)
# # # =============================================================================
# # def update_trending_csv(trend_csv: Path, summary_rows: list[dict]) -> pd.DataFrame:
# #     trend_csv = Path(trend_csv)
# #     trend_csv.parent.mkdir(parents=True, exist_ok=True)

# #     today_df = pd.DataFrame(summary_rows)

# #     if trend_csv.exists():
# #         all_df = pd.read_csv(trend_csv)
# #         all_df = pd.concat([all_df, today_df], ignore_index=True)
# #     else:
# #         all_df = today_df.copy()

# #     key_cols = [c for c in ["Date", "Machine", "bank"] if c in all_df.columns]
# #     if key_cols:
# #         all_df = all_df.drop_duplicates(subset=key_cols, keep="last")

# #     all_df.to_csv(trend_csv, index=False)
# #     return all_df


# # # =============================================================================
# # # Public API
# # # =============================================================================
# # def generate_pdf_qa_report_bytes(
# #     mU: pd.DataFrame,
# #     mL: pd.DataFrame,
# #     report_title: str = "MRIdian MLC Positional QA Report",
# #     tolerances: Optional[Dict[str, float]] = None,
# #     trend_csv: Optional[Path] = None,
# #     site: Optional[str] = None,
# #     machine: Optional[str] = None,
# #     reviewer: Optional[str] = None,
# #     gantry_bin_step_deg: float = 90.0,
# # ) -> bytes:
# #     """
# #     Generate a multi-page PDF QA report and return it as bytes.

# #     Header includes:
# #       - Site / Machine / Reviewer (passed from app)
# #       - Patient Name / ID / Plan / Date (from merged dfs)
# #       - Gantry angles (°) – Upper Stack (binned)
# #       - Gantry angles (°) – Lower Stack (binned)

# #     Trending CSV update (optional) uses Machine but does NOT store gantry raw/bin.
# #     """
# #     if tolerances is None:
# #         tolerances = {"warn_rms": 0.8, "fail_rms": 1.2, "warn_max": 1.5, "fail_max": 2.0}

# #     # Validate required columns (fail early with a clear message)
# #     required_cols = ["err_left_mm", "err_right_mm", "abs_err_left_mm", "abs_err_right_mm", "leaf_pair"]
# #     _require_columns(mU, required_cols, "mU (upper)")
# #     _require_columns(mL, required_cols, "mL (lower)")

# #     # --- metadata (from merged dataframes) ---
# #     patient_name = _get_first(mU, "Patient Name") or _get_first(mL, "Patient Name")
# #     patient_id = _get_first(mU, "Patient ID") or _get_first(mL, "Patient ID")
# #     plan_name = _get_first(mU, "Plan Name") or _get_first(mL, "Plan Name")
# #     date_raw = _get_first(mU, "Date") or _get_first(mL, "Date")
# #     date_ymd = parse_log_date_to_ymd(date_raw) or _safe_str(date_raw, "UNKNOWN")

# #     # fallback machine from data if not passed
# #     if machine is None:
# #         machine = _get_first(mU, "Machine") or _get_first(mL, "Machine")

# #     # --- gantry bins per bank ---
# #     binsU, _ = _extract_binned_gantry_angles_for_bank(mU, step=gantry_bin_step_deg)
# #     binsL, _ = _extract_binned_gantry_angles_for_bank(mL, step=gantry_bin_step_deg)
# #     binsU_str = _format_bins(binsU)
# #     binsL_str = _format_bins(binsL)

# #     # --- summaries ---
# #     sumU = compute_bank_summary(mU, "upper")
# #     sumL = compute_bank_summary(mL, "lower")
# #     sumU["status"] = classify_status(sumU["mean_rms_mm"], sumU["max_abs_mm"], **tolerances)
# #     sumL["status"] = classify_status(sumL["mean_rms_mm"], sumL["max_abs_mm"], **tolerances)

# #     # --- per-leaf metrics (for plots) ---
# #     rmsU = compute_per_leaf_rms(mU)
# #     rmsL = compute_per_leaf_rms(mL)
# #     maxU = compute_per_leaf_maxabs(mU)
# #     maxL = compute_per_leaf_maxabs(mL)

# #     # --- optional trending update ---
# #     if trend_csv is not None:
# #         machine_for_csv = _safe_str(machine, default="MRIdian")
# #         rows = [
# #             {
# #                 "Date": date_ymd,
# #                 "Machine": machine_for_csv,
# #                 "bank": "upper",
# #                 "mean_rms_mm": sumU["mean_rms_mm"],
# #                 "p95_abs_mm": sumU["p95_abs_mm"],
# #                 "max_abs_mm": sumU["max_abs_mm"],
# #                 "status": sumU["status"],
# #             },
# #             {
# #                 "Date": date_ymd,
# #                 "Machine": machine_for_csv,
# #                 "bank": "lower",
# #                 "mean_rms_mm": sumL["mean_rms_mm"],
# #                 "p95_abs_mm": sumL["p95_abs_mm"],
# #                 "max_abs_mm": sumL["max_abs_mm"],
# #                 "status": sumL["status"],
# #             },
# #         ]
# #         update_trending_csv(Path(trend_csv), rows)

# #     # --- plots (PNG in memory) ---
# #     rmsU_png = _plot_rms_bytes(rmsU, "RMS of MLC Position Error - Upper")
# #     rmsL_png = _plot_rms_bytes(rmsL, "RMS of MLC Position Error - Lower")
# #     maxU_png = _plot_maxabs_bytes(maxU, "Max Abs MLC Position Error - Upper")
# #     maxL_png = _plot_maxabs_bytes(maxL, "Max Abs MLC Position Error - Lower")

# #     # --- build PDF ---
# #     styles = getSampleStyleSheet()
# #     styleN = styles["Normal"]
# #     styleH = styles["Heading2"]

# #     pdf_buf = BytesIO()

# #     def footer(canvas, doc):
# #         canvas.saveState()
# #         canvas.setFont("Helvetica", 9)
# #         canvas.setFillColor(colors.grey)
# #         canvas.drawString(0.75 * inch, 0.5 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
# #         canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc.page}")
# #         canvas.restoreState()

# #     doc = SimpleDocTemplate(
# #         pdf_buf,
# #         pagesize=letter,
# #         leftMargin=0.75 * inch,
# #         rightMargin=0.75 * inch,
# #         topMargin=0.75 * inch,
# #         bottomMargin=0.75 * inch,
# #     )

# #     story = []
# #     story.append(Paragraph(report_title, styles["Title"]))
# #     story.append(Spacer(1, 0.15 * inch))

# #     # --- Header table (includes Site/Machine/Reviewer) ---
# #     header_data = [
# #         ["Site", _safe_str(site)],
# #         ["Machine", _safe_str(machine, default="MRIdian")],
# #         ["Reviewer", _safe_str(reviewer)],
# #         ["Patient Name", _safe_str(patient_name)],
# #         ["Patient ID", _safe_str(patient_id)],
# #         ["Plan Name", _safe_str(plan_name)],
# #         ["Date", _safe_str(date_ymd)],
# #         ["Gantry angles (°) – Upper Stack", binsU_str],
# #         ["Gantry angles (°) – Lower Stack", binsL_str],
# #     ]
# #     t = Table(header_data, colWidths=[2.2 * inch, 4.8 * inch])
# #     t.setStyle(
# #         TableStyle(
# #             [
# #                 ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
# #                 ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
# #                 ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
# #                 ("FONTSIZE", (0, 0), (-1, -1), 10),
# #                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
# #             ]
# #         )
# #     )
# #     story.append(t)
# #     story.append(Spacer(1, 0.2 * inch))

# #     # --- Summary table ---
# #     story.append(Paragraph("Summary Metrics", styleH))
# #     summary_tbl = pd.DataFrame([sumU, sumL])[["bank", "n", "mean_rms_mm", "p95_abs_mm", "max_abs_mm", "status"]]
# #     summary_tbl = summary_tbl.round({"mean_rms_mm": 3, "p95_abs_mm": 3, "max_abs_mm": 3})
# #     data = [summary_tbl.columns.tolist()] + summary_tbl.values.tolist()

# #     tt = Table(
# #         data,
# #         colWidths=[1.0 * inch, 0.8 * inch, 1.3 * inch, 1.2 * inch, 1.2 * inch, 0.9 * inch],
# #     )
# #     tt.setStyle(
# #         TableStyle(
# #             [
# #                 ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
# #                 ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
# #                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
# #                 ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
# #                 ("FONTSIZE", (0, 0), (-1, -1), 9),
# #                 ("ALIGN", (1, 1), (-2, -1), "CENTER"),
# #                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
# #             ]
# #         )
# #     )
# #     story.append(tt)
# #     story.append(Spacer(1, 0.2 * inch))

# #     # --- Plots ---
# #     story.append(Paragraph("Per-leaf Performance", styleH))
# #     story.append(Paragraph("RMS and Max Abs error per leaf are shown for upper and lower banks.", styleN))
# #     story.append(Spacer(1, 0.15 * inch))

# #     img_w = 6.8 * inch
# #     img_h = 3.6 * inch

# #     story.append(Image(rmsU_png, width=img_w, height=img_h))
# #     story.append(Spacer(1, 0.1 * inch))
# #     story.append(Image(rmsL_png, width=img_w, height=img_h))
# #     story.append(PageBreak())

# #     story.append(Paragraph("Per-leaf Maximum Absolute Error", styleH))
# #     story.append(Spacer(1, 0.1 * inch))
# #     story.append(Image(maxU_png, width=img_w, height=img_h))
# #     story.append(Spacer(1, 0.1 * inch))
# #     story.append(Image(maxL_png, width=img_w, height=img_h))

# #     doc.build(story, onFirstPage=footer, onLaterPages=footer)

# #     pdf_buf.seek(0)
# #     return pdf_buf.getvalue()


# # def generate_pdf_qa_report(
# #     output_pdf: Path,
# #     mU: pd.DataFrame,
# #     mL: pd.DataFrame,
# #     trend_csv: Optional[Path] = None,
# #     report_title: str = "MRIdian MLC Positional QA Report",
# #     tolerances: Optional[Dict[str, float]] = None,
# #     site: Optional[str] = None,
# #     machine: Optional[str] = None,
# #     reviewer: Optional[str] = None,
# #     gantry_bin_step_deg: float = 90.0,
# # ) -> Path:
# #     """
# #     Convenience wrapper: writes the PDF to disk (and optionally updates trending CSV).
# #     Returns output_pdf.
# #     """
# #     output_pdf = Path(output_pdf)
# #     output_pdf.parent.mkdir(parents=True, exist_ok=True)

# #     pdf_bytes = generate_pdf_qa_report_bytes(
# #         mU=mU,
# #         mL=mL,
# #         report_title=report_title,
# #         tolerances=tolerances,
# #         trend_csv=trend_csv,
# #         site=site,
# #         machine=machine,
# #         reviewer=reviewer,
# #         gantry_bin_step_deg=gantry_bin_step_deg,
# #     )
# #     output_pdf.write_bytes(pdf_bytes)
# #     return output_pdf
