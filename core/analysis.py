# core/analysis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates


# =============================================================================
# Constants / utils
# =============================================================================
# For MaxAbs criteria:
#   PASS:    <= warn_mm
#   WARNING: (warn_mm, fail_mm] mm
#   ACTION:  >  fail_mm
DEFAULT_WARN_MM = 0.5   # WARNING threshold
DEFAULT_FAIL_MM = 1.0   # ACTION threshold


def _require_columns(df: pd.DataFrame, cols: List[str], fn: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fn}: missing required columns {missing}")


def _to_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _safe_date_ymd_from_any(x: Any) -> Optional[str]:
    """
    Convert anything date-like to YYYY-MM-DD.
    Returns None if not parseable.
    """
    if x is None:
        return None
    try:
        dt_ = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt_):
            return pd.Timestamp(dt_).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


# =============================================================================
# Plot helpers (bands, ticks, padding)
# =============================================================================
def _add_threshold_bands(
    ax: plt.Axes,
    warn_mm: float,
    fail_mm: float,
    *,
    color_pass: str = "#2ca02c",
    color_warn: str = "#ff7f0e",
    color_fail: str = "#d62728",
    alpha_pass: float = 0.10,
    alpha_warn: float = 0.10,
    alpha_fail: float = 0.08,
) -> None:
    """
    Shaded PASS/WARN/FAIL bands using the current y-limits.
    """
    warn_mm = float(warn_mm)
    fail_mm = float(fail_mm)

    y0, y1 = ax.get_ylim()
    ymin, ymax = (y0, y1) if y0 < y1 else (y1, y0)

    # PASS zone: [ymin, warn]
    ax.axhspan(ymin, warn_mm, alpha=alpha_pass, color=color_pass, zorder=0)
    # WARN zone: (warn, fail]
    ax.axhspan(warn_mm, fail_mm, alpha=alpha_warn, color=color_warn, zorder=0)
    # FAIL zone: (fail, ymax]
    ax.axhspan(fail_mm, ymax, alpha=alpha_fail, color=color_fail, zorder=0)

    ax.set_ylim(y0, y1)


def _nice_ytick_step(span: float, target_ticks: int = 6) -> float:
    span = float(span)
    if not np.isfinite(span) or span <= 0:
        return 0.2
    raw = span / max(2, int(target_ticks))
    base = 10 ** np.floor(np.log10(raw))
    for m in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
        step = m * base
        if step >= raw:
            return float(step)
    return float(10 * base)


def _y_bottom_pad(ymax: float, frac: float = 0.06, abs_min: float = 0.02) -> float:
    """
    Small negative padding below zero for nicer visuals.
    """
    ymax = float(ymax)
    if not np.isfinite(ymax) or ymax <= 0:
        return -abs_min
    return -max(abs_min, frac * ymax)


def _set_y_ticks_nice_range(ax: plt.Axes, ymin: float, ymax: float) -> None:
    step = _nice_ytick_step((ymax - ymin) if ymax > ymin else ymax)
    start = np.floor(float(ymin) / step) * step
    ax.set_yticks(np.arange(start, float(ymax) + step, step))


def _set_smart_date_axis(ax: plt.Axes) -> None:
    """
    Cleaner x-axis for dates (prevents ugly labels).
    """
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", labelsize=10, pad=6)


def _set_smart_xlim(ax: plt.Axes, x: pd.Series | np.ndarray) -> None:
    """
    Prevents years of empty x-range when only one point exists.
    """
    if len(x) == 0:
        return
    xmin = pd.Timestamp(np.min(x))
    xmax = pd.Timestamp(np.max(x))
    if xmin == xmax:
        pad = pd.Timedelta(days=14)
        ax.set_xlim(xmin - pad, xmax + pad)
    else:
        pad = (xmax - xmin) * 0.05
        ax.set_xlim(xmin - pad, xmax + pad)


def top_worst_leaves(
    df: pd.DataFrame,
    n: int = 5,
    mode: str = "max",  # "max" or "rms"
) -> pd.DataFrame:
    out = df.copy()
    if mode == "max":
        _require_columns(out, ["mlc_leaf", "max_abs_left", "max_abs_right"], "top_worst_leaves(mode='max')")
        out["worst_mm"] = np.nanmax(out[["max_abs_left", "max_abs_right"]].to_numpy(dtype=float), axis=1)
    elif mode == "rms":
        _require_columns(out, ["mlc_leaf", "rms_left_mm", "rms_right_mm"], "top_worst_leaves(mode='rms')")
        out["worst_mm"] = np.nanmax(out[["rms_left_mm", "rms_right_mm"]].to_numpy(dtype=float), axis=1)
    else:
        raise ValueError("top_worst_leaves(): mode must be 'max' or 'rms'")

    out = out.sort_values("worst_mm", ascending=False).head(int(n)).reset_index(drop=True)
    return out[["mlc_leaf", "worst_mm"]]


# =============================================================================
# Errors
# =============================================================================
def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _require_columns(df, ["left_mm", "right_mm", "left_mm_nom", "right_mm_nom"], "add_errors()")

    left = _to_float_series(df["left_mm"])
    right = _to_float_series(df["right_mm"])
    left_nom = _to_float_series(df["left_mm_nom"])
    right_nom = _to_float_series(df["right_mm_nom"])

    err_left = left - left_nom
    err_right = right - right_nom

    df["err_left_mm"] = err_left
    df["err_right_mm"] = err_right
    df["abs_err_left_mm"] = np.abs(err_left)
    df["abs_err_right_mm"] = np.abs(err_right)
    df["abs_err_max_mm"] = np.maximum(
        df["abs_err_left_mm"].to_numpy(dtype=float),
        df["abs_err_right_mm"].to_numpy(dtype=float),
    )
    return df


def error_describe(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["err_left_mm", "err_right_mm"], "error_describe()")
    return df[["err_left_mm", "err_right_mm"]].describe()


# =============================================================================
# Leaf metrics
# =============================================================================
def leaf_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    _require_columns(
        df,
        ["leaf_pair", "err_left_mm", "err_right_mm", "abs_err_left_mm", "abs_err_right_mm"],
        "leaf_metrics()",
    )

    g = df.groupby("leaf_pair", dropna=True)

    def _rms(x: pd.Series) -> float:
        a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        return float(np.sqrt(np.mean(a**2))) if a.size else np.nan

    out = (
        g.agg(
            max_abs_left_mm=("abs_err_left_mm", "max"),
            rms_left_mm=("err_left_mm", _rms),
            max_abs_right_mm=("abs_err_right_mm", "max"),
            rms_right_mm=("err_right_mm", _rms),
        )
        .reset_index()
    )
    out["table"] = str(label)
    return out


def classify(metrics_df, warn_mm=None, fail_mm=None, *args, **kwargs):
    """
    Robust PASS/WARN/FAIL classifier based on MAX(max_abs_left_mm, max_abs_right_mm).
    """
    if warn_mm is None:
        warn_mm = kwargs.pop("warn", None)
    if fail_mm is None:
        fail_mm = kwargs.pop("fail", None)

    if warn_mm is None:
        warn_mm = kwargs.pop("tolerance_mm", None)
    if fail_mm is None:
        fail_mm = kwargs.pop("action_mm", None)

    if (warn_mm is None or fail_mm is None) and len(args) >= 2:
        if warn_mm is None:
            warn_mm = args[0]
        if fail_mm is None:
            fail_mm = args[1]

    warn_mm = DEFAULT_WARN_MM if warn_mm is None else float(warn_mm)
    fail_mm = DEFAULT_FAIL_MM if fail_mm is None else float(fail_mm)

    required = ["max_abs_left_mm", "max_abs_right_mm"]
    missing = [c for c in required if c not in metrics_df.columns]
    if missing:
        raise ValueError(f"classify(): missing columns {missing}")

    m = float(np.nanmax(metrics_df[["max_abs_left_mm", "max_abs_right_mm"]].to_numpy(dtype=float)))

    if not np.isfinite(m):
        return "UNKNOWN", np.nan
    if m > fail_mm:
        return "FAIL", m
    if m > warn_mm:
        return "WARN", m
    return "PASS", m


# =============================================================================
# Virtual picket fence plot
# =============================================================================
def compute_gap_df_from_merged(m_df: pd.DataFrame, use_nominal: bool = False) -> pd.DataFrame:
    df = m_df.copy()
    if use_nominal:
        _require_columns(df, ["leaf_pair", "left_mm_nom", "right_mm_nom"], "compute_gap_df_from_merged()")
        gap = _to_float_series(df["right_mm_nom"]) - _to_float_series(df["left_mm_nom"])
    else:
        _require_columns(df, ["leaf_pair", "left_mm", "right_mm"], "compute_gap_df_from_merged()")
        gap = _to_float_series(df["right_mm"]) - _to_float_series(df["left_mm"])

    out = (
        pd.DataFrame({"leaf_pair": pd.to_numeric(df["leaf_pair"], errors="coerce"), "gap_mm": gap})
        .dropna(subset=["leaf_pair"])
        .assign(leaf_pair=lambda d: d["leaf_pair"].astype(int))
        .groupby("leaf_pair", as_index=False)
        .agg(gap_mm=("gap_mm", "mean"))
    )
    return out


def plot_virtual_picket_fence(
    picket_centers_mm: List[float],
    gap_df: pd.DataFrame,
    title: str,
    xlim_mm: Tuple[float, float] = (-110, 110),
    xticks: Optional[np.ndarray] = None,
    ytick_step: int = 1,
) -> plt.Figure:
    _require_columns(gap_df, ["leaf_pair", "gap_mm"], "plot_virtual_picket_fence()")

    leaves = np.sort(pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).unique())
    if leaves.size == 0:
        raise ValueError("plot_virtual_picket_fence(): gap_df has no leaves")

    y0, y1 = int(leaves.min()), int(leaves.max())
    gap_map = dict(
        zip(
            pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).tolist(),
            pd.to_numeric(gap_df["gap_mm"], errors="coerce").to_numpy(dtype=float).tolist(),
        )
    )

    fig, ax = plt.subplots(figsize=(6.4, 7.2), dpi=150)
    ax.set_title(title, fontweight="bold")

    for y in range(y0, y1 + 1):
        w = float(gap_map.get(y, np.nan))
        if not np.isfinite(w) or w <= 0:
            continue

        for xc in picket_centers_mm:
            ax.add_patch(
                Rectangle(
                    (float(xc) - w / 2.0, y - 0.5),
                    w,
                    1.0,
                    facecolor="black",
                    edgecolor="black",
                    linewidth=0,
                )
            )
        ax.axhline(y, linewidth=0.4, alpha=0.18)

    ax.set_xlim(*xlim_mm)
    ax.set_ylim(y1 + 1, y0 - 1)  # invert
    ax.set_xlabel("Distance (mm)")
    ax.set_ylabel("Leaf index")

    if xticks is not None:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks(np.arange(-100, 101, 20))

    ax.set_yticks(np.arange(y0, y1 + 1, max(1, int(ytick_step))))
    ax.grid(False)
    fig.tight_layout()
    return fig


# =============================================================================
# Combined leaf indexing + MaxAbs plot helper
# =============================================================================
def make_combined_leaf_index(mU: pd.DataFrame, mL: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    _require_columns(mU, ["leaf_pair"], "make_combined_leaf_index(mU)")
    _require_columns(mL, ["leaf_pair"], "make_combined_leaf_index(mL)")

    U = mU.copy()
    L = mL.copy()

    upper_pairs = sorted(pd.to_numeric(U["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())
    lower_pairs = sorted(pd.to_numeric(L["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())

    U_map = {p: i + 1 for i, p in enumerate(upper_pairs)}
    L_map = {p: i + 1 + len(upper_pairs) for i, p in enumerate(lower_pairs)}

    U["leaf_pair"] = pd.to_numeric(U["leaf_pair"], errors="coerce").astype("Int64")
    L["leaf_pair"] = pd.to_numeric(L["leaf_pair"], errors="coerce").astype("Int64")

    U["mlc_leaf"] = U["leaf_pair"].map(U_map)
    L["mlc_leaf"] = L["leaf_pair"].map(L_map)

    comb = pd.concat([U, L], ignore_index=True)

    sort_cols = ["mlc_leaf"]
    if "match_id" in comb.columns:
        sort_cols.append("match_id")

    comb = comb.sort_values(sort_cols).reset_index(drop=True)
    return comb, U_map, L_map


def max_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["mlc_leaf", "err_left_mm", "err_right_mm"], "max_by_leaf()")

    g = df.groupby("mlc_leaf", dropna=True)

    def _maxabs(x: pd.Series) -> float:
        a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        return float(np.max(np.abs(a))) if a.size else np.nan

    out = pd.DataFrame(
        {
            "mlc_leaf": g.size().index.to_numpy(dtype=int),
            "max_abs_left": g["err_left_mm"].apply(_maxabs).to_numpy(dtype=float),
            "max_abs_right": g["err_right_mm"].apply(_maxabs).to_numpy(dtype=float),
        }
    )
    return out.sort_values("mlc_leaf").reset_index(drop=True)


def plot_max_errors_by_leaf(
    mx_df: pd.DataFrame,
    threshold_mm: float = DEFAULT_WARN_MM,  # WARNING
    title: str = "Max Abs Error by Leaf",
    fail_mm: Optional[float] = None,        # ACTION
    show_bands: bool = True,
    label_worst_n: int = 0,
) -> plt.Figure:
    _require_columns(mx_df, ["mlc_leaf", "max_abs_left", "max_abs_right"], "plot_max_errors_by_leaf()")

    warn_mm = float(threshold_mm)
    action_mm = float(fail_mm) if fail_mm is not None else None

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=150)

    ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_left"], marker="o", markersize=3, linewidth=1.6, label="Left Bank")
    ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_right"], marker="o", markersize=3, linewidth=1.6, label="Right Bank")

    ax.axhline(warn_mm, linewidth=1.3, linestyle="--", label=f"Warning = {warn_mm:.2f} mm")
    if action_mm is not None and np.isfinite(action_mm):
        ax.axhline(action_mm, linewidth=1.3, linestyle="-.", label=f"Action = {action_mm:.2f} mm")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Leaf index")
    ax.set_ylabel("Max abs error (mm)")
    ax.grid(True, alpha=0.22)

    vals = pd.to_numeric(
        pd.concat([mx_df["max_abs_left"], mx_df["max_abs_right"]]),
        errors="coerce",
    ).to_numpy(dtype=float)
    vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else warn_mm
    ref_top = max(vmax * 1.15, warn_mm * 1.25, (action_mm or 0.0) * 1.25)

    ybottom = _y_bottom_pad(ref_top)
    ax.set_ylim(ybottom, ref_top)
    _set_y_ticks_nice_range(ax, ybottom, ref_top)

    if show_bands and action_mm is not None and np.isfinite(action_mm):
        _add_threshold_bands(ax, warn_mm, action_mm)

    if int(label_worst_n) > 0:
        tmp = mx_df.copy()
        tmp["worst"] = np.nanmax(tmp[["max_abs_left", "max_abs_right"]].to_numpy(dtype=float), axis=1)
        tmp = tmp.sort_values("worst", ascending=False).head(int(label_worst_n))
        for _, r in tmp.iterrows():
            x_ = float(r["mlc_leaf"])
            y_ = float(r["worst"])
            if np.isfinite(x_) and np.isfinite(y_):
                ax.annotate(
                    str(int(x_)),
                    xy=(x_, y_),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    alpha=0.85,
                )

    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
# Trending
# =============================================================================
def summarize_overall_max_error(m_df: pd.DataFrame, scope: str) -> Dict:
    """
    Creates one trend row with a robust YYYY-MM-DD date.
    Falls back to today's date if missing/unparseable (prevents blank/#### in Excel).
    """
    if m_df is None or m_df.empty:
        return {"Date": None, "GantryBin": np.nan, "scope": scope, "overall_max_abs_mm": np.nan, "n_points": 0}

    _require_columns(m_df, ["abs_err_left_mm", "abs_err_right_mm"], "summarize_overall_max_error()")

    # Robust date source: try common column names
    date_raw = None
    for candidate in ("Date", "DateTime", "Timestamp", "datetime", "time"):
        if candidate in m_df.columns:
            date_raw = m_df[candidate].iloc[0]
            break

    date_ymd = _safe_date_ymd_from_any(date_raw)

    # Fallback: set today's date so CSV always has valid Date
    if date_ymd is None:
        date_ymd = pd.Timestamp.today().strftime("%Y-%m-%d")

    gantry_bin = np.nan
    if "GantryBin" in m_df.columns:
        try:
            gantry_bin = float(pd.to_numeric(m_df["GantryBin"].iloc[0], errors="coerce"))
        except Exception:
            gantry_bin = np.nan

    overall_max = float(
        np.nanmax(
            np.r_[
                pd.to_numeric(m_df["abs_err_left_mm"], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(m_df["abs_err_right_mm"], errors="coerce").to_numpy(dtype=float),
            ]
        )
    )

    return {
        "Date": date_ymd,  # ALWAYS YYYY-MM-DD
        "GantryBin": gantry_bin,
        "scope": str(scope),
        "overall_max_abs_mm": overall_max,
        "n_points": int(len(m_df)),
    }


def append_trending_csv(
    trend_csv_path: Path,
    new_rows: List[Dict],
    dedup_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Appends rows to trend CSV, dedups by (Date, GantryBin, scope),
    forces Date to YYYY-MM-DD before saving, and aggregates duplicate dates
    (important if you run multiple tests per same day).
    """
    trend_csv_path = Path(trend_csv_path)
    trend_csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(new_rows)

    # Force date formatting in new rows
    if "Date" in new_df.columns:
        new_df["Date"] = pd.to_datetime(new_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if trend_csv_path.exists():
        old_df = pd.read_csv(trend_csv_path)
        if "Date" in old_df.columns:
            old_df["Date"] = pd.to_datetime(old_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        all_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_df = new_df.copy()

    if dedup_cols is None:
        dedup_cols = ["Date", "GantryBin", "scope"]

    for c in dedup_cols:
        if c not in all_df.columns:
            all_df[c] = np.nan

    # overall_max_abs_mm: take max (worst case)
    # n_points: sum
    if "overall_max_abs_mm" in all_df.columns:
        all_df["overall_max_abs_mm"] = pd.to_numeric(all_df["overall_max_abs_mm"], errors="coerce")
    if "n_points" in all_df.columns:
        all_df["n_points"] = pd.to_numeric(all_df["n_points"], errors="coerce")

    grouped = all_df.groupby(dedup_cols, dropna=False, as_index=False).agg(
        overall_max_abs_mm=("overall_max_abs_mm", "max"),
        n_points=("n_points", "sum"),
    )

    grouped["Date_dt"] = pd.to_datetime(grouped["Date"], errors="coerce")
    grouped = grouped.sort_values(["scope", "GantryBin", "Date_dt"]).reset_index(drop=True)
    grouped.drop(columns=["Date_dt"], inplace=True, errors="ignore")

    grouped.to_csv(trend_csv_path, index=False)
    return grouped


def plot_overall_max_trending(
    trend_df: pd.DataFrame,
    scope: str = "combined",
    gantry_bin: Optional[float] = None,
    fail_mm: float = DEFAULT_FAIL_MM,             # ACTION
    title: str = "Trending: Overall Max Absolute MLC Error",
    warn_mm: Optional[float] = DEFAULT_WARN_MM,   # WARNING
    show_bands: bool = True,
    annotate_last: bool = True,
    show_n: bool = True,
) -> plt.Figure:
    """
    Trending plot:
      - PASS/WARN/FAIL colored markers automatically
      - Shaded tolerance zones (green/yellow/red bands)
      - Smart x-limits (no multi-year axis when only 1 point)
      - Concise date formatting on x-axis
      - Cleaner grid/spines for a clinical look
      - Optional annotation of the last value and n= count
    """
    df = trend_df.copy()

    # Backward-compat (if you ever had different column names)
    if "overall_max_abs_mm" not in df.columns and "max_abs_mm" in df.columns:
        df["overall_max_abs_mm"] = df["max_abs_mm"]

    if "scope" in df.columns:
        df = df[df["scope"] == scope]

    if gantry_bin is not None and "GantryBin" in df.columns:
        df = df[df["GantryBin"] == gantry_bin]

    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df["overall_max_abs_mm"] = pd.to_numeric(df["overall_max_abs_mm"], errors="coerce")

    # If there are accidental duplicates, aggregate again here (safe)
    if {"Date_dt", "GantryBin", "scope"}.issubset(df.columns):
        key_cols = ["Date_dt", "GantryBin", "scope"]
        df = df.groupby(key_cols, as_index=False).agg(
            overall_max_abs_mm=("overall_max_abs_mm", "max"),
            n_points=("n_points", "sum") if "n_points" in df.columns else ("overall_max_abs_mm", "size"),
        )

    df = df.dropna(subset=["Date_dt", "overall_max_abs_mm"]).sort_values("Date_dt")

    fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=150)

    if df.empty:
        ax.set_title("No trend history found yet")
        ax.axis("off")
        fig.tight_layout()
        return fig

    x = df["Date_dt"]
    y = df["overall_max_abs_mm"].to_numpy(dtype=float)

    warn = float(warn_mm) if warn_mm is not None and np.isfinite(float(warn_mm)) else None
    fail = float(fail_mm)

    # Status classification
    status = np.full(y.shape, "FAIL", dtype=object)
    if warn is not None:
        status[y <= fail] = "WARN"
        status[y <= warn] = "PASS"
    else:
        status[y <= fail] = "PASS"

    colors = {"PASS": "#2ca02c", "WARN": "#ff7f0e", "FAIL": "#d62728"}

    # Run line (neutral + subtle)
    ax.plot(x, y, linewidth=1.9, alpha=0.40, label="Run (line)")

    # Colored markers
    for key in ("PASS", "WARN", "FAIL"):
        idx = np.where(status == key)[0]
        if idx.size == 0:
            continue
        ax.scatter(
            x.iloc[idx],
            y[idx],
            s=70,
            color=colors[key],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
            label=key,
        )

    # Threshold lines
    if warn is not None:
        ax.axhline(warn, linestyle="--", linewidth=1.5, color=colors["WARN"], alpha=0.95)
    ax.axhline(fail, linestyle="-.", linewidth=1.5, color=colors["FAIL"], alpha=0.95)

    # Y limits
    ymax = float(np.nanmax(y))
    top = max(ymax * 1.15, fail * 1.25, (warn * 1.25) if warn is not None else 0.0)
    ybottom = _y_bottom_pad(top)
    ax.set_ylim(ybottom, top)
    _set_y_ticks_nice_range(ax, ybottom, top)

    if show_bands and warn is not None:
        _add_threshold_bands(
            ax,
            warn,
            fail,
            color_pass=colors["PASS"],
            color_warn=colors["WARN"],
            color_fail=colors["FAIL"],
        )

    # X axis cleanup
    _set_smart_xlim(ax, x.to_numpy())
    _set_smart_date_axis(ax)

    suffix = f" (Gantry {int(gantry_bin)})" if gantry_bin is not None and np.isfinite(gantry_bin) else ""
    ax.set_title(title + suffix, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Overall max abs error (mm)", fontsize=11)

    # Grid + spines
    ax.grid(True, axis="y", alpha=0.22)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate last point (value)
    if annotate_last and len(df) >= 1:
        x_last = x.iloc[-1]
        y_last = float(y[-1])
        ax.annotate(
            f"{y_last:.2f} mm",
            xy=(x_last, y_last),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # Show n= in axes coords
    if show_n:
        n_runs = int(len(df))
        ax.text(
            0.01,
            0.98,
            f"n = {n_runs}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="0.35",
        )

    # Legend order (only include what exists)
    handles, labels = ax.get_legend_handles_labels()
    desired = ["PASS", "WARN", "FAIL", "Run (line)"]

    seen = set()
    ordered: List[Tuple[Any, str]] = []
    for name in desired:
        for h, l in zip(handles, labels):
            if l == name and l not in seen:
                ordered.append((h, l))
                seen.add(l)

    if ordered:
        ax.legend([h for h, _ in ordered], [l for _, l in ordered], frameon=True, loc="upper right")
    else:
        ax.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    return fig


# # core/analysis.py
# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Any

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# # =============================================================================
# # Constants / utils
# # =============================================================================
# # For MaxAbs criteria:
# #   PASS:    <= 0.5 mm
# #   WARNING: (0.5, 1.0] mm
# #   ACTION:  >  1.0 mm
# DEFAULT_WARN_MM = 0.5   # WARNING threshold
# DEFAULT_FAIL_MM = 1.0   # ACTION threshold


# def _require_columns(df: pd.DataFrame, cols: List[str], fn: str) -> None:
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"{fn}: missing required columns {missing}")


# def _to_float_series(s: pd.Series) -> np.ndarray:
#     return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


# def _safe_date_ymd_from_any(x: Any) -> Optional[str]:
#     if x is None:
#         return None
#     try:
#         dt = pd.to_datetime(x, errors="coerce")
#         if pd.notna(dt):
#             return pd.Timestamp(dt).strftime("%Y-%m-%d")
#     except Exception:
#         pass
#     return None


# def _finite(a: np.ndarray) -> np.ndarray:
#     a = np.asarray(a, dtype=float)
#     return a[np.isfinite(a)]


# # =============================================================================
# # Plot helpers (bands, ticks, padding, tables)
# # =============================================================================
# def _add_threshold_bands(ax: plt.Axes, warn_mm: float, fail_mm: float) -> None:
#     """
#     Shaded PASS/WARN/ACTION bands.
#     Works even if y-axis starts slightly negative for visual padding.
#     """
#     warn_mm = float(warn_mm)
#     fail_mm = float(fail_mm)

#     y0, y1 = ax.get_ylim()
#     ymin, ymax = (y0, y1) if y0 < y1 else (y1, y0)

#     ax.axhspan(ymin, warn_mm, alpha=0.05)
#     ax.axhspan(warn_mm, fail_mm, alpha=0.08)
#     ax.axhspan(fail_mm, ymax, alpha=0.10)

#     ax.set_ylim(y0, y1)


# def _nice_ytick_step(span: float, target_ticks: int = 6) -> float:
#     span = float(span)
#     if not np.isfinite(span) or span <= 0:
#         return 0.2
#     raw = span / max(2, int(target_ticks))
#     base = 10 ** np.floor(np.log10(raw))
#     for m in [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]:
#         step = m * base
#         if step >= raw:
#             return float(step)
#     return float(10 * base)


# def _y_bottom_pad(ymax: float, frac: float = 0.06, abs_min: float = 0.02) -> float:
#     """
#     Small negative padding below zero for nicer visuals.
#     Example: ymax=1.0 -> bottom ~ -0.06 (but at least -0.02)
#     """
#     ymax = float(ymax)
#     if not np.isfinite(ymax) or ymax <= 0:
#         return -abs_min
#     return -max(abs_min, frac * ymax)


# def _set_y_ticks_nice_range(ax: plt.Axes, ymin: float, ymax: float) -> None:
#     step = _nice_ytick_step(ymax - ymin if ymax > ymin else ymax)
#     start = np.floor(float(ymin) / step) * step
#     ax.set_yticks(np.arange(start, float(ymax) + step, step))


# def top_worst_leaves(
#     df: pd.DataFrame,
#     n: int = 5,
#     mode: str = "max",  # "max" or "rms"
# ) -> pd.DataFrame:
#     out = df.copy()
#     if mode == "max":
#         _require_columns(out, ["mlc_leaf", "max_abs_left", "max_abs_right"], "top_worst_leaves(mode='max')")
#         out["worst_mm"] = np.nanmax(out[["max_abs_left", "max_abs_right"]].to_numpy(dtype=float), axis=1)
#     elif mode == "rms":
#         _require_columns(out, ["mlc_leaf", "rms_left_mm", "rms_right_mm"], "top_worst_leaves(mode='rms')")
#         out["worst_mm"] = np.nanmax(out[["rms_left_mm", "rms_right_mm"]].to_numpy(dtype=float), axis=1)
#     else:
#         raise ValueError("top_worst_leaves(): mode must be 'max' or 'rms'")

#     out = out.sort_values("worst_mm", ascending=False).head(int(n)).reset_index(drop=True)
#     return out[["mlc_leaf", "worst_mm"]]


# # =============================================================================
# # Errors
# # =============================================================================
# def add_errors(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     _require_columns(df, ["left_mm", "right_mm", "left_mm_nom", "right_mm_nom"], "add_errors()")

#     left = _to_float_series(df["left_mm"])
#     right = _to_float_series(df["right_mm"])
#     left_nom = _to_float_series(df["left_mm_nom"])
#     right_nom = _to_float_series(df["right_mm_nom"])

#     err_left = left - left_nom
#     err_right = right - right_nom

#     df["err_left_mm"] = err_left
#     df["err_right_mm"] = err_right
#     df["abs_err_left_mm"] = np.abs(err_left)
#     df["abs_err_right_mm"] = np.abs(err_right)
#     df["abs_err_max_mm"] = np.maximum(
#         df["abs_err_left_mm"].to_numpy(dtype=float),
#         df["abs_err_right_mm"].to_numpy(dtype=float),
#     )
#     return df


# def error_describe(df: pd.DataFrame) -> pd.DataFrame:
#     _require_columns(df, ["err_left_mm", "err_right_mm"], "error_describe()")
#     return df[["err_left_mm", "err_right_mm"]].describe()


# # =============================================================================
# # Leaf metrics
# # =============================================================================
# def leaf_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
#     _require_columns(
#         df,
#         ["leaf_pair", "err_left_mm", "err_right_mm", "abs_err_left_mm", "abs_err_right_mm"],
#         "leaf_metrics()",
#     )

#     g = df.groupby("leaf_pair", dropna=True)

#     def _rms(x: pd.Series) -> float:
#         a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
#         a = a[np.isfinite(a)]
#         return float(np.sqrt(np.mean(a**2))) if a.size else np.nan

#     out = (
#         g.agg(
#             max_abs_left_mm=("abs_err_left_mm", "max"),
#             rms_left_mm=("err_left_mm", _rms),
#             max_abs_right_mm=("abs_err_right_mm", "max"),
#             rms_right_mm=("err_right_mm", _rms),
#         )
#         .reset_index()
#     )
#     out["table"] = str(label)
#     return out


# def classify(metrics_df, warn_mm=None, fail_mm=None, *args, **kwargs):
#     """
#     Robust PASS/WARN/FAIL classifier based on MAX(max_abs_left_mm, max_abs_right_mm).

#     Preferred:
#       classify(df, warn_mm=0.5, fail_mm=1.0)

#     Accepts aliases / legacy:
#       classify(df, warn=..., fail=...)
#       classify(df, tolerance_mm=..., action_mm=...)  (alias)
#       classify(df, 0.5, 1.0)  (positional fallback)
#     """
#     # --- Resolve thresholds from kwargs / aliases / positional ---
#     if warn_mm is None:
#         warn_mm = kwargs.pop("warn", None)
#     if fail_mm is None:
#         fail_mm = kwargs.pop("fail", None)

#     if warn_mm is None:
#         warn_mm = kwargs.pop("tolerance_mm", None)
#     if fail_mm is None:
#         fail_mm = kwargs.pop("action_mm", None)

#     # Positional fallback: classify(df, warn, fail)
#     if (warn_mm is None or fail_mm is None) and len(args) >= 2:
#         if warn_mm is None:
#             warn_mm = args[0]
#         if fail_mm is None:
#             fail_mm = args[1]

#     warn_mm = DEFAULT_WARN_MM if warn_mm is None else float(warn_mm)
#     fail_mm = DEFAULT_FAIL_MM if fail_mm is None else float(fail_mm)

#     required = ["max_abs_left_mm", "max_abs_right_mm"]
#     missing = [c for c in required if c not in metrics_df.columns]
#     if missing:
#         raise ValueError(f"classify(): missing columns {missing}")

#     m = float(np.nanmax(metrics_df[["max_abs_left_mm", "max_abs_right_mm"]].to_numpy(dtype=float)))

#     if not np.isfinite(m):
#         return "UNKNOWN", np.nan
#     if m > fail_mm:
#         return "FAIL", m
#     if m > warn_mm:
#         return "WARN", m
#     return "PASS", m


# # =============================================================================
# # Virtual picket fence plot
# # =============================================================================
# def compute_gap_df_from_merged(m_df: pd.DataFrame, use_nominal: bool = False) -> pd.DataFrame:
#     df = m_df.copy()
#     if use_nominal:
#         _require_columns(df, ["leaf_pair", "left_mm_nom", "right_mm_nom"], "compute_gap_df_from_merged()")
#         gap = _to_float_series(df["right_mm_nom"]) - _to_float_series(df["left_mm_nom"])
#     else:
#         _require_columns(df, ["leaf_pair", "left_mm", "right_mm"], "compute_gap_df_from_merged()")
#         gap = _to_float_series(df["right_mm"]) - _to_float_series(df["left_mm"])

#     out = (
#         pd.DataFrame({"leaf_pair": pd.to_numeric(df["leaf_pair"], errors="coerce"), "gap_mm": gap})
#         .dropna(subset=["leaf_pair"])
#         .assign(leaf_pair=lambda d: d["leaf_pair"].astype(int))
#         .groupby("leaf_pair", as_index=False)
#         .agg(gap_mm=("gap_mm", "mean"))
#     )
#     return out


# def plot_virtual_picket_fence(
#     picket_centers_mm: List[float],
#     gap_df: pd.DataFrame,
#     title: str,
#     xlim_mm: Tuple[float, float] = (-110, 110),
#     xticks: Optional[np.ndarray] = None,
#     ytick_step: int = 1,
# ) -> plt.Figure:
#     _require_columns(gap_df, ["leaf_pair", "gap_mm"], "plot_virtual_picket_fence()")

#     leaves = np.sort(pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).unique())
#     if leaves.size == 0:
#         raise ValueError("plot_virtual_picket_fence(): gap_df has no leaves")

#     y0, y1 = int(leaves.min()), int(leaves.max())
#     gap_map = dict(
#         zip(
#             pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).tolist(),
#             pd.to_numeric(gap_df["gap_mm"], errors="coerce").to_numpy(dtype=float).tolist(),
#         )
#     )

#     fig, ax = plt.subplots(figsize=(6.4, 7.2))
#     ax.set_title(title)

#     for y in range(y0, y1 + 1):
#         w = float(gap_map.get(y, np.nan))
#         if not np.isfinite(w) or w <= 0:
#             continue

#         for xc in picket_centers_mm:
#             ax.add_patch(
#                 Rectangle(
#                     (float(xc) - w / 2.0, y - 0.5),
#                     w,
#                     1.0,
#                     facecolor="black",
#                     edgecolor="black",
#                     linewidth=0,
#                 )
#             )
#         ax.axhline(y, linewidth=0.4, alpha=0.25)

#     ax.set_xlim(*xlim_mm)
#     ax.set_ylim(y1 + 1, y0 - 1)  # invert
#     ax.set_xlabel("Distance (mm)")
#     ax.set_ylabel("Leaf index")

#     if xticks is not None:
#         ax.set_xticks(xticks)
#     else:
#         ax.set_xticks(np.arange(-100, 101, 20))

#     ax.set_yticks(np.arange(y0, y1 + 1, max(1, int(ytick_step))))
#     ax.grid(False)
#     fig.tight_layout()
#     return fig


# # =============================================================================
# # Combined leaf indexing + MaxAbs plot helper
# # =============================================================================
# def make_combined_leaf_index(mU: pd.DataFrame, mL: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
#     _require_columns(mU, ["leaf_pair"], "make_combined_leaf_index(mU)")
#     _require_columns(mL, ["leaf_pair"], "make_combined_leaf_index(mL)")

#     U = mU.copy()
#     L = mL.copy()

#     upper_pairs = sorted(pd.to_numeric(U["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())
#     lower_pairs = sorted(pd.to_numeric(L["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())

#     U_map = {p: i + 1 for i, p in enumerate(upper_pairs)}
#     L_map = {p: i + 1 + len(upper_pairs) for i, p in enumerate(lower_pairs)}

#     U["leaf_pair"] = pd.to_numeric(U["leaf_pair"], errors="coerce").astype("Int64")
#     L["leaf_pair"] = pd.to_numeric(L["leaf_pair"], errors="coerce").astype("Int64")

#     U["mlc_leaf"] = U["leaf_pair"].map(U_map)
#     L["mlc_leaf"] = L["leaf_pair"].map(L_map)

#     comb = pd.concat([U, L], ignore_index=True)

#     sort_cols = ["mlc_leaf"]
#     if "match_id" in comb.columns:
#         sort_cols.append("match_id")

#     comb = comb.sort_values(sort_cols).reset_index(drop=True)
#     return comb, U_map, L_map


# def max_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
#     _require_columns(df, ["mlc_leaf", "err_left_mm", "err_right_mm"], "max_by_leaf()")

#     g = df.groupby("mlc_leaf", dropna=True)

#     def _maxabs(x: pd.Series) -> float:
#         a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
#         a = a[np.isfinite(a)]
#         return float(np.max(np.abs(a))) if a.size else np.nan

#     out = pd.DataFrame(
#         {
#             "mlc_leaf": g.size().index.to_numpy(dtype=int),
#             "max_abs_left": g["err_left_mm"].apply(_maxabs).to_numpy(dtype=float),
#             "max_abs_right": g["err_right_mm"].apply(_maxabs).to_numpy(dtype=float),
#         }
#     )
#     return out.sort_values("mlc_leaf").reset_index(drop=True)


# def plot_max_errors_by_leaf(
#     mx_df: pd.DataFrame,
#     threshold_mm: float = DEFAULT_WARN_MM,  # WARNING
#     title: str = "Max Abs Error by Leaf",
#     fail_mm: Optional[float] = None,        # ACTION
#     show_bands: bool = True,
#     label_worst_n: int = 0,
# ) -> plt.Figure:
#     _require_columns(mx_df, ["mlc_leaf", "max_abs_left", "max_abs_right"], "plot_max_errors_by_leaf()")

#     warn_mm = float(threshold_mm)
#     action_mm = float(fail_mm) if fail_mm is not None else None

#     fig, ax = plt.subplots(figsize=(7.4, 4.2))
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_left"], marker="o", markersize=3, linewidth=1.5, label="Left Bank")
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_right"], marker="o", markersize=3, linewidth=1.5, label="Right Bank")

#     # Correct labels:
#     ax.axhline(warn_mm, linewidth=1.2, linestyle="--", label=f"Warning = {warn_mm:.2f} mm")
#     if action_mm is not None and np.isfinite(action_mm):
#         ax.axhline(action_mm, linewidth=1.2, linestyle="-.", label=f"Action = {action_mm:.2f} mm")

#     ax.set_title(title)
#     ax.set_xlabel("Leaf index")
#     ax.set_ylabel("Max abs error (mm)")
#     ax.grid(True, alpha=0.25)

#     vals = pd.to_numeric(
#         pd.concat([mx_df["max_abs_left"], mx_df["max_abs_right"]]),
#         errors="coerce",
#     ).to_numpy(dtype=float)
#     vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else warn_mm
#     ref_top = max(vmax * 1.15, warn_mm * 1.25, (action_mm or 0.0) * 1.25)

#     ybottom = _y_bottom_pad(ref_top)
#     ax.set_ylim(ybottom, ref_top)
#     _set_y_ticks_nice_range(ax, ybottom, ref_top)

#     if show_bands and action_mm is not None and np.isfinite(action_mm):
#         _add_threshold_bands(ax, warn_mm, action_mm)

#     if int(label_worst_n) > 0:
#         tmp = mx_df.copy()
#         tmp["worst"] = np.nanmax(tmp[["max_abs_left", "max_abs_right"]].to_numpy(dtype=float), axis=1)
#         tmp = tmp.sort_values("worst", ascending=False).head(int(label_worst_n))
#         for _, r in tmp.iterrows():
#             x = float(r["mlc_leaf"])
#             y = float(r["worst"])
#             if np.isfinite(x) and np.isfinite(y):
#                 ax.annotate(
#                     str(int(x)),
#                     xy=(x, y),
#                     xytext=(0, 6),
#                     textcoords="offset points",
#                     ha="center",
#                     fontsize=9,
#                     alpha=0.85,
#                 )

#     ax.legend()
#     fig.tight_layout()
#     return fig


# # =============================================================================
# # Trending
# # =============================================================================
# def summarize_overall_max_error(m_df: pd.DataFrame, scope: str) -> Dict:
#     if m_df is None or m_df.empty:
#         return {"Date": None, "GantryBin": np.nan, "scope": scope, "overall_max_abs_mm": np.nan, "n_points": 0}

#     _require_columns(m_df, ["abs_err_left_mm", "abs_err_right_mm"], "summarize_overall_max_error()")

#     date_raw = m_df["Date"].iloc[0] if "Date" in m_df.columns else None
#     date_ymd = _safe_date_ymd_from_any(date_raw)

#     gantry_bin = np.nan
#     if "GantryBin" in m_df.columns:
#         try:
#             gantry_bin = float(pd.to_numeric(m_df["GantryBin"].iloc[0], errors="coerce"))
#         except Exception:
#             gantry_bin = np.nan

#     overall_max = float(
#         np.nanmax(
#             np.r_[
#                 pd.to_numeric(m_df["abs_err_left_mm"], errors="coerce").to_numpy(dtype=float),
#                 pd.to_numeric(m_df["abs_err_right_mm"], errors="coerce").to_numpy(dtype=float),
#             ]
#         )
#     )

#     return {
#         "Date": date_ymd,
#         "GantryBin": gantry_bin,
#         "scope": str(scope),
#         "overall_max_abs_mm": overall_max,
#         "n_points": int(len(m_df)),
#     }


# def append_trending_csv(
#     trend_csv_path: Path,
#     new_rows: List[Dict],
#     dedup_cols: Optional[List[str]] = None,
# ) -> pd.DataFrame:
#     trend_csv_path = Path(trend_csv_path)
#     trend_csv_path.parent.mkdir(parents=True, exist_ok=True)

#     new_df = pd.DataFrame(new_rows)
#     if trend_csv_path.exists():
#         old_df = pd.read_csv(trend_csv_path)
#         all_df = pd.concat([old_df, new_df], ignore_index=True)
#     else:
#         all_df = new_df.copy()

#     if dedup_cols is None:
#         dedup_cols = ["Date", "GantryBin", "scope"]

#     for c in dedup_cols:
#         if c not in all_df.columns:
#             all_df[c] = np.nan

#     all_df = all_df.drop_duplicates(subset=dedup_cols, keep="last")
#     all_df["Date_dt"] = pd.to_datetime(all_df["Date"], errors="coerce")
#     all_df = all_df.sort_values(["scope", "GantryBin", "Date_dt"]).reset_index(drop=True)
#     all_df.drop(columns=["Date_dt"], inplace=True, errors="ignore")

#     all_df.to_csv(trend_csv_path, index=False)
#     return all_df


# def plot_overall_max_trending(
#     trend_df: pd.DataFrame,
#     scope: str = "combined",
#     gantry_bin: Optional[float] = None,
#     fail_mm: float = DEFAULT_FAIL_MM,          # ACTION
#     title: str = "Trending: Overall Max Absolute MLC Error",
#     warn_mm: Optional[float] = DEFAULT_WARN_MM,  # WARNING
#     rolling_window: int = 3,
#     show_rolling: bool = True,
#     show_bands: bool = True,
# ) -> plt.Figure:
#     df = trend_df.copy()

#     if "scope" in df.columns:
#         df = df[df["scope"] == scope]

#     if gantry_bin is not None and "GantryBin" in df.columns:
#         df = df[df["GantryBin"] == gantry_bin]

#     df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["overall_max_abs_mm"] = pd.to_numeric(df["overall_max_abs_mm"], errors="coerce")
#     df = df.dropna(subset=["Date_dt", "overall_max_abs_mm"]).sort_values("Date_dt")

#     fig, ax = plt.subplots(figsize=(8.2, 3.9))
#     ax.plot(df["Date_dt"], df["overall_max_abs_mm"], marker="o", markersize=3, linewidth=1.5, label="Run")

#     if show_rolling and len(df) >= 2:
#         w = max(1, int(rolling_window))
#         roll = df["overall_max_abs_mm"].rolling(w, min_periods=1).mean()
#         ax.plot(df["Date_dt"], roll, linewidth=2.2, label=f"Rolling mean ({w})")

#     # Correct labels:
#     ax.axhline(float(fail_mm), linestyle="-.", linewidth=1.2, label=f"Action = {float(fail_mm):.2f} mm")
#     if warn_mm is not None and np.isfinite(float(warn_mm)):
#         ax.axhline(float(warn_mm), linestyle="--", linewidth=1.2, label=f"Warning = {float(warn_mm):.2f} mm")

#     suffix = f" (Gantry {int(gantry_bin)})" if gantry_bin is not None and np.isfinite(gantry_bin) else ""
#     ax.set_title(title + suffix)
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Overall max abs error (mm)")
#     ax.grid(True, alpha=0.25)

#     ymax = float(np.nanmax(df["overall_max_abs_mm"].to_numpy(dtype=float))) if len(df) else float(fail_mm)
#     top = max(
#         ymax * 1.15,
#         float(fail_mm) * 1.25,
#         (float(warn_mm) * 1.25) if warn_mm is not None else 0.0,
#     )

#     ybottom = _y_bottom_pad(top)
#     ax.set_ylim(ybottom, top)
#     _set_y_ticks_nice_range(ax, ybottom, top)

#     if show_bands and warn_mm is not None and np.isfinite(float(warn_mm)):
#         _add_threshold_bands(ax, float(warn_mm), float(fail_mm))

#     ax.legend()
#     fig.tight_layout()
#     return fig




# # core/analysis.py
# from __future__ import annotations

# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Any

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# # =============================================================================
# # Constants / utils
# # =============================================================================
# DEFAULT_WARN_MM = 0.5
# DEFAULT_FAIL_MM = 1.0


# def _require_columns(df: pd.DataFrame, cols: List[str], fn: str) -> None:
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"{fn}: missing required columns {missing}")


# def _to_float_series(s: pd.Series) -> np.ndarray:
#     return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


# def _safe_date_ymd_from_any(x: Any) -> Optional[str]:
#     """
#     Best-effort conversion to YYYY-MM-DD.
#     Handles:
#       - datetime-like
#       - pandas Timestamp
#       - strings like 'Jan 13 2026 17:52' or ISO
#     """
#     if x is None:
#         return None
#     try:
#         dt = pd.to_datetime(x, errors="coerce")
#         if pd.notna(dt):
#             return pd.Timestamp(dt).strftime("%Y-%m-%d")
#     except Exception:
#         pass
#     return None


# # =============================================================================
# # 3.0 Errors
# # =============================================================================
# def add_errors(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute delivered - nominal errors (mm) and abs metrics.

#     Required columns:
#       left_mm, right_mm, left_mm_nom, right_mm_nom
#     Produced columns:
#       err_left_mm, err_right_mm, abs_err_left_mm, abs_err_right_mm, abs_err_max_mm
#     """
#     df = df.copy()
#     _require_columns(df, ["left_mm", "right_mm", "left_mm_nom", "right_mm_nom"], "add_errors()")

#     left = _to_float_series(df["left_mm"])
#     right = _to_float_series(df["right_mm"])
#     left_nom = _to_float_series(df["left_mm_nom"])
#     right_nom = _to_float_series(df["right_mm_nom"])

#     err_left = left - left_nom
#     err_right = right - right_nom

#     df["err_left_mm"] = err_left
#     df["err_right_mm"] = err_right
#     df["abs_err_left_mm"] = np.abs(err_left)
#     df["abs_err_right_mm"] = np.abs(err_right)
#     df["abs_err_max_mm"] = np.maximum(df["abs_err_left_mm"].to_numpy(dtype=float), df["abs_err_right_mm"].to_numpy(dtype=float))

#     return df


# def error_describe(df: pd.DataFrame) -> pd.DataFrame:
#     """Return describe() table for left/right errors."""
#     _require_columns(df, ["err_left_mm", "err_right_mm"], "error_describe()")
#     return df[["err_left_mm", "err_right_mm"]].describe()


# # =============================================================================
# # 3.1 Leaf metrics
# # =============================================================================
# def leaf_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
#     """
#     Per-leaf max abs and RMS for left/right.
#     Requires: leaf_pair, err_* and abs_err_* columns (call add_errors() first).
#     """
#     _require_columns(
#         df,
#         ["leaf_pair", "err_left_mm", "err_right_mm", "abs_err_left_mm", "abs_err_right_mm"],
#         "leaf_metrics()",
#     )

#     g = df.groupby("leaf_pair", dropna=True)

#     def _rms(x: pd.Series) -> float:
#         a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
#         a = a[np.isfinite(a)]
#         return float(np.sqrt(np.mean(a**2))) if a.size else np.nan

#     out = (
#         g.agg(
#             max_abs_left_mm=("abs_err_left_mm", "max"),
#             rms_left_mm=("err_left_mm", _rms),
#             max_abs_right_mm=("abs_err_right_mm", "max"),
#             rms_right_mm=("err_right_mm", _rms),
#         )
#         .reset_index()
#     )
#     out["table"] = str(label)
#     return out


# def classify(metrics_df: pd.DataFrame, warn_mm: float = DEFAULT_WARN_MM, fail_mm: float = DEFAULT_FAIL_MM) -> Tuple[str, float]:
#     """
#     PASS/WARN/FAIL based on max of max_abs_left/right.
#     NOTE: uses strict '>' thresholds to match your existing behavior.
#     """
#     _require_columns(metrics_df, ["max_abs_left_mm", "max_abs_right_mm"], "classify()")

#     m = float(np.nanmax(metrics_df[["max_abs_left_mm", "max_abs_right_mm"]].to_numpy(dtype=float)))
#     if not np.isfinite(m):
#         return "UNKNOWN", np.nan
#     if m > float(fail_mm):
#         return "FAIL", m
#     if m > float(warn_mm):
#         return "WARN", m
#     return "PASS", m


# # =============================================================================
# # 3.3 Virtual picket fence plot
# # =============================================================================
# def compute_gap_df_from_merged(m_df: pd.DataFrame, use_nominal: bool = False) -> pd.DataFrame:
#     """
#     Per-leaf mean gap (mm).
#       Delivered uses right_mm - left_mm,
#       Nominal uses right_mm_nom - left_mm_nom.
#     """
#     df = m_df.copy()
#     if use_nominal:
#         _require_columns(df, ["leaf_pair", "left_mm_nom", "right_mm_nom"], "compute_gap_df_from_merged()")
#         gap = _to_float_series(df["right_mm_nom"]) - _to_float_series(df["left_mm_nom"])
#     else:
#         _require_columns(df, ["leaf_pair", "left_mm", "right_mm"], "compute_gap_df_from_merged()")
#         gap = _to_float_series(df["right_mm"]) - _to_float_series(df["left_mm"])

#     out = (
#         pd.DataFrame({"leaf_pair": pd.to_numeric(df["leaf_pair"], errors="coerce"), "gap_mm": gap})
#         .dropna(subset=["leaf_pair"])
#         .assign(leaf_pair=lambda d: d["leaf_pair"].astype(int))
#         .groupby("leaf_pair", as_index=False)
#         .agg(gap_mm=("gap_mm", "mean"))
#     )
#     return out


# def plot_virtual_picket_fence(
#     picket_centers_mm: List[float],
#     gap_df: pd.DataFrame,
#     title: str,
#     xlim_mm: Tuple[float, float] = (-110, 110),
#     xticks: Optional[np.ndarray] = None,
#     ytick_step: int = 1,
# ) -> plt.Figure:
#     """
#     Draw a virtual picket fence where each leaf has its own gap width.
#     Returns a matplotlib Figure (safe for st.pyplot).
#     """
#     _require_columns(gap_df, ["leaf_pair", "gap_mm"], "plot_virtual_picket_fence()")

#     leaves = np.sort(pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).unique())
#     if leaves.size == 0:
#         raise ValueError("plot_virtual_picket_fence(): gap_df has no leaves")

#     y0, y1 = int(leaves.min()), int(leaves.max())
#     gap_map = dict(
#         zip(
#             pd.to_numeric(gap_df["leaf_pair"], errors="coerce").dropna().astype(int).tolist(),
#             pd.to_numeric(gap_df["gap_mm"], errors="coerce").to_numpy(dtype=float).tolist(),
#         )
#     )

#     fig, ax = plt.subplots(figsize=(6.4, 7.2))
#     ax.set_title(title)

#     for y in range(y0, y1 + 1):
#         w = float(gap_map.get(y, np.nan))
#         if not np.isfinite(w) or w <= 0:
#             continue

#         for xc in picket_centers_mm:
#             ax.add_patch(
#                 Rectangle(
#                     (float(xc) - w / 2.0, y - 0.5),
#                     w,
#                     1.0,
#                     facecolor="black",
#                     edgecolor="black",
#                     linewidth=0,
#                 )
#             )
#         ax.axhline(y, linewidth=0.4, alpha=0.25)

#     ax.set_xlim(*xlim_mm)
#     ax.set_ylim(y1 + 1, y0 - 1)  # invert
#     ax.set_xlabel("Distance (mm)")
#     ax.set_ylabel("Leaf index")

#     if xticks is not None:
#         ax.set_xticks(xticks)
#     else:
#         ax.set_xticks(np.arange(-100, 101, 20))

#     ax.set_yticks(np.arange(y0, y1 + 1, max(1, int(ytick_step))))
#     ax.grid(False)
#     fig.tight_layout()
#     return fig


# # =============================================================================
# # 3.5/3.6 Combined leaf indexing + plots
# # =============================================================================
# def make_combined_leaf_index(mU: pd.DataFrame, mL: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
#     """
#     Create a combined leaf index across upper and lower stacks:
#       - upper mapped to 1..N
#       - lower mapped to N+1..N+M

#     Requires columns:
#       leaf_pair (in both)
#       match_id (for sorting; if absent, sorts without it)
#     """
#     _require_columns(mU, ["leaf_pair"], "make_combined_leaf_index(mU)")
#     _require_columns(mL, ["leaf_pair"], "make_combined_leaf_index(mL)")

#     U = mU.copy()
#     L = mL.copy()

#     upper_pairs = sorted(pd.to_numeric(U["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())
#     lower_pairs = sorted(pd.to_numeric(L["leaf_pair"], errors="coerce").dropna().astype(int).unique().tolist())

#     U_map = {p: i + 1 for i, p in enumerate(upper_pairs)}
#     L_map = {p: i + 1 + len(upper_pairs) for i, p in enumerate(lower_pairs)}

#     U["leaf_pair"] = pd.to_numeric(U["leaf_pair"], errors="coerce").astype("Int64")
#     L["leaf_pair"] = pd.to_numeric(L["leaf_pair"], errors="coerce").astype("Int64")

#     U["mlc_leaf"] = U["leaf_pair"].map(U_map)
#     L["mlc_leaf"] = L["leaf_pair"].map(L_map)

#     comb = pd.concat([U, L], ignore_index=True)

#     sort_cols = ["mlc_leaf"]
#     if "match_id" in comb.columns:
#         sort_cols.append("match_id")

#     comb = comb.sort_values(sort_cols).reset_index(drop=True)
#     return comb, U_map, L_map


# def rms_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Compute RMS per combined leaf for left/right.
#     Requires: mlc_leaf, err_left_mm, err_right_mm
#     """
#     _require_columns(df, ["mlc_leaf", "err_left_mm", "err_right_mm"], "rms_by_leaf()")

#     g = df.groupby("mlc_leaf", dropna=True)

#     def _rms(x: pd.Series) -> float:
#         a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
#         a = a[np.isfinite(a)]
#         return float(np.sqrt(np.mean(a**2))) if a.size else np.nan

#     out = pd.DataFrame(
#         {
#             "mlc_leaf": g.size().index.to_numpy(dtype=int),
#             "rms_left_mm": g["err_left_mm"].apply(_rms).to_numpy(dtype=float),
#             "rms_right_mm": g["err_right_mm"].apply(_rms).to_numpy(dtype=float),
#         }
#     )
#     return out.sort_values("mlc_leaf").reset_index(drop=True)


# def plot_rms_errors_by_leaf(rms_df: pd.DataFrame, title: str = "RMS Error by Leaf") -> plt.Figure:
#     _require_columns(rms_df, ["mlc_leaf", "rms_left_mm", "rms_right_mm"], "plot_rms_errors_by_leaf()")

#     fig, ax = plt.subplots(figsize=(7.4, 4.2))
#     ax.plot(rms_df["mlc_leaf"], rms_df["rms_left_mm"], marker="o", markersize=3, linewidth=1.5, label="Left Bank")
#     ax.plot(rms_df["mlc_leaf"], rms_df["rms_right_mm"], marker="o", markersize=3, linewidth=1.5, label="Right Bank")
#     ax.set_title(title)
#     ax.set_xlabel("Leaf index")
#     ax.set_ylabel("RMS error (mm)")
#     ax.grid(True, alpha=0.25)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# def max_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Max abs error per combined leaf for left/right.
#     Requires: mlc_leaf, err_left_mm, err_right_mm
#     """
#     _require_columns(df, ["mlc_leaf", "err_left_mm", "err_right_mm"], "max_by_leaf()")

#     g = df.groupby("mlc_leaf", dropna=True)

#     def _maxabs(x: pd.Series) -> float:
#         a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
#         a = a[np.isfinite(a)]
#         return float(np.max(np.abs(a))) if a.size else np.nan

#     out = pd.DataFrame(
#         {
#             "mlc_leaf": g.size().index.to_numpy(dtype=int),
#             "max_abs_left": g["err_left_mm"].apply(_maxabs).to_numpy(dtype=float),
#             "max_abs_right": g["err_right_mm"].apply(_maxabs).to_numpy(dtype=float),
#         }
#     )
#     return out.sort_values("mlc_leaf").reset_index(drop=True)


# def plot_max_errors_by_leaf(mx_df: pd.DataFrame, threshold_mm: float = DEFAULT_WARN_MM, title: str = "Max Abs Error by Leaf") -> plt.Figure:
#     _require_columns(mx_df, ["mlc_leaf", "max_abs_left", "max_abs_right"], "plot_max_errors_by_leaf()")

#     fig, ax = plt.subplots(figsize=(7.4, 4.2))
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_left"], marker="o", markersize=3, linewidth=1.5, label="Left Bank")
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_right"], marker="o", markersize=3, linewidth=1.5, label="Right Bank")
#     ax.axhline(float(threshold_mm), linewidth=1.3, linestyle="--", label=f"Reference = {float(threshold_mm):.2f} mm")
#     ax.set_title(title)
#     ax.set_xlabel("Leaf index")
#     ax.set_ylabel("Max abs error (mm)")
#     ax.grid(True, alpha=0.25)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# # =============================================================================
# # 3.9 Trending (simple, project-relative)
# # =============================================================================
# def summarize_overall_max_error(m_df: pd.DataFrame, scope: str) -> Dict:
#     """
#     One-row summary:
#       Date (best-effort), GantryBin (best-effort), overall_max_abs_mm, n_points

#     Notes:
#       - Date is best-effort; if not parseable, returns None.
#       - GantryBin is optional; if not present, NaN.
#       - Requires abs_err_left_mm/abs_err_right_mm (call add_errors() first).
#     """
#     if m_df is None or m_df.empty:
#         return {"Date": None, "GantryBin": np.nan, "scope": scope, "overall_max_abs_mm": np.nan, "n_points": 0}

#     _require_columns(m_df, ["abs_err_left_mm", "abs_err_right_mm"], "summarize_overall_max_error()")

#     date_raw = m_df["Date"].iloc[0] if "Date" in m_df.columns else None
#     date_ymd = _safe_date_ymd_from_any(date_raw)

#     gantry_bin = np.nan
#     if "GantryBin" in m_df.columns:
#         try:
#             gantry_bin = float(pd.to_numeric(m_df["GantryBin"].iloc[0], errors="coerce"))
#         except Exception:
#             gantry_bin = np.nan

#     overall_max = float(
#         np.nanmax(
#             np.r_[
#                 pd.to_numeric(m_df["abs_err_left_mm"], errors="coerce").to_numpy(dtype=float),
#                 pd.to_numeric(m_df["abs_err_right_mm"], errors="coerce").to_numpy(dtype=float),
#             ]
#         )
#     )

#     return {
#         "Date": date_ymd,
#         "GantryBin": gantry_bin,
#         "scope": str(scope),
#         "overall_max_abs_mm": overall_max,
#         "n_points": int(len(m_df)),
#     }


# def append_trending_csv(trend_csv_path: Path, new_rows: List[Dict], dedup_cols: Optional[List[str]] = None) -> pd.DataFrame:
#     """
#     Append new rows to a trend CSV and return the full dataframe.

#     Default de-dup keys: Date, GantryBin, scope
#     Sort order: scope, GantryBin, Date
#     """
#     trend_csv_path = Path(trend_csv_path)
#     trend_csv_path.parent.mkdir(parents=True, exist_ok=True)

#     new_df = pd.DataFrame(new_rows)
#     if trend_csv_path.exists():
#         old_df = pd.read_csv(trend_csv_path)
#         all_df = pd.concat([old_df, new_df], ignore_index=True)
#     else:
#         all_df = new_df.copy()

#     if dedup_cols is None:
#         dedup_cols = ["Date", "GantryBin", "scope"]

#     for c in dedup_cols:
#         if c not in all_df.columns:
#             all_df[c] = np.nan

#     all_df = all_df.drop_duplicates(subset=dedup_cols, keep="last")
#     all_df["Date_dt"] = pd.to_datetime(all_df["Date"], errors="coerce")
#     all_df = all_df.sort_values(["scope", "GantryBin", "Date_dt"]).reset_index(drop=True)
#     all_df.drop(columns=["Date_dt"], inplace=True, errors="ignore")

#     all_df.to_csv(trend_csv_path, index=False)
#     return all_df


# def plot_overall_max_trending(
#     trend_df: pd.DataFrame,
#     scope: str = "combined",
#     gantry_bin: Optional[float] = None,
#     fail_mm: float = DEFAULT_WARN_MM,
#     title: str = "Trending: Overall Max Absolute MLC Error",
# ) -> plt.Figure:
#     """
#     Plot overall max abs error over time.

#     Requires trend_df columns:
#       Date, scope, overall_max_abs_mm
#     Optional:
#       GantryBin
#     """
#     df = trend_df.copy()

#     # Filter by scope
#     if "scope" in df.columns:
#         df = df[df["scope"] == scope]

#     # Optional gantry bin filter
#     if gantry_bin is not None and "GantryBin" in df.columns:
#         df = df[df["GantryBin"] == gantry_bin]

#     # Parse dates and clean
#     df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["overall_max_abs_mm"] = pd.to_numeric(df["overall_max_abs_mm"], errors="coerce")
#     df = df.dropna(subset=["Date_dt", "overall_max_abs_mm"]).sort_values("Date_dt")

#     fig, ax = plt.subplots(figsize=(8.2, 3.9))
#     ax.plot(df["Date_dt"], df["overall_max_abs_mm"], marker="o", markersize=3, linewidth=1.5)

#     ax.axhline(float(fail_mm), linestyle="--", linewidth=1.2, label=f"Reference = {float(fail_mm):.2f} mm")
#     suffix = f" (Gantry {int(gantry_bin)})" if gantry_bin is not None and np.isfinite(gantry_bin) else ""
#     ax.set_title(title + suffix)

#     ax.set_xlabel("Date")
#     ax.set_ylabel("Overall max abs error (mm)")
#     ax.grid(True, alpha=0.25)
#     ax.legend()
#     fig.tight_layout()
#     return fig

# # core/analysis.py
# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle


# # -----------------------------
# # 3.0 Errors
# # -----------------------------
# def add_errors(df: pd.DataFrame) -> pd.DataFrame:
#     """Compute delivered - nominal errors (mm) and abs metrics."""
#     df = df.copy()
#     required = ["left_mm", "right_mm", "left_mm_nom", "right_mm_nom"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValueError(f"add_errors(): missing columns {missing}")

#     df["err_left_mm"] = df["left_mm"] - df["left_mm_nom"]
#     df["err_right_mm"] = df["right_mm"] - df["right_mm_nom"]
#     df["abs_err_left_mm"] = np.abs(df["err_left_mm"])
#     df["abs_err_right_mm"] = np.abs(df["err_right_mm"])
#     df["abs_err_max_mm"] = np.maximum(df["abs_err_left_mm"], df["abs_err_right_mm"])
#     return df


# def error_describe(df: pd.DataFrame) -> pd.DataFrame:
#     """Return describe() table for left/right errors."""
#     return df[["err_left_mm", "err_right_mm"]].describe()


# # -----------------------------
# # 3.1 Leaf metrics
# # -----------------------------
# def leaf_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
#     """Per-leaf max abs and RMS for left/right."""
#     if "leaf_pair" not in df.columns:
#         raise ValueError("leaf_metrics(): missing 'leaf_pair'")

#     g = df.groupby("leaf_pair")
#     out = g.agg(
#         max_abs_left_mm=("abs_err_left_mm", "max"),
#         rms_left_mm=("err_left_mm", lambda s: float(np.sqrt(np.mean(s**2)))),
#         max_abs_right_mm=("abs_err_right_mm", "max"),
#         rms_right_mm=("err_right_mm", lambda s: float(np.sqrt(np.mean(s**2)))),
#     ).reset_index()
#     out["table"] = label
#     return out


# def classify(metrics_df: pd.DataFrame, warn_mm: float = 0.5, fail_mm: float = 1.0) -> Tuple[str, float]:
#     """PASS/WARN/FAIL based on max of max_abs_left/right."""
#     m = float(metrics_df[["max_abs_left_mm", "max_abs_right_mm"]].to_numpy().max())
#     if m > fail_mm:
#         return "FAIL", m
#     if m > warn_mm:
#         return "WARN", m
#     return "PASS", m


# # -----------------------------
# # 3.3 Virtual picket fence plot
# # -----------------------------
# def compute_gap_df_from_merged(m_df: pd.DataFrame, use_nominal: bool = False) -> pd.DataFrame:
#     """
#     Per-leaf mean gap (mm). Delivered uses right_mm - left_mm, nominal uses *_nom.
#     """
#     df = m_df.copy()
#     if use_nominal:
#         gap = df["right_mm_nom"] - df["left_mm_nom"]
#     else:
#         gap = df["right_mm"] - df["left_mm"]

#     out = (pd.DataFrame({
#         "leaf_pair": df["leaf_pair"].astype(int),
#         "gap_mm": gap.astype(float),
#     })
#     .groupby("leaf_pair", as_index=False)
#     .agg(gap_mm=("gap_mm", "mean")))
#     return out


# def plot_virtual_picket_fence(
#     picket_centers_mm: List[float],
#     gap_df: pd.DataFrame,
#     title: str,
#     xlim_mm: Tuple[float, float] = (-110, 110),
#     xticks: Optional[np.ndarray] = None,
#     ytick_step: int = 1,
# ) -> plt.Figure:
#     """
#     Draw a virtual picket fence where each leaf has its own gap width.
#     Returns a matplotlib Figure (for st.pyplot).
#     """
#     leaves = np.sort(gap_df["leaf_pair"].dropna().unique().astype(int))
#     if len(leaves) == 0:
#         raise ValueError("plot_virtual_picket_fence(): gap_df has no leaves")

#     y0, y1 = int(leaves.min()), int(leaves.max())
#     gap_map = dict(zip(gap_df["leaf_pair"].astype(int), gap_df["gap_mm"].astype(float)))

#     fig, ax = plt.subplots(figsize=(6, 7))
#     ax.set_title(title)

#     for y in range(y0, y1 + 1):
#         w = float(gap_map.get(y, np.nan))
#         if not np.isfinite(w) or w <= 0:
#             continue

#         for xc in picket_centers_mm:
#             ax.add_patch(Rectangle(
#                 (xc - w / 2.0, y - 0.5),
#                 w, 1.0,
#                 facecolor="black",
#                 edgecolor="black",
#                 linewidth=0
#             ))
#         ax.axhline(y, linewidth=0.4, alpha=0.3)

#     ax.set_xlim(*xlim_mm)
#     ax.set_ylim(y1 + 1, y0 - 1)  # invert
#     ax.set_xlabel("Distance (mm)")
#     ax.set_ylabel("Leaf index")

#     if xticks is not None:
#         ax.set_xticks(xticks)
#     else:
#         ax.set_xticks(np.arange(-100, 101, 20))

#     ax.set_yticks(np.arange(y0, y1 + 1, ytick_step))
#     fig.tight_layout()
#     return fig


# # -----------------------------
# # 3.5/3.6 Combined leaf indexing + plots
# # -----------------------------
# def make_combined_leaf_index(mU: pd.DataFrame, mL: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
#     U = mU.copy()
#     L = mL.copy()

#     upper_pairs = sorted(U["leaf_pair"].unique())
#     lower_pairs = sorted(L["leaf_pair"].unique())

#     U_map = {p: i + 1 for i, p in enumerate(upper_pairs)}
#     L_map = {p: i + 1 + len(upper_pairs) for i, p in enumerate(lower_pairs)}

#     U["mlc_leaf"] = U["leaf_pair"].map(U_map)
#     L["mlc_leaf"] = L["leaf_pair"].map(L_map)

#     comb = pd.concat([U, L], ignore_index=True)
#     comb = comb.sort_values(["mlc_leaf", "match_id"]).reset_index(drop=True)
#     return comb, U_map, L_map


# def rms_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
#     g = df.groupby("mlc_leaf")
#     rms_left = g["err_left_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2))))
#     rms_right = g["err_right_mm"].apply(lambda s: float(np.sqrt(np.mean(s**2))))
#     return pd.DataFrame({
#         "mlc_leaf": rms_left.index.values,
#         "rms_left_mm": rms_left.values,
#         "rms_right_mm": rms_right.values,
#     })


# def plot_rms_errors_by_leaf(rms_df: pd.DataFrame) -> plt.Figure:
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(rms_df["mlc_leaf"], rms_df["rms_left_mm"], marker="o", markersize=3, linewidth=1.5, label="Left Bank")
#     ax.plot(rms_df["mlc_leaf"], rms_df["rms_right_mm"], marker="o", markersize=3, linewidth=1.5, label="Right Bank")
#     ax.set_xlabel("Leaf index")
#     ax.set_ylabel("RMS error (mm)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# def max_by_leaf(df: pd.DataFrame) -> pd.DataFrame:
#     g = df.groupby("mlc_leaf")
#     return pd.DataFrame({
#         "mlc_leaf": g.size().index.values,
#         "max_abs_left": g["err_left_mm"].apply(lambda s: float(np.max(np.abs(s)))).values,
#         "max_abs_right": g["err_right_mm"].apply(lambda s: float(np.max(np.abs(s)))).values,
#     })


# def plot_max_errors_by_leaf(mx_df: pd.DataFrame, threshold_mm: float = 0.5) -> plt.Figure:
#     fig, ax = plt.subplots(figsize=(7, 4))
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_left"], marker="o", markersize=3, linewidth=1.5, label="Left Bank")
#     ax.plot(mx_df["mlc_leaf"], mx_df["max_abs_right"], marker="o", markersize=3, linewidth=1.5, label="Right Bank")
#     ax.axhline(threshold_mm, linewidth=1.5, linestyle="--", label=f"Tolerance = {threshold_mm:.2f} mm")
#     ax.set_xlabel("Leaf index")
#     ax.set_ylabel("Max abs error (mm)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     fig.tight_layout()
#     return fig


# # -----------------------------
# # 3.9 Trending (simple, project-relative)
# # -----------------------------
# def summarize_overall_max_error(m_df: pd.DataFrame, scope: str) -> Dict:
#     """
#     One row summary:
#       Date (best-effort), GantryBin (best-effort), overall_max_abs_mm, n_points
#     """
#     if m_df is None or m_df.empty:
#         return {"Date": None, "GantryBin": np.nan, "scope": scope, "overall_max_abs_mm": np.nan, "n_points": 0}

#     date_raw = m_df["Date"].iloc[0] if "Date" in m_df.columns else None
#     date_dt = pd.to_datetime(date_raw, errors="coerce")
#     date_ymd = date_dt.strftime("%Y-%m-%d") if pd.notna(date_dt) else None

#     gantry_bin = float(m_df["GantryBin"].iloc[0]) if "GantryBin" in m_df.columns else np.nan
#     overall_max = float(np.nanmax(np.r_[m_df["abs_err_left_mm"].to_numpy(), m_df["abs_err_right_mm"].to_numpy()]))

#     return {
#         "Date": date_ymd,
#         "GantryBin": gantry_bin,
#         "scope": scope,
#         "overall_max_abs_mm": overall_max,
#         "n_points": int(len(m_df)),
#     }


# def append_trending_csv(trend_csv_path: Path, new_rows: List[Dict], dedup_cols: Optional[List[str]] = None) -> pd.DataFrame:
#     trend_csv_path.parent.mkdir(parents=True, exist_ok=True)
#     new_df = pd.DataFrame(new_rows)

#     if trend_csv_path.exists():
#         old_df = pd.read_csv(trend_csv_path)
#         all_df = pd.concat([old_df, new_df], ignore_index=True)
#     else:
#         all_df = new_df.copy()

#     if dedup_cols is None:
#         dedup_cols = ["Date", "GantryBin", "scope"]

#     all_df = all_df.drop_duplicates(subset=dedup_cols, keep="last")
#     all_df["Date_dt"] = pd.to_datetime(all_df["Date"], errors="coerce")
#     all_df = all_df.sort_values(["scope", "GantryBin", "Date_dt"]).reset_index(drop=True)
#     all_df.drop(columns=["Date_dt"], inplace=True, errors="ignore")

#     all_df.to_csv(trend_csv_path, index=False)
#     return all_df


# def plot_overall_max_trending(trend_df: pd.DataFrame, scope: str = "combined", gantry_bin: Optional[float] = None,
#                              fail_mm: float = 0.5, title: str = "Trending: Overall Max Absolute MLC Error") -> plt.Figure:
#     df = trend_df.copy()
#     df = df[df["scope"] == scope]
#     if gantry_bin is not None:
#         df = df[df["GantryBin"] == gantry_bin]

#     df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
#     df = df.dropna(subset=["Date_dt", "overall_max_abs_mm"]).sort_values("Date_dt")

#     fig, ax = plt.subplots(figsize=(8, 3.8))
#     ax.plot(df["Date_dt"], df["overall_max_abs_mm"], marker="o", markersize=3, linewidth=1.5)
#     ax.axhline(fail_mm, linestyle="--", linewidth=1.2, label=f"FAIL > {fail_mm} mm")
#     ax.set_title(title + (f" (Gantry {int(gantry_bin)})" if gantry_bin is not None else ""))
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Overall max abs error (mm)")
#     ax.grid(True, alpha=0.3)
#     ax.legend()
#     fig.tight_layout()
#     return fig
