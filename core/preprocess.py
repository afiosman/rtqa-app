# core/preprocess.py
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Edge exclusions (per your current convention)
# =============================================================================
UPPER_EDGE_EXCLUDE = {1, 2, 33, 34}
LOWER_EDGE_EXCLUDE = {1, 2, 3, 33, 34, 35}

# =============================================================================
# Gantry binning
# =============================================================================
def gantry_to_bin(deg: object, step: float = 90.0) -> float:
    """
    Map gantry angle to nearest bin (default: 0/90/180/270).
    - Normalizes into [0, 360)
    - Returns NaN if input is missing
    """
    if pd.isna(deg):
        return np.nan
    d = float(deg) % 360.0
    b = (np.round(d / step) * step) % 360.0
    if np.isclose(b, 360.0) or np.isclose(b, 0.0):
        return 0.0
    return float(b)


# =============================================================================
# Helpers
# =============================================================================
_LEAF_RE = re.compile(r"Leaf_(\d+)_")

def _find_leaf_pairs(df_wide: pd.DataFrame) -> List[int]:
    """
    Find all leaf pairs available in a wide dataframe with columns like:
      Leaf_<p>_Left(cm), Leaf_<p>_Right(cm)
    """
    pairs: List[int] = []
    for c in df_wide.columns:
        m = _LEAF_RE.search(str(c))
        if m:
            try:
                pairs.append(int(m.group(1)))
            except Exception:
                pass
    return sorted(set(pairs))


def _as_int_or_nan(x: object) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return float(int(x))
    except Exception:
        return np.nan


def _as_float_or_nan(x: object) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


# =============================================================================
# Wide -> long
# =============================================================================
def wide_to_long(df_wide: pd.DataFrame, table: str, prefix: str) -> pd.DataFrame:
    """
    Convert wide leaf columns into long-form per-leaf records.

    table:  'upper' | 'lower'
    prefix: 'plan'  | 'log'

    Output includes:
      prefix, table, leaf_pair, left_cm/right_cm, left_mm/right_mm,
      Beam, Segment, GantryDeg, GantryBin, TotalMU,
      plus any available metadata columns.
    """
    if table not in ("upper", "lower"):
        raise ValueError("wide_to_long(): table must be 'upper' or 'lower'")
    if prefix not in ("plan", "log"):
        raise ValueError("wide_to_long(): prefix must be 'plan' or 'log'")

    if df_wide is None or df_wide.empty:
        return pd.DataFrame(
            columns=[
                "prefix",
                "table",
                "leaf_pair",
                "left_cm",
                "right_cm",
                "left_mm",
                "right_mm",
                "Beam",
                "Segment",
                "GantryDeg",
                "GantryBin",
                "TotalMU",
            ]
        )

    pairs = _find_leaf_pairs(df_wide)

    meta_candidates = [
        "SourceFile",
        "Patient Name",
        "Patient ID",
        "Plan Name",
        "Date",
        "Beam",
        "Segment",
        "GantryDeg",
        "TotalMU",
    ]
    meta_cols = [c for c in meta_candidates if c in df_wide.columns]

    out: List[Dict] = []
    # NOTE: iterrows is OK here; if you need speed later, we can vectorize.
    for _, row in df_wide.iterrows():
        beam = _as_int_or_nan(row.get("Beam", np.nan)) if "Beam" in df_wide.columns else np.nan
        seg = _as_int_or_nan(row.get("Segment", np.nan)) if "Segment" in df_wide.columns else np.nan
        gantry = _as_float_or_nan(row.get("GantryDeg", np.nan)) if "GantryDeg" in df_wide.columns else np.nan
        mu = _as_float_or_nan(row.get("TotalMU", np.nan)) if "TotalMU" in df_wide.columns else np.nan

        gantry_bin = gantry_to_bin(gantry, step=90.0)

        for p in pairs:
            L = row.get(f"Leaf_{p}_Left(cm)", np.nan)
            R = row.get(f"Leaf_{p}_Right(cm)", np.nan)

            if pd.isna(L) and pd.isna(R):
                continue

            rec = {c: row.get(c, None) for c in meta_cols}
            rec.update(
                {
                    "prefix": prefix,
                    "table": table,
                    "Beam": beam,
                    "Segment": seg,
                    "GantryDeg": gantry,
                    "GantryBin": gantry_bin,
                    "TotalMU": mu,
                    "leaf_pair": int(p),
                    "left_cm": _as_float_or_nan(L),
                    "right_cm": _as_float_or_nan(R),
                }
            )
            # Convert cm -> mm (safe for NaN)
            rec["left_mm"] = rec["left_cm"] * 10.0 if np.isfinite(rec["left_cm"]) else np.nan
            rec["right_mm"] = rec["right_cm"] * 10.0 if np.isfinite(rec["right_cm"]) else np.nan

            out.append(rec)

    df_long = pd.DataFrame(out)

    # enforce types where possible
    if not df_long.empty:
        df_long["leaf_pair"] = pd.to_numeric(df_long["leaf_pair"], errors="coerce").astype("Int64")
        for c in ("Beam", "Segment"):
            if c in df_long.columns:
                df_long[c] = pd.to_numeric(df_long[c], errors="coerce")
        for c in ("GantryDeg", "GantryBin", "TotalMU", "left_mm", "right_mm", "left_cm", "right_cm"):
            if c in df_long.columns:
                df_long[c] = pd.to_numeric(df_long[c], errors="coerce")

    return df_long


# =============================================================================
# Exclusions
# =============================================================================
def apply_edge_exclusion(df_long: pd.DataFrame, table: str) -> pd.DataFrame:
    """
    Remove edge leaves that are known to be unreliable / outside analysis scope.
    """
    if df_long is None or df_long.empty:
        return df_long.copy()

    df_long = df_long.copy()
    if "leaf_pair" not in df_long.columns:
        return df_long

    if table == "upper":
        excl = UPPER_EDGE_EXCLUDE
    elif table == "lower":
        excl = LOWER_EDGE_EXCLUDE
    else:
        raise ValueError("apply_edge_exclusion(): table must be 'upper' or 'lower'")

    leaf = pd.to_numeric(df_long["leaf_pair"], errors="coerce")
    keep = ~leaf.isin(list(excl))
    return df_long.loc[keep].reset_index(drop=True)


# =============================================================================
# Merge plan/log
# =============================================================================
def merge_plan_log(log_df: pd.DataFrame, plan_nom: pd.DataFrame, table_label: str) -> pd.DataFrame:
    """
    Merge log long-form data with nominal plan long-form on:
      GantryBin, match_id, leaf_pair
    """
    if log_df is None or log_df.empty:
        return pd.DataFrame()

    keys = ["GantryBin", "match_id", "leaf_pair"]
    needed_plan_cols = keys + ["left_mm_nom", "right_mm_nom"]

    missing = [c for c in needed_plan_cols if c not in plan_nom.columns]
    if missing:
        raise ValueError(f"merge_plan_log(): plan_nom missing columns: {missing}")

    merged = log_df.merge(
        plan_nom[needed_plan_cols],
        on=keys,
        how="inner",
        validate="m:1",
    )
    merged["table"] = table_label
    return merged


# =============================================================================
# Public pipeline
# =============================================================================
def preprocess_and_merge(
    dfP_upper: pd.DataFrame,
    dfP_lower: pd.DataFrame,
    dfL_upper: pd.DataFrame,
    dfL_lower: pd.DataFrame,
    drop_stack_by_plan_name: bool = True,
) -> dict:
    """
    Returns dict with:
      planU, planL, logU, logL, planU_nom, planL_nom, mU, mL

    Assumes:
      - Plan wide files contain Beam as the matching id
      - Log wide files contain Segment as the matching id
      - GantryBin is used as coarse match key (90° bins)
    """

    # Optional: remove wrong stack by Plan Name convention
    if drop_stack_by_plan_name and "Plan Name" in dfP_upper.columns and "Plan Name" in dfL_upper.columns:
        maskP = ~dfP_upper["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 2 g")
        maskL = ~dfL_upper["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 2 g")
        dfP_upper = dfP_upper.loc[maskP].copy()
        dfL_upper = dfL_upper.loc[maskL].copy()

    if drop_stack_by_plan_name and "Plan Name" in dfP_lower.columns and "Plan Name" in dfL_lower.columns:
        maskP = ~dfP_lower["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 1 g")
        maskL = ~dfL_lower["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 1 g")
        dfP_lower = dfP_lower.loc[maskP].copy()
        dfL_lower = dfL_lower.loc[maskL].copy()

    # Numeric normalize (safe)
    for d in (dfP_upper, dfP_lower, dfL_upper, dfL_lower):
        if d is None or d.empty:
            continue
        for c in ("Beam", "Segment", "GantryDeg"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

    # Sort (safe even if cols missing)
    def _sort_safe(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        cols = [c for c in ("GantryDeg", "Beam") if c in df.columns]
        return df.sort_values(cols, na_position="last") if cols else df

    dfP_upper = _sort_safe(dfP_upper)
    dfP_lower = _sort_safe(dfP_lower)
    dfL_upper = _sort_safe(dfL_upper)
    dfL_lower = _sort_safe(dfL_lower)

    # Wide -> long + exclusions
    planU = apply_edge_exclusion(wide_to_long(dfP_upper, "upper", "plan"), "upper")
    planL = apply_edge_exclusion(wide_to_long(dfP_lower, "lower", "plan"), "lower")
    logU = apply_edge_exclusion(wide_to_long(dfL_upper, "upper", "log"), "upper")
    logL = apply_edge_exclusion(wide_to_long(dfL_lower, "lower", "log"), "lower")

    # Match id convention
    if not logU.empty and "Segment" in logU.columns:
        logU["match_id"] = pd.to_numeric(logU["Segment"], errors="coerce").astype("Int64")
    if not logL.empty and "Segment" in logL.columns:
        logL["match_id"] = pd.to_numeric(logL["Segment"], errors="coerce").astype("Int64")

    if not planU.empty and "Beam" in planU.columns:
        planU["match_id"] = pd.to_numeric(planU["Beam"], errors="coerce").astype("Int64")
    if not planL.empty and "Beam" in planL.columns:
        planL["match_id"] = pd.to_numeric(planL["Beam"], errors="coerce").astype("Int64")

    # Drop rows missing match keys before building nominal
    def _clean_keys(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        for c in ("GantryBin", "match_id", "leaf_pair"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        need = [c for c in ("GantryBin", "match_id", "leaf_pair") if c in df.columns]
        if need:
            df = df.dropna(subset=need)
        return df

    planU = _clean_keys(planU)
    planL = _clean_keys(planL)
    logU = _clean_keys(logU)
    logL = _clean_keys(logL)

    # Build nominal tables (unique per key)
    planU_nom = (
        planU.drop_duplicates(subset=["GantryBin", "match_id", "leaf_pair"], keep="first")
        .rename(columns={"left_mm": "left_mm_nom", "right_mm": "right_mm_nom"})
        .copy()
    )
    planL_nom = (
        planL.drop_duplicates(subset=["GantryBin", "match_id", "leaf_pair"], keep="first")
        .rename(columns={"left_mm": "left_mm_nom", "right_mm": "right_mm_nom"})
        .copy()
    )

    # Merge
    mU = merge_plan_log(logU, planU_nom, "upper")
    mL = merge_plan_log(logL, planL_nom, "lower")

    return {
        "planU": planU,
        "planL": planL,
        "logU": logU,
        "logL": logL,
        "planU_nom": planU_nom,
        "planL_nom": planL_nom,
        "mU": mU,
        "mL": mL,
    }

# # preprocess.py

# import re
# import numpy as np
# import pandas as pd

# UPPER_EDGE_EXCLUDE = {1, 2, 33, 34}
# LOWER_EDGE_EXCLUDE = {1, 2, 3, 33, 34, 35}

# def gantry_to_bin(deg, step=90):
#     if pd.isna(deg):
#         return np.nan
#     d = float(deg) % 360.0
#     b = (np.round(d / step) * step) % 360.0
#     if np.isclose(b, 360.0) or np.isclose(b, 0.0):
#         return 0.0
#     return float(b)

# def wide_to_long(df_wide: pd.DataFrame, table: str, prefix: str) -> pd.DataFrame:
#     assert table in ("upper", "lower")
#     assert prefix in ("plan", "log")

#     pairs = sorted({
#         int(re.search(r"Leaf_(\d+)_", c).group(1))
#         for c in df_wide.columns
#         if c.startswith("Leaf_") and re.search(r"Leaf_(\d+)_", c)
#     })

#     meta_cols = ["SourceFile", "Patient Name", "Patient ID", "Plan Name", "Date",
#                  "Beam", "Segment", "GantryDeg", "TotalMU"]
#     meta_cols = [c for c in meta_cols if c in df_wide.columns]

#     out = []
#     for _, row in df_wide.iterrows():
#         beam = int(row["Beam"]) if "Beam" in df_wide.columns and not pd.isna(row["Beam"]) else np.nan
#         seg  = int(row["Segment"]) if "Segment" in df_wide.columns and not pd.isna(row["Segment"]) else np.nan
#         gantry = float(row["GantryDeg"]) if "GantryDeg" in df_wide.columns and not pd.isna(row["GantryDeg"]) else np.nan
#         mu = float(row["TotalMU"]) if "TotalMU" in df_wide.columns and not pd.isna(row["TotalMU"]) else np.nan

#         gantry_bin = gantry_to_bin(gantry, step=90)

#         for p in pairs:
#             L = row.get(f"Leaf_{p}_Left(cm)", np.nan)
#             R = row.get(f"Leaf_{p}_Right(cm)", np.nan)
#             if pd.isna(L) and pd.isna(R):
#                 continue

#             rec = {c: row.get(c, None) for c in meta_cols}
#             rec.update({
#                 "prefix": prefix,
#                 "table": table,
#                 "Beam": beam,
#                 "Segment": seg,
#                 "GantryDeg": gantry,
#                 "GantryBin": gantry_bin,
#                 "TotalMU": mu,
#                 "leaf_pair": int(p),
#                 "left_cm": float(L),
#                 "right_cm": float(R),
#                 "left_mm": float(L) * 10.0,
#                 "right_mm": float(R) * 10.0,
#             })
#             out.append(rec)

#     return pd.DataFrame(out)

# def apply_edge_exclusion(df_long: pd.DataFrame, table: str) -> pd.DataFrame:
#     df_long = df_long.copy()
#     if table == "upper":
#         return df_long[~df_long["leaf_pair"].isin(UPPER_EDGE_EXCLUDE)].reset_index(drop=True)
#     if table == "lower":
#         return df_long[~df_long["leaf_pair"].isin(LOWER_EDGE_EXCLUDE)].reset_index(drop=True)
#     raise ValueError("table must be 'upper' or 'lower'")

# def merge_plan_log(log_df: pd.DataFrame, plan_nom: pd.DataFrame, table_label: str) -> pd.DataFrame:
#     keys = ["GantryBin", "match_id", "leaf_pair"]
#     merged = log_df.merge(
#         plan_nom[keys + ["left_mm_nom", "right_mm_nom"]],
#         on=keys,
#         how="inner",
#         validate="m:1"
#     )
#     merged["table"] = table_label
#     return merged

# def preprocess_and_merge(
#     dfP_upper: pd.DataFrame,
#     dfP_lower: pd.DataFrame,
#     dfL_upper: pd.DataFrame,
#     dfL_lower: pd.DataFrame,
#     drop_stack_by_plan_name: bool = True,
# ) -> dict:
#     """
#     Returns dict with:
#       planU, planL, logU, logL, planU_nom, planL_nom, mU, mL
#     """

#     # Optional: remove wrong stack by Plan Name convention
#     if drop_stack_by_plan_name and "Plan Name" in dfP_upper.columns and "Plan Name" in dfL_upper.columns:
#         dfP_upper = dfP_upper[~dfP_upper["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 2 g")]
#         dfL_upper = dfL_upper[~dfL_upper["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 2 g")]

#     if drop_stack_by_plan_name and "Plan Name" in dfP_lower.columns and "Plan Name" in dfL_lower.columns:
#         dfP_lower = dfP_lower[~dfP_lower["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 1 g")]
#         dfL_lower = dfL_lower[~dfL_lower["Plan Name"].astype(str).str.strip().str.lower().str.startswith("mlc 1 g")]

#     # Numeric + sort
#     for d in (dfP_upper, dfP_lower, dfL_upper, dfL_lower):
#         if "Beam" in d.columns: d["Beam"] = pd.to_numeric(d["Beam"], errors="coerce")
#         if "Segment" in d.columns: d["Segment"] = pd.to_numeric(d["Segment"], errors="coerce")
#         if "GantryDeg" in d.columns: d["GantryDeg"] = pd.to_numeric(d["GantryDeg"], errors="coerce")

#     dfP_upper = dfP_upper.sort_values(["GantryDeg", "Beam"], na_position="last")
#     dfP_lower = dfP_lower.sort_values(["GantryDeg", "Beam"], na_position="last")
#     dfL_upper = dfL_upper.sort_values(["GantryDeg", "Beam"], na_position="last")
#     dfL_lower = dfL_lower.sort_values(["GantryDeg", "Beam"], na_position="last")

#     # Wide -> long + exclusions
#     planU = apply_edge_exclusion(wide_to_long(dfP_upper, "upper", "plan"), "upper")
#     planL = apply_edge_exclusion(wide_to_long(dfP_lower, "lower", "plan"), "lower")
#     logU  = apply_edge_exclusion(wide_to_long(dfL_upper, "upper", "log"),  "upper")
#     logL  = apply_edge_exclusion(wide_to_long(dfL_lower, "lower", "log"),  "lower")

#     # PF match_id convention
#     logU["match_id"]  = logU["Segment"].astype(int)
#     logL["match_id"]  = logL["Segment"].astype(int)
#     planU["match_id"] = planU["Beam"].astype(int)
#     planL["match_id"] = planL["Beam"].astype(int)

#     planU_nom = (planU
#         .drop_duplicates(subset=["GantryBin", "match_id", "leaf_pair"], keep="first")
#         .rename(columns={"left_mm": "left_mm_nom", "right_mm": "right_mm_nom"})
#         .copy()
#     )
#     planL_nom = (planL
#         .drop_duplicates(subset=["GantryBin", "match_id", "leaf_pair"], keep="first")
#         .rename(columns={"left_mm": "left_mm_nom", "right_mm": "right_mm_nom"})
#         .copy()
#     )

#     mU = merge_plan_log(logU, planU_nom, "upper")
#     mL = merge_plan_log(logL, planL_nom, "lower")

#     return {
#         "planU": planU, "planL": planL,
#         "logU": logU, "logL": logL,
#         "planU_nom": planU_nom, "planL_nom": planL_nom,
#         "mU": mU, "mL": mL
#     }
