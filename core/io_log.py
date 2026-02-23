from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple, Union, Optional

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


# =============================================================================
# 1) Core parser for ONE log (from text)
# =============================================================================
def extract_log_text(text: str, source_name: str = "uploaded.txt") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse one ViewRay Treatment Delivery Record from raw text.
    Returns: df_all, df_upper, df_lower
    """
    # -----------------------
    # Header extraction
    # -----------------------
    RX_PATIENT_ID = re.compile(r"Patient\s*ID\s*:\s*(.+)", re.I)
    RX_PATIENT_NM = re.compile(r"Patient\s*Name\s*:\s*(.+)", re.I)
    RX_DELIV_DATE = re.compile(r"Delivered\s*on\s*:\s*([A-Za-z]{3}\s+\d{1,2}\s+\d{4})", re.I)
    RX_PLAN_NAME  = re.compile(r"^\s*Plan\s*Name\s*:\s*([^\r\n]+)\s*$", re.I | re.M)

    def find_first(regex):
        m = regex.search(text)
        return m.group(1).strip() if m else None

    patient_id    = find_first(RX_PATIENT_ID)
    patient_name  = find_first(RX_PATIENT_NM)
    plan_name     = find_first(RX_PLAN_NAME)
    delivery_date = find_first(RX_DELIV_DATE)

    # Rename specific plan
    if plan_name == "MLC G180 VCU":
        plan_name = "MLC 2 G180 VCU"

    # -----------------------
    # Section patterns
    # -----------------------
    RX_BEAM  = re.compile(r"^\s*Beam\s*:\s*(\d+)", re.I)
    RX_ANGLE = re.compile(r"^\s*Angle\s*:\s*([+-]?\d+(?:[.,]\d+)?)", re.I)
    RX_SEGID = re.compile(r"^\s*Segment\s*ID\s*:\s*(\d+)", re.I)
    RX_MU    = re.compile(r"^\s*Monitor\s+Units\s+per\s+Fraction\s*:\s*([+-]?\d+(?:[.,]\d+)?)", re.I)

    RX_LEAF_HDR  = re.compile(r"^\s*Leaf\s+Positions\s*$", re.I)
    RX_TABLE_HDR = re.compile(
        r"^\s*Leaf\s+Pair\s+Left\s+Location\s*\(cm\)\s+Right\s+Location\s*\(cm\)\s*$", re.I
    )
    RX_LEAF_ROW = re.compile(r"^\s*(\d+)\s+([+-]?\d+(?:[.,]\d+)?)\s+([+-]?\d+(?:[.,]\d+)?)\s*$")

    def fnum(s) -> float:
        return float(str(s).replace(",", "."))

    lines = text.splitlines()
    n = len(lines)
    i = 0

    rows = []

    # current metadata (updated as we parse)
    beam: Optional[int] = None
    gantry: Optional[float] = None
    segment: Optional[int] = None
    total_mu: Optional[float] = None

    def read_exact_leaf_rows(start_idx: int, expected_count: int, ctx_label: str):
        out = {}
        idx = start_idx
        for _ in range(expected_count):
            if idx >= n:
                raise ValueError(f"{ctx_label}: expected {expected_count} leaf rows, hit EOF.")
            line = lines[idx].rstrip("\n")
            m = RX_LEAF_ROW.match(line)
            if not m:
                raise ValueError(f"{ctx_label}: bad leaf row at line {idx+1}: {line}")
            pair = int(m.group(1))
            L = fnum(m.group(2))
            R = fnum(m.group(3))
            out[pair] = (L, R)
            idx += 1
        return idx, out

    def skip_blank(idx: int) -> int:
        while idx < n and not lines[idx].strip():
            idx += 1
        return idx

    def emit_row(table_pairs: dict, is_upper: bool):
        expected = range(1, 35) if is_upper else range(1, 36)

        row = {
            "SourceFile": source_name,
            "Patient Name": patient_name,
            "Patient ID": patient_id,
            "Plan Name": plan_name,
            "Date": delivery_date,
            "Beam": beam,
            "Segment": segment,
            "GantryDeg": float(gantry) if gantry is not None else np.nan,
            "TotalMU": float(total_mu) if total_mu is not None else np.nan,
            "TableOrder": 0 if is_upper else 1,
        }

        for p in expected:
            if p in table_pairs:
                L, R = table_pairs[p]
                row[f"Leaf_{p}_Left(cm)"] = L
                row[f"Leaf_{p}_Right(cm)"] = R

        rows.append(row)

    # -----------------------
    # Main scan
    # -----------------------
    while i < n:
        line = lines[i].rstrip("\n")

        if (m := RX_BEAM.match(line)):
            beam = int(m.group(1)); i += 1; continue
        if (m := RX_ANGLE.match(line)):
            gantry = fnum(m.group(1)); i += 1; continue
        if (m := RX_SEGID.match(line)):
            segment = int(m.group(1)); i += 1; continue
        if (m := RX_MU.match(line)):
            total_mu = fnum(m.group(1)); i += 1; continue

        if RX_LEAF_HDR.match(line):
            i += 1
            if i >= n or not RX_TABLE_HDR.match(lines[i]):
                raise ValueError(f"{source_name} | Segment {segment}: missing leaf table header.")
            i += 1

            ctx = f"{source_name} | Segment {segment} Beam {beam} (UPPER)"
            i, upper_pairs = read_exact_leaf_rows(i, 34, ctx)

            i = skip_blank(i)

            ctx = f"{source_name} | Segment {segment} Beam {beam} (LOWER)"
            i, lower_pairs = read_exact_leaf_rows(i, 35, ctx)

            emit_row(upper_pairs, True)
            emit_row(lower_pairs, False)
            continue

        i += 1

    # -----------------------
    # Build outputs
    # -----------------------
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df.copy(), df.copy()

    base_cols = [
        "SourceFile", "Patient Name", "Patient ID", "Plan Name", "Date",
        "Beam", "Segment", "GantryDeg", "TotalMU", "TableOrder"
    ]
    leaf_cols = sorted(
        [c for c in df.columns if c.startswith("Leaf_")],
        key=lambda x: int(re.search(r"Leaf_(\d+)_", x).group(1))
    )

    df_all = df[base_cols + leaf_cols]
    df_all = df_all.sort_values(by=["Segment", "TableOrder"], kind="stable").reset_index(drop=True)

    df_upper = df_all[df_all["TableOrder"] == 0].drop(columns=["TableOrder"]).reset_index(drop=True)
    df_lower = df_all[df_all["TableOrder"] == 1].drop(columns=["TableOrder"]).reset_index(drop=True)

    # Drop last 2 columns from upper (final leaf pair)
    if df_upper.shape[1] > (len(base_cols) - 1):
        df_upper = df_upper.iloc[:, :-2]

    df_all = df_all.drop(columns=["TableOrder"])
    return df_all, df_upper, df_lower


def extract_log_file(log_path: PathLike) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(log_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return extract_log_text(text, source_name=path.name)


# =============================================================================
# 2) Batch loader (for Streamlit multi-upload or folder batch offline)
# =============================================================================
def extract_logs_texts(
    texts_and_names: Iterable[Tuple[str, str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_all, all_upper, all_lower = [], [], []

    for text, name in texts_and_names:
        df_all, df_upper, df_lower = extract_log_text(text, source_name=name)
        all_all.append(df_all)
        all_upper.append(df_upper)
        all_lower.append(df_lower)

    if not all_all:
        raise FileNotFoundError("No log texts provided to parse.")

    return (
        pd.concat(all_all, ignore_index=True),
        pd.concat(all_upper, ignore_index=True),
        pd.concat(all_lower, ignore_index=True),
    )
