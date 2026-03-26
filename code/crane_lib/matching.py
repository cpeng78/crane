from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


def normalize_name(text: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_bofa_issuer(text: str) -> bool:
    normalized = normalize_name(text)
    return normalized.startswith("bank of america") or normalized.startswith("banc of america")


def complete_monthly_series(
    frame: pd.DataFrame,
    all_dates: pd.Series | pd.Index,
    value_col: str,
) -> pd.DataFrame:
    out = frame.set_index("as_of_date").reindex(pd.Index(all_dates, name="as_of_date")).reset_index()
    out[value_col] = out[value_col].fillna(0.0)
    return out


def sort_by_date(frame: pd.DataFrame) -> pd.DataFrame:
    if "as_of_date" not in frame.columns:
        return frame.copy()
    return frame.sort_values("as_of_date").reset_index(drop=True)


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    mask = series.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return math.nan
    return float(np.average(series[mask], weights=weights[mask]))
