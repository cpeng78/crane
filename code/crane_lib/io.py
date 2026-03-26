from __future__ import annotations

from typing import Iterable

import pandas as pd

from .constants import FUNDING_CATEGORIES, PROCESSED_DIR


def normalize_type_value(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text or text in {"-", "nan", "None", "<NA>"}:
        return "Other"

    upper = text.upper()
    if "REPURCHASE" in upper or upper == "REPO":
        return "Repo"
    if "CERTIFICATE" in upper or upper == "CD":
        return "CD"
    if "COMMERCIAL PAPER" in upper or upper == "CP":
        return "CP"
    if "TREASURY" in upper or "GOVERNMENT BOND" in upper:
        return "Treasury"
    if "AGENC" in upper:
        return "Agency"
    if "VRDN" in upper or "VARIABLE RATE DEMAND NOTE" in upper:
        return "VRDN"
    if "MUNICIPAL BOND" in upper:
        return "Municipal Bond"
    if upper == "OTHER":
        return "Other"
    return text


def read_prefer_full(name: str, columns: Iterable[str] | None = None) -> pd.DataFrame:
    full_path = PROCESSED_DIR / f"{name}.parquet"
    event_path = PROCESSED_DIR / f"{name}_event.parquet"
    path = full_path if full_path.exists() else event_path
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def prepare_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "as_of_date" in out.columns:
        out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    if "release_date" in out.columns:
        out["release_date"] = pd.to_datetime(out["release_date"], errors="coerce")

    for column in ["holding", "issuer", "category", "type", "fund", "source_file"]:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].astype(str)

    for column in ["value", "coupon"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in ["maturity_date", "final_maturity"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")

    out = out.dropna(subset=["as_of_date"])
    out["is_funding_category"] = out["category"].isin(FUNDING_CATEGORIES)
    out["normalized_type"] = out["type"].map(normalize_type_value).astype(str)
    return out


def prepare_generic(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in ["as_of_date", "release_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    if "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    if "source_file" in out.columns:
        out["source_file"] = out["source_file"].astype(str)
    return out


def load_holdings(columns: Iterable[str] | None = None) -> pd.DataFrame:
    return prepare_holdings(read_prefer_full("holdings", columns=columns))


def load_analysis_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = load_holdings()
    issuers = prepare_generic(read_prefer_full("issuers"))
    composition = prepare_generic(read_prefer_full("composition"))
    country = prepare_generic(read_prefer_full("country"))
    return holdings, issuers, composition, country
