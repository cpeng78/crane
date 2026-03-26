from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from crane_lib.io import normalize_type_value  # noqa: E402
from crane_lib.matching import normalize_name  # noqa: E402


PROCESSED_DIR = ROOT / "processed_data"
DASH_DATA_DIR = ROOT / "dashapp" / "data"

MARKET_FILES = {
    "trend": DASH_DATA_DIR / "trend_data.parquet",
    "meta": DASH_DATA_DIR / "dataset_meta.parquet",
    "overview_holdings": DASH_DATA_DIR / "overview_holdings.parquet",
}
ISSUER_FILES = {
    "lookup": DASH_DATA_DIR / "issuer_lookup.parquet",
    "exposure": DASH_DATA_DIR / "issuer_exposure.parquet",
    "summary": DASH_DATA_DIR / "issuer_summary.parquet",
    "category": DASH_DATA_DIR / "issuer_category.parquet",
    "type": DASH_DATA_DIR / "issuer_type.parquet",
    "maturity": DASH_DATA_DIR / "issuer_maturity.parquet",
    "market_maturity": DASH_DATA_DIR / "market_maturity.parquet",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    group_column: str
    value_label: str
    source_note: str


DATASETS = [
    DatasetSpec("issuers", "Top Issuers", "group_name", "Issuer exposure", "Derived from Issuers summary"),
    DatasetSpec("composition", "Composition", "group_name", "Product mix", "Derived from Composition summary"),
    DatasetSpec("maturity", "Maturity Buckets", "group_name", "Tenor bucket", "Derived from HoldingList maturity dates"),
    DatasetSpec("country", "By Country", "group_name", "Country exposure", "Derived from By Country summary"),
    DatasetSpec("funds", "Top Funds", "group_name", "Fund assets", "Derived from HoldingList fund totals"),
]


def ensure_dash_data(force: bool = False) -> None:
    DASH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    required = list(MARKET_FILES.values()) + list(ISSUER_FILES.values())
    if all(path.exists() for path in required) and not force:
        return

    build_market_overview_holdings()

    frames = [
        build_issuer_trends(),
        build_composition_trends(),
        build_maturity_trends(),
        build_country_trends(),
        build_fund_trends(),
    ]
    trend_data = pd.concat(frames, ignore_index=True).sort_values(["dataset", "as_of_date", "group_name"])
    trend_data.to_parquet(MARKET_FILES["trend"], index=False)

    meta_rows = []
    for spec in DATASETS:
        frame = trend_data[trend_data["dataset"] == spec.key]
        latest_date = frame["as_of_date"].max()
        latest = frame[frame["as_of_date"] == latest_date].sort_values("value", ascending=False)
        meta_rows.append(
            {
                "dataset": spec.key,
                "label": spec.label,
                "value_label": spec.value_label,
                "source_note": spec.source_note,
                "min_date": frame["as_of_date"].min(),
                "max_date": latest_date,
                "group_count": frame["group_name"].nunique(),
                "default_groups": latest["group_name"].head(6).tolist(),
            }
        )
    pd.DataFrame(meta_rows).to_parquet(MARKET_FILES["meta"], index=False)

    build_issuer_detail_frames()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "nan", "None", "<NA>"}:
        return ""
    return text


def load_holdings_base() -> pd.DataFrame:
    columns = [
        "as_of_date",
        "release_date",
        "fund_type",
        "fund_family",
        "fund",
        "issuer",
        "holding",
        "category",
        "type",
        "country",
        "region",
        "sector",
        "value",
        "principal",
        "maturity_date",
        "final_maturity",
    ]
    holdings = pd.read_parquet(PROCESSED_DIR / "holdings.parquet", columns=columns).copy()
    holdings["as_of_date"] = pd.to_datetime(holdings["as_of_date"], errors="coerce")
    holdings["release_date"] = pd.to_datetime(holdings["release_date"], errors="coerce")
    holdings["maturity_date"] = pd.to_datetime(holdings["maturity_date"], errors="coerce")
    holdings["final_maturity"] = pd.to_datetime(holdings["final_maturity"], errors="coerce")
    for column in ["value", "principal"]:
        holdings[column] = pd.to_numeric(holdings[column], errors="coerce")
    for column in ["fund_type", "fund_family", "fund", "issuer", "holding", "category", "type", "country", "region", "sector"]:
        holdings[column] = holdings[column].map(clean_text)
    holdings = holdings.dropna(subset=["as_of_date", "value"])
    holdings["normalized_type"] = holdings["type"].map(normalize_type_value).astype(str)
    holdings["issuer_source"] = holdings["issuer"].where(holdings["issuer"] != "", holdings["holding"])
    holdings["issuer_norm"] = holdings["issuer_source"].map(normalize_name)
    holdings = holdings[holdings["issuer_norm"] != ""].copy()
    holdings["weight"] = holdings["principal"].where(holdings["principal"].gt(0), holdings["value"])
    holdings["days_to_maturity"] = (holdings["maturity_date"] - holdings["as_of_date"]).dt.days.clip(lower=0)
    holdings["days_to_final"] = (holdings["final_maturity"] - holdings["as_of_date"]).dt.days.clip(lower=0)
    holdings["weighted_maturity_component"] = holdings["days_to_maturity"] * holdings["weight"]
    holdings["weighted_final_component"] = holdings["days_to_final"] * holdings["weight"]
    return holdings


def build_market_overview_holdings() -> None:
    holdings = load_holdings_base()
    frame = holdings[
        [
            "as_of_date",
            "fund_type",
            "fund_family",
            "fund",
            "category",
            "type",
            "country",
            "region",
            "sector",
            "issuer_source",
            "maturity_date",
            "value",
        ]
    ].copy()
    frame = frame.rename(columns={"issuer_source": "issuer"})
    frame.to_parquet(MARKET_FILES["overview_holdings"], index=False)


def select_primary_labels(holdings: pd.DataFrame) -> pd.DataFrame:
    labels = (
        holdings.groupby(["issuer_norm", "issuer_source"], as_index=False)["value"]
        .sum()
        .sort_values(["issuer_norm", "value"], ascending=[True, False])
        .drop_duplicates("issuer_norm")
        .rename(columns={"issuer_source": "issuer_display", "value": "label_value"})
    )
    return labels[["issuer_norm", "issuer_display"]]


def build_issuer_detail_frames() -> None:
    holdings = load_holdings_base()
    labels = select_primary_labels(holdings)

    market_totals = holdings.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "market_total"})

    exposure = holdings.groupby(["as_of_date", "issuer_norm"], as_index=False)["value"].sum()
    exposure = exposure.merge(labels, on="issuer_norm", how="left")
    exposure = exposure.merge(market_totals, on="as_of_date", how="left")
    exposure["market_share"] = exposure["value"] / exposure["market_total"]
    exposure.to_parquet(ISSUER_FILES["exposure"], index=False)

    lookup_latest_date = exposure["as_of_date"].max()
    latest_exposure = (
        exposure[exposure["as_of_date"] == lookup_latest_date][["issuer_norm", "value", "market_share"]]
        .rename(columns={"value": "latest_value", "market_share": "latest_market_share"})
    )
    date_counts = exposure.groupby("issuer_norm", as_index=False)["as_of_date"].nunique().rename(columns={"as_of_date": "date_count"})
    lookup = labels.merge(latest_exposure, on="issuer_norm", how="left").merge(date_counts, on="issuer_norm", how="left")
    lookup = lookup.sort_values(["latest_value", "issuer_display"], ascending=[False, True]).reset_index(drop=True)
    lookup.to_parquet(ISSUER_FILES["lookup"], index=False)

    issuers = pd.read_parquet(PROCESSED_DIR / "issuers.parquet").copy()
    issuers["as_of_date"] = pd.to_datetime(issuers["as_of_date"], errors="coerce")
    issuers["issuer"] = issuers["issuer"].map(clean_text)
    issuers["issuer_norm"] = issuers["issuer"].map(normalize_name)
    issuer_summary = issuers.dropna(subset=["as_of_date"]).groupby(["as_of_date", "issuer_norm"], as_index=False).agg(
        summary_value=("value", "sum"),
        summary_share=("pct_total", "sum"),
    )
    issuer_summary.to_parquet(ISSUER_FILES["summary"], index=False)

    category = (
        holdings[holdings["category"] != ""]
        .groupby(["as_of_date", "issuer_norm", "category"], as_index=False)["value"]
        .sum()
    )
    category_totals = category.groupby(["as_of_date", "issuer_norm"], as_index=False)["value"].sum().rename(columns={"value": "issuer_total"})
    category = category.merge(category_totals, on=["as_of_date", "issuer_norm"], how="left")
    category["issuer_share"] = category["value"] / category["issuer_total"]
    category.to_parquet(ISSUER_FILES["category"], index=False)

    type_frame = holdings.groupby(["as_of_date", "issuer_norm", "normalized_type"], as_index=False)["value"].sum()
    type_totals = type_frame.groupby(["as_of_date", "issuer_norm"], as_index=False)["value"].sum().rename(columns={"value": "issuer_total"})
    type_frame = type_frame.merge(type_totals, on=["as_of_date", "issuer_norm"], how="left")
    type_frame["issuer_share"] = type_frame["value"] / type_frame["issuer_total"]
    type_frame = type_frame.rename(columns={"normalized_type": "group_name"})
    type_frame.to_parquet(ISSUER_FILES["type"], index=False)

    maturity = holdings.groupby(["as_of_date", "issuer_norm"], as_index=False).agg(
        exposure_value=("value", "sum"),
        weight_sum=("weight", "sum"),
        weighted_maturity_component=("weighted_maturity_component", "sum"),
        weighted_final_component=("weighted_final_component", "sum"),
    )
    maturity["wa_maturity_days"] = maturity["weighted_maturity_component"] / maturity["weight_sum"]
    maturity["wa_final_maturity_days"] = maturity["weighted_final_component"] / maturity["weight_sum"]
    maturity.to_parquet(ISSUER_FILES["maturity"], index=False)

    market_maturity = holdings.groupby("as_of_date", as_index=False).agg(
        total_value=("value", "sum"),
        weight_sum=("weight", "sum"),
        weighted_maturity_component=("weighted_maturity_component", "sum"),
        weighted_final_component=("weighted_final_component", "sum"),
    )
    market_maturity["wa_maturity_days"] = market_maturity["weighted_maturity_component"] / market_maturity["weight_sum"]
    market_maturity["wa_final_maturity_days"] = market_maturity["weighted_final_component"] / market_maturity["weight_sum"]
    market_maturity.to_parquet(ISSUER_FILES["market_maturity"], index=False)


def build_issuer_trends() -> pd.DataFrame:
    issuers = pd.read_parquet(PROCESSED_DIR / "issuers.parquet")
    frame = issuers.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date", "issuer", "value"])
    frame = frame.groupby(["as_of_date", "issuer"], as_index=False).agg(
        value=("value", "sum"),
        share=("pct_total", "sum"),
    )
    frame["dataset"] = "issuers"
    frame["group_name"] = frame["issuer"].astype(str)
    return frame[["dataset", "as_of_date", "group_name", "value", "share"]]


def build_composition_trends() -> pd.DataFrame:
    composition = pd.read_parquet(PROCESSED_DIR / "composition.parquet")
    frame = composition.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date", "category", "value"])
    frame = frame.groupby(["as_of_date", "category"], as_index=False).agg(
        value=("value", "sum"),
        share=("share", "sum"),
    )
    frame["dataset"] = "composition"
    frame["group_name"] = frame["category"].astype(str)
    return frame[["dataset", "as_of_date", "group_name", "value", "share"]]


def build_maturity_trends() -> pd.DataFrame:
    holdings = pd.read_parquet(PROCESSED_DIR / "holdings.parquet", columns=["as_of_date", "value", "maturity_date"])
    frame = holdings.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["maturity_date"] = pd.to_datetime(frame["maturity_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date", "value", "maturity_date"])
    days = (frame["maturity_date"] - frame["as_of_date"]).dt.days.clip(lower=0)
    bins = [-1, 1, 7, 30, 60, 90, 180, 365, 10_000]
    labels = [
        "Overnight",
        "2 - 7 Days",
        "8 - 30 Days",
        "31 - 60 Days",
        "61 - 90 Days",
        "91 - 180 Days",
        "181 - 365 Days",
        "366 + Days",
    ]
    frame["group_name"] = pd.cut(days, bins=bins, labels=labels)
    frame = frame.dropna(subset=["group_name"])
    grouped = frame.groupby(["as_of_date", "group_name"], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "total_value"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["share"] = grouped["value"] / grouped["total_value"]
    grouped["dataset"] = "maturity"
    return grouped[["dataset", "as_of_date", "group_name", "value", "share"]]


def build_country_trends() -> pd.DataFrame:
    country = pd.read_parquet(PROCESSED_DIR / "country.parquet")
    frame = country.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    missing_mask = frame["as_of_date"].isna() & frame["release_date"].notna()
    frame.loc[missing_mask, "as_of_date"] = frame.loc[missing_mask, "release_date"] - pd.offsets.MonthEnd(1)
    frame = frame.dropna(subset=["as_of_date", "country", "value"])
    frame = frame.groupby(["as_of_date", "country"], as_index=False).agg(
        value=("value", "sum"),
        share=("pct_total", "sum"),
    )
    frame["dataset"] = "country"
    frame["group_name"] = frame["country"].astype(str)
    return frame[["dataset", "as_of_date", "group_name", "value", "share"]]


def build_fund_trends() -> pd.DataFrame:
    holdings = pd.read_parquet(PROCESSED_DIR / "holdings.parquet", columns=["as_of_date", "fund", "value"])
    frame = holdings.copy()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce")
    frame = frame.dropna(subset=["as_of_date", "fund", "value"])
    grouped = frame.groupby(["as_of_date", "fund"], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "total_value"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["share"] = grouped["value"] / grouped["total_value"]
    grouped["dataset"] = "funds"
    grouped["group_name"] = grouped["fund"].astype(str)
    return grouped[["dataset", "as_of_date", "group_name", "value", "share"]]


if __name__ == "__main__":
    ensure_dash_data(force=True)
    print(f"Saved dash data to {DASH_DATA_DIR}")
