from __future__ import annotations

import base64
import re
import sys
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dash_table, dcc, html, no_update

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

from crane_lib.constants import CATEGORY_COLORS, CATEGORY_SHORT_LABELS, TYPE_COLORS  # noqa: E402
from crane_lib.matching import normalize_name  # noqa: E402
from prepare_dash_data import DASH_DATA_DIR, DATASETS, ensure_dash_data  # noqa: E402


APP_TITLE = "Crane Insight"
APP_DESCRIPTION = "Professional dashboard for historical money fund holdings analysis across issuers, funds, countries, product mix, and maturity."
PORT = 8052
NAV_ITEMS = [
    ("/", "Market Overview"),
    ("/market-breakdown", "Market Breakdown"),
    ("/country", "Country"),
    ("/maturity", "Maturity"),
    ("/issuer", "Issuer Deep Dive"),
    ("/fund", "Fund Deep Dive"),
]
PALETTE = [
    "#0F766E",
    "#1D4ED8",
    "#D97706",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#65A30D",
    "#DB2777",
    "#4B5563",
    "#8B5CF6",
    "#EA580C",
    "#059669",
]
OVERVIEW_DATASETS = ["issuers", "composition", "country", "funds"]
OVERVIEW_GROUP_COLUMNS = {
    "issuers": "issuer",
    "composition": "category",
    "country": "country",
    "region": "region",
    "funds": "fund",
}
OVERVIEW_DATASET_LABELS = {
    "issuers": "Top Issuers",
    "composition": "Portfolio Composition",
    "country": "Country Breakout",
    "region": "Region Breakout",
    "funds": "Top Funds",
}
OVERVIEW_FILTER_SPECS = [
    ("fund_type", "Fund Type", "ov-filter-fund-type"),
    ("fund_family", "Fund Family", "ov-filter-fund-family"),
    ("fund", "Fund", "ov-filter-fund"),
    ("category", "Category", "ov-filter-category"),
    ("type", "Type", "ov-filter-type"),
    ("country", "Country", "ov-filter-country"),
    ("region", "Region", "ov-filter-region"),
    ("sector", "Issuer Sector", "ov-filter-sector"),
]
DEFAULT_EXCLUDE = {
    "composition": {"Grand Total", "Total Holdings"},
    "country": {"Global", "Grand Total", "Americas", "Europe", "Eurozone", "Asia & Pacific", "Other"},
}
DATASET_HELP = {
    "issuers": "Top-issuer lines are rebuilt as a historical view from the monthly issuer summary.",
    "composition": "Default view suppresses workbook roll-ups so the chart opens on detailed product categories.",
    "funds": "Fund trends are rebuilt from HoldingList, so the panel can extend beyond the single-month top-10 table.",
    "country": "Default view suppresses regional totals so the chart opens on country-level exposure lines.",
    "maturity": "Buckets are recomputed from HoldingList maturity dates and mapped to the same tenor buckets used in FundCharts.",
}
METRIC_OPTIONS = [
    {"label": "USD bn", "value": "value"},
    {"label": "Share of total", "value": "share"},
]
CHART_OPTIONS = [
    {"label": "Line", "value": "line"},
    {"label": "Stacked Area", "value": "area"},
]
PRESET_OPTIONS = [
    {"label": "Custom", "value": "custom"},
    {"label": "Top 5", "value": "top_5"},
    {"label": "Top 10", "value": "top_10"},
]
OVERVIEW_PRESET_OPTIONS = [
    {"label": "Top 5", "value": "top_5"},
    {"label": "Top 10", "value": "top_10"},
    {"label": "All", "value": "all"},
]
BREAKDOWN_PRESET_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Top 5", "value": "top_5"},
    {"label": "Top 10", "value": "top_10"},
]
ISSUER_GROUP_OPTIONS = [
    {"label": "Category", "value": "category"},
    {"label": "Type", "value": "type"},
]
ISSUER_PRESET_OPTIONS = [
    {"label": "All", "value": "all"},
    {"label": "Top 5", "value": "top_5"},
    {"label": "Custom", "value": "custom"},
]
EVENT_PRESET_OPTIONS = [
    {"label": "CS stress (2022-08 to 2023-04)", "value": "cs_stress"},
    {"label": "Rate hikes (2022-01 to 2023-12)", "value": "rate_hikes"},
    {"label": "DB 2016 (2016-06 to 2017-03)", "value": "db_2016"},
    {"label": "BAC 2011 (2011-02 to 2012-03)", "value": "bac_2011"},
    {"label": "Full sample", "value": "full_sample"},
    {"label": "Last 24 months", "value": "last_24m"},
    {"label": "Custom", "value": "custom"},
]
EVENT_WINDOWS = {
    "cs_stress": ("2022-08-31", "2023-04-30"),
    "rate_hikes": ("2022-01-31", "2023-12-31"),
    "db_2016": ("2016-06-30", "2017-03-31"),
    "bac_2011": ("2011-02-28", "2012-03-31"),
}
GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True,
    "doubleClick": "reset",
}
FIXED_TREND_HEIGHT = 560
COMPACT_TREND_HEIGHT = 430
RANGE_SLIDER_STYLE = {
    "visible": True,
    "thickness": 0.1,
    "bgcolor": "#F4F7FB",
    "bordercolor": "rgba(22,48,74,0.08)",
}
BLANK_FILTER_VALUE = "(Blank)"
COUNTRY_NAME_MAP = {
    "US": "United States",
    "U.S.": "United States",
    "UNITED STATES": "United States",
    "GB": "United Kingdom",
    "U.K.": "United Kingdom",
    "SCOTLAND": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "CA": "Canada",
    "FR": "France",
    "JP": "Japan",
    "CH": "Switzerland",
    "AU": "Australia",
    "DE": "Germany",
    "NL": "Netherlands",
    "SE": "Sweden",
    "NO": "Norway",
    "ES": "Spain",
    "FI": "Finland",
    "BE": "Belgium",
    "DK": "Denmark",
    "IE": "Ireland",
    "AT": "Austria",
    "SG": "Singapore",
    "CN": "China",
    "CL": "Chile",
    "VE": "Venezuela",
    "LU": "Luxembourg",
    "IN": "India",
    "IT": "Italy",
    "KR": "South Korea",
    "KOREA": "South Korea",
    "MY": "Malaysia",
    "KW": "Kuwait",
    "QA": "Qatar",
    "SA": "Saudi Arabia",
    "AE": "United Arab Emirates",
    "ABU DHABI": "United Arab Emirates",
    "NEW ZEALAND": "New Zealand",
    "XS": "Supranational",
    "SU": "Supranational",
}
CATEGORY_NAME_MAP = {
    "Treasury Debt": "U.S. Treasury Debt",
    "Government Agency Debt": "U.S. Government Agency Debt",
    "Treasury Repurchase Agreement": "U.S. Treasury Repurchase Agreement",
    "Government Agency Repurchase Agreement": "U.S. Government Agency Repurchase Agreement",
    "Other Commercial Paper": "Non-Financial Company Commercial Paper",
    "Other Instrument (Time Deposit)": "Non-Negotiable Time Deposit",
}


def clean_filter_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "-", "(Blank)", "blank", "None", "nan"}:
        return ""
    return text


def normalize_country_name(value: object) -> str:
    text = clean_filter_text(value)
    if not text:
        return ""
    mapped = COUNTRY_NAME_MAP.get(text.upper(), text)
    if mapped.isupper():
        return mapped.title()
    return mapped


def normalize_category_name(value: object) -> str:
    text = clean_filter_text(value)
    if not text:
        return ""
    return CATEGORY_NAME_MAP.get(text, text)


def derive_issuer_display_label(issuer: object, holding: object) -> str:
    issuer_text = clean_filter_text(issuer)
    holding_text = clean_filter_text(holding)

    extraction_patterns = [
        r"Agreement with ([^,]+)",
        r"TRI-PARTY REPURCHASE AGREEMENT WITH ([^,]+)",
        r"In a joint trading account with (.+?)(?: at|,|$)",
        r"In a joint trading account with ([^,]+?) at",
        r"In a joint trading account with ([^,]+)$",
        r"TriParty Repurchase Agreement \(([^)]+)\)",
    ]
    for source in [issuer_text, holding_text]:
        for pattern in extraction_patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                label = clean_filter_text(match.group(1))
                if label:
                    return label

    for vrdn_marker in [" Weekly VRDN", " Daily VRDN", " Weekly VRDNs", " Daily VRDNs", " VRDN LOC ", " VRDN"]:
        source = issuer_text or holding_text
        if vrdn_marker in source:
            candidate = clean_filter_text(source.split(vrdn_marker, 1)[0].strip(" -("))
            if candidate:
                return candidate

    rate_markers = [
        r"^(.*?),\s*\d+(?:\.\d+)?%\s+dated\s+",
        r"^(.*?)\s+at\s+\d+(?:\.\d+)?%,\s+dated\s+",
        r"^(.*?)\s+at\s+\d+(?:\.\d+)?%\s+dated\s+",
        r"^(.*?),\s*\d+(?:\.\d+)?%$",
        r"^(.*?)\s+at\s+\d+(?:\.\d+)?%,?$",
    ]
    for source in [issuer_text, holding_text]:
        for pattern in rate_markers:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                candidate = clean_filter_text(match.group(1))
                if candidate:
                    return candidate

    if issuer_text and len(issuer_text) <= 80:
        return issuer_text

    source = issuer_text or holding_text
    if "Floating Rate Notes" in source:
        candidate = clean_filter_text(source.split("Floating Rate Notes", 1)[0].strip(" -,"))
        if candidate:
            return candidate

    if "Support Agreement" in source:
        candidate = clean_filter_text(source.split("Support Agreement", 1)[0].strip(" -,("))
        if candidate:
            return candidate

    if issuer_text:
        return issuer_text[:80].rstrip()
    return holding_text[:80].rstrip()

ISSUER_PRESETS = {
    "preset::credit_suisse": {
        "label": "Credit Suisse",
        "patterns": ["credit suisse", "csfb"],
        "match_mode": "contains",
        "auto_window": "cs_stress",
    },
    "preset::bank_of_america": {
        "label": "Bank of America",
        "patterns": ["bank of america", "banc of america"],
        "match_mode": "startswith",
        "auto_window": "rate_hikes",
    },
    "preset::deutsche_bank": {
        "label": "Deutsche Bank",
        "patterns": ["deutsche bank"],
        "match_mode": "startswith",
        "auto_window": "db_2016",
    },
    "preset::ubs": {
        "label": "UBS",
        "patterns": ["ubs"],
        "match_mode": "startswith",
        "auto_window": "last_24m",
    },
}


ensure_dash_data(force=False)
TREND_DATA = pd.read_parquet(DASH_DATA_DIR / "trend_data.parquet")
META = pd.read_parquet(DASH_DATA_DIR / "dataset_meta.parquet")
OVERVIEW_HOLDINGS = pd.read_parquet(DASH_DATA_DIR / "overview_holdings.parquet")
FUND_DETAIL_HOLDINGS = pd.read_parquet(
    PROJECT_ROOT / "processed_data" / "holdings.parquet",
    columns=[
        "as_of_date",
        "fund",
        "fund_family",
        "fund_type",
        "issuer",
        "holding",
        "category",
        "type",
        "value",
        "principal",
        "maturity_date",
        "final_maturity",
    ],
)
ISSUER_LOOKUP = pd.read_parquet(DASH_DATA_DIR / "issuer_lookup.parquet")
ISSUER_EXPOSURE = pd.read_parquet(DASH_DATA_DIR / "issuer_exposure.parquet")
ISSUER_SUMMARY = pd.read_parquet(DASH_DATA_DIR / "issuer_summary.parquet")
ISSUER_CATEGORY = pd.read_parquet(DASH_DATA_DIR / "issuer_category.parquet")
ISSUER_TYPE = pd.read_parquet(DASH_DATA_DIR / "issuer_type.parquet")
ISSUER_MATURITY = pd.read_parquet(DASH_DATA_DIR / "issuer_maturity.parquet")
MARKET_MATURITY = pd.read_parquet(DASH_DATA_DIR / "market_maturity.parquet")

for frame in [TREND_DATA, META, OVERVIEW_HOLDINGS, FUND_DETAIL_HOLDINGS, ISSUER_EXPOSURE, ISSUER_SUMMARY, ISSUER_CATEGORY, ISSUER_TYPE, ISSUER_MATURITY, MARKET_MATURITY]:
    for column in ["as_of_date", "min_date", "max_date"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
for column in ["maturity_date", "final_maturity"]:
    if column in FUND_DETAIL_HOLDINGS.columns:
        FUND_DETAIL_HOLDINGS[column] = pd.to_datetime(FUND_DETAIL_HOLDINGS[column], errors="coerce")

for column in ["fund_type", "fund_family", "fund", "category", "type", "region", "sector", "issuer"]:
    if column in OVERVIEW_HOLDINGS.columns:
        OVERVIEW_HOLDINGS[column] = OVERVIEW_HOLDINGS[column].map(clean_filter_text)
for column in ["fund_type", "fund_family", "fund", "issuer", "holding", "category", "type"]:
    if column in FUND_DETAIL_HOLDINGS.columns:
        FUND_DETAIL_HOLDINGS[column] = FUND_DETAIL_HOLDINGS[column].map(clean_filter_text)
if "country" in OVERVIEW_HOLDINGS.columns:
    OVERVIEW_HOLDINGS["country"] = OVERVIEW_HOLDINGS["country"].map(normalize_country_name)
if "category" in OVERVIEW_HOLDINGS.columns:
    OVERVIEW_HOLDINGS["category"] = OVERVIEW_HOLDINGS["category"].map(normalize_category_name)
if "issuer" in OVERVIEW_HOLDINGS.columns:
    OVERVIEW_HOLDINGS["issuer_norm"] = OVERVIEW_HOLDINGS["issuer"].map(normalize_name)
if "category" in FUND_DETAIL_HOLDINGS.columns:
    FUND_DETAIL_HOLDINGS["category"] = FUND_DETAIL_HOLDINGS["category"].map(normalize_category_name)
if "issuer" in FUND_DETAIL_HOLDINGS.columns:
    FUND_DETAIL_HOLDINGS["issuer_display"] = [
        derive_issuer_display_label(issuer, holding)
        for issuer, holding in zip(FUND_DETAIL_HOLDINGS["issuer"], FUND_DETAIL_HOLDINGS["holding"])
    ]
    FUND_DETAIL_HOLDINGS["issuer_norm"] = FUND_DETAIL_HOLDINGS["issuer_display"].map(normalize_name)

GLOBAL_MIN_DATE = TREND_DATA["as_of_date"].min().date()
GLOBAL_MAX_DATE = TREND_DATA["as_of_date"].max().date()
ALL_DATES = pd.Index(sorted(TREND_DATA["as_of_date"].dropna().unique()))
DEFAULT_ISSUER_END = GLOBAL_MAX_DATE
DEFAULT_ISSUER_START = (pd.Timestamp(GLOBAL_MAX_DATE) - pd.DateOffset(months=24)).date()


def canonicalize_type_name(value: object) -> str:
    text = clean_filter_text(value)
    if not text:
        return ""
    lower = text.lower()
    special = {
        "fnma": "FNMA",
        "repo": "Repo",
        "cp": "CP",
        "cd": "CD",
        "vrdn": "VRDN",
    }
    if lower in special:
        return special[lower]
    if text.isupper():
        return text.title()
    return text


if "group_name" in ISSUER_TYPE.columns:
    ISSUER_TYPE["group_name"] = ISSUER_TYPE["group_name"].map(canonicalize_type_name)
    ISSUER_TYPE = (
        ISSUER_TYPE.groupby(["as_of_date", "issuer_norm", "group_name"], as_index=False)["value"]
        .sum()
        .merge(
            ISSUER_TYPE.groupby(["as_of_date", "issuer_norm"], as_index=False)["value"].sum().rename(columns={"value": "issuer_total"}),
            on=["as_of_date", "issuer_norm"],
            how="left",
        )
    )
    ISSUER_TYPE["issuer_share"] = ISSUER_TYPE["value"] / ISSUER_TYPE["issuer_total"]

if "type" in OVERVIEW_HOLDINGS.columns:
    OVERVIEW_HOLDINGS["type"] = OVERVIEW_HOLDINGS["type"].map(canonicalize_type_name)
if "type" in FUND_DETAIL_HOLDINGS.columns:
    FUND_DETAIL_HOLDINGS["type"] = FUND_DETAIL_HOLDINGS["type"].map(canonicalize_type_name)
for column in ["value", "principal"]:
    if column in FUND_DETAIL_HOLDINGS.columns:
        FUND_DETAIL_HOLDINGS[column] = pd.to_numeric(FUND_DETAIL_HOLDINGS[column], errors="coerce")
FUND_DETAIL_HOLDINGS["weight"] = FUND_DETAIL_HOLDINGS["principal"].where(FUND_DETAIL_HOLDINGS["principal"].gt(0), FUND_DETAIL_HOLDINGS["value"])
FUND_DETAIL_HOLDINGS["days_to_maturity"] = (FUND_DETAIL_HOLDINGS["maturity_date"] - FUND_DETAIL_HOLDINGS["as_of_date"]).dt.days.clip(lower=0)
FUND_DETAIL_HOLDINGS["days_to_final"] = (FUND_DETAIL_HOLDINGS["final_maturity"] - FUND_DETAIL_HOLDINGS["as_of_date"]).dt.days.clip(lower=0)
FUND_DETAIL_HOLDINGS["weighted_maturity_component"] = FUND_DETAIL_HOLDINGS["days_to_maturity"] * FUND_DETAIL_HOLDINGS["weight"]
FUND_DETAIL_HOLDINGS["weighted_final_component"] = FUND_DETAIL_HOLDINGS["days_to_final"] * FUND_DETAIL_HOLDINGS["weight"]

DATASET_OPTIONS = [{"label": OVERVIEW_DATASET_LABELS[spec.key], "value": spec.key} for spec in DATASETS if spec.key in OVERVIEW_DATASETS]
COUNTRY_OPTIONS = [{"label": "By Country", "value": "country"}]
MATURITY_GROUPS = sorted(TREND_DATA[TREND_DATA["dataset"] == "maturity"]["group_name"].dropna().unique().tolist())
ISSUER_OPTION_FRAME = ISSUER_LOOKUP[(ISSUER_LOOKUP["latest_value"].fillna(0) >= 10_000_000) & (ISSUER_LOOKUP["date_count"].fillna(0) >= 6)].copy()
latest_exposure_date = ISSUER_EXPOSURE["as_of_date"].max()


def format_latest_exposure_label(value: float) -> str:
    if pd.isna(value) or value <= 0:
        return "0.0bn"
    if value < 50_000_000:
        return "<0.1bn"
    return f"{value / 1e9:,.1f}bn"


def preset_latest_exposure(patterns: list[str], match_mode: str) -> float:
    matched_norms: set[str] = set()
    for pattern in patterns:
        if match_mode == "startswith":
            mask = ISSUER_LOOKUP["issuer_norm"].str.startswith(pattern)
        else:
            mask = ISSUER_LOOKUP["issuer_norm"].str.contains(pattern, regex=False)
        matched_norms.update(ISSUER_LOOKUP.loc[mask, "issuer_norm"].tolist())
    if not matched_norms:
        return 0.0
    matched = ISSUER_EXPOSURE[ISSUER_EXPOSURE["issuer_norm"].isin(matched_norms)]
    if matched.empty:
        return 0.0
    matched = matched[matched["value"] > 0]
    if matched.empty:
        return 0.0
    latest_date = matched["as_of_date"].max()
    latest = matched[matched["as_of_date"] == latest_date]
    return float(latest["value"].sum())


ISSUER_DROPDOWN_OPTIONS = [
    {
        "label": f"{preset['label']} ({format_latest_exposure_label(preset_latest_exposure(preset['patterns'], preset.get('match_mode', 'startswith')))} latest)",
        "value": value,
    }
    for value, preset in ISSUER_PRESETS.items()
]
ISSUER_DROPDOWN_OPTIONS += [
    {
        "label": f"{row['issuer_display']} ({format_latest_exposure_label(row['latest_value'])} latest)",
        "value": f"norm::{row['issuer_norm']}",
    }
    for _, row in ISSUER_OPTION_FRAME.iterrows()
]

FUND_OPTION_FRAME = (
    OVERVIEW_HOLDINGS.groupby(["fund", "as_of_date"], as_index=False)["value"]
    .sum()
    .sort_values(["fund", "as_of_date"])
)
if not FUND_OPTION_FRAME.empty:
    fund_counts = FUND_OPTION_FRAME.groupby("fund", as_index=False)["as_of_date"].nunique().rename(columns={"as_of_date": "date_count"})
    latest_fund_date = FUND_OPTION_FRAME["as_of_date"].max()
    latest_fund_assets = (
        FUND_OPTION_FRAME[FUND_OPTION_FRAME["as_of_date"] == latest_fund_date][["fund", "value"]]
        .rename(columns={"value": "latest_value"})
    )
    FUND_OPTION_FRAME = (
        FUND_OPTION_FRAME.groupby("fund", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "lifetime_value"})
        .merge(latest_fund_assets, on="fund", how="left")
        .merge(fund_counts, on="fund", how="left")
    )
    FUND_OPTION_FRAME = FUND_OPTION_FRAME[
        (FUND_OPTION_FRAME["fund"] != "")
        & (FUND_OPTION_FRAME["latest_value"].fillna(0) >= 10_000_000)
        & (FUND_OPTION_FRAME["date_count"].fillna(0) >= 6)
    ].sort_values(["latest_value", "fund"], ascending=[False, True])
else:
    latest_fund_date = pd.Timestamp(GLOBAL_MAX_DATE)
FUND_DROPDOWN_OPTIONS = [
    {
        "label": f"{row['fund']} ({format_latest_exposure_label(row['latest_value'])} latest)",
        "value": row["fund"],
    }
    for _, row in FUND_OPTION_FRAME.iterrows()
]
DEFAULT_FUND_SELECTION = FUND_DROPDOWN_OPTIONS[0]["value"] if FUND_DROPDOWN_OPTIONS else None


app = Dash(
    __name__,
    title=APP_TITLE,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": APP_DESCRIPTION},
    ],
)
server = app.server


def dataset_meta(dataset: str) -> pd.Series:
    return META.loc[META["dataset"] == dataset].iloc[0]


def filter_frame(dataset: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = TREND_DATA[TREND_DATA["dataset"] == dataset].copy()
    if start_date:
        frame = frame[frame["as_of_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["as_of_date"] <= pd.Timestamp(end_date)]
    return frame.sort_values("as_of_date")


def format_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if metric == "share":
        return f"{value:.1%}"
    return f"{value / 1e9:,.1f}bn"


def format_bn_label(value: float) -> str:
    if pd.isna(value) or value <= 0:
        return "0.0bn"
    if value < 50_000_000:
        return "<0.1bn"
    return f"{value / 1e9:,.1f}bn"


def grouping_labels(grouping: str) -> tuple[str, str]:
    if grouping == "type":
        return "Type", "Types"
    return "Category", "Categories"


def safe_triggered_id() -> str | None:
    try:
        return ctx.triggered_id
    except Exception:
        return None


def dataset_color_map(groups: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for index, group in enumerate(groups):
        if group in CATEGORY_COLORS:
            color_map[group] = CATEGORY_COLORS[group]
        elif group in TYPE_COLORS:
            color_map[group] = TYPE_COLORS[group]
        else:
            color_map[group] = PALETTE[index % len(PALETTE)]
    return color_map


def sanitize_selection(values: list[str] | None, valid_values: list[str]) -> list[str]:
    valid_set = set(valid_values)
    return [value for value in (values or []) if value in valid_set]


def dropdown_options_from_values(values: list[str]) -> list[dict[str, str]]:
    return [{"label": value, "value": value} for value in values]


def wrap_legend_label(text: str, width: int = 24) -> str:
    parts = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return "<br>".join(parts) if parts else text


def trace_hover_template(metric: str) -> str:
    if metric == "value":
        return "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f}bn<extra></extra>"
    return "%{x|%Y-%m-%d}<br>%{meta}: %{y:.1%}<extra></extra>"


def apply_historical_chart_style(
    fig: go.Figure,
    *,
    y_title: str,
    y_tickformat: str | None = None,
    height: int = FIXED_TREND_HEIGHT,
    legend_title: str | None = "Group",
    compact: bool = False,
    legend_orientation: str = "v",
) -> go.Figure:
    right_margin = 220 if legend_orientation == "v" else 24
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        margin={"l": 34, "r": right_margin, "t": 68, "b": 56},
        font={"family": "Arial", "size": 13, "color": "#16304A"},
        height=COMPACT_TREND_HEIGHT if compact else height,
        legend={
            "orientation": legend_orientation,
            "y": 1 if legend_orientation == "v" else 1.12,
            "x": 1.02 if legend_orientation == "v" else 0,
            "xanchor": "left" if legend_orientation == "v" else "left",
            "yanchor": "top" if legend_orientation == "v" else "bottom",
            "title": {"text": legend_title} if legend_title else {"text": ""},
            "font": {"size": 12},
        },
    )
    fig.update_xaxes(
        title="As-of date (month-end)",
        type="date",
        automargin=True,
        title_standoff=16,
        showgrid=True,
        gridcolor="rgba(22,48,74,0.08)",
        rangeslider=RANGE_SLIDER_STYLE,
    )
    fig.update_yaxes(
        title=y_title,
        tickformat=y_tickformat,
        automargin=True,
        title_standoff=12,
        showgrid=True,
        gridcolor="rgba(22,48,74,0.08)",
        fixedrange=True,
        autorange=True,
    )
    return fig


def apply_overview_filters(
    start_date: str | None,
    end_date: str | None,
    filters: dict[str, list[str]],
    *,
    exclude_field: str | None = None,
) -> pd.DataFrame:
    frame = OVERVIEW_HOLDINGS.copy()
    if start_date:
        frame = frame[frame["as_of_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["as_of_date"] <= pd.Timestamp(end_date)]
    for field, values in filters.items():
        if field == exclude_field or not values:
            continue
        valid_in_frame = sorted(value for value in frame[field].dropna().astype(str).unique().tolist() if value)
        selected = [value for value in values if value in valid_in_frame]
        if selected and set(selected) != set(valid_in_frame):
            frame = frame[frame[field].isin(selected)]
    return frame


def overview_filter_state(
    start_date: str | None,
    end_date: str | None,
    current_filters: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[str]], pd.DataFrame]:
    options_by_field: dict[str, list[dict[str, str]]] = {}
    sanitized_filters: dict[str, list[str]] = {}
    for field, _, _ in OVERVIEW_FILTER_SPECS:
        frame = apply_overview_filters(start_date, end_date, current_filters, exclude_field=field)
        valid_values = sorted(value for value in frame[field].dropna().astype(str).unique().tolist() if value)
        options_by_field[field] = dropdown_options_from_values(valid_values)
        selected = sanitize_selection(current_filters.get(field), valid_values)
        sanitized_filters[field] = selected if selected else valid_values
    filtered = apply_overview_filters(start_date, end_date, sanitized_filters)
    return options_by_field, sanitized_filters, filtered


def build_overview_frame(dataset: str, start_date: str | None, end_date: str | None, filters: dict[str, list[str]]) -> pd.DataFrame:
    frame = apply_overview_filters(start_date, end_date, filters)
    group_column = OVERVIEW_GROUP_COLUMNS[dataset]
    frame = frame[(frame[group_column].notna()) & (frame[group_column].astype(str) != "")]
    grouped = frame.groupby(["as_of_date", group_column], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "total_value"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["share"] = grouped["value"] / grouped["total_value"]
    grouped["group_name"] = grouped[group_column].astype(str)
    if dataset in DEFAULT_EXCLUDE:
        grouped = grouped[~grouped["group_name"].isin(DEFAULT_EXCLUDE[dataset])].copy()
    return grouped[["as_of_date", "group_name", "value", "share"]].sort_values(["as_of_date", "value"], ascending=[True, False])


def build_holdings_group_frame(
    group_column: str,
    start_date: str | None,
    end_date: str | None,
    *,
    exclude_groups: set[str] | None = None,
    unknown_label: str | None = None,
) -> pd.DataFrame:
    frame = OVERVIEW_HOLDINGS.copy()
    if start_date:
        frame = frame[frame["as_of_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["as_of_date"] <= pd.Timestamp(end_date)]
    if unknown_label:
        frame[group_column] = frame[group_column].where(frame[group_column].astype(str) != "", unknown_label)
    frame = frame[(frame[group_column].notna()) & (frame[group_column].astype(str) != "")]
    grouped = frame.groupby(["as_of_date", group_column], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "total_value"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["share"] = grouped["value"] / grouped["total_value"]
    grouped["group_name"] = grouped[group_column].astype(str)
    if exclude_groups:
        grouped = grouped[~grouped["group_name"].isin(exclude_groups)].copy()
    return grouped[["as_of_date", "group_name", "value", "share"]].sort_values(["as_of_date", "value"], ascending=[True, False])


def selected_groups_from_latest(frame: pd.DataFrame, preset: str, custom_groups: list[str] | None) -> list[str]:
    if frame.empty:
        return []
    latest_date = frame["as_of_date"].max()
    latest = frame[frame["as_of_date"] == latest_date].sort_values("value", ascending=False)
    if preset == "all":
        return (
            frame.groupby("group_name", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)["group_name"]
            .tolist()
        )
    if preset == "top_10":
        return latest["group_name"].head(10).tolist()
    if preset == "custom":
        selected = sanitize_selection(custom_groups, latest["group_name"].tolist())
        if selected:
            return selected
    return latest["group_name"].head(5).tolist()


def selected_groups_for_frame(dataset: str, frame: pd.DataFrame, preset: str, custom_groups: list[str] | None) -> list[str]:
    latest_date = frame["as_of_date"].max()
    latest = frame[frame["as_of_date"] == latest_date].sort_values("value", ascending=False)
    exclude = DEFAULT_EXCLUDE.get(dataset, set())
    if exclude:
        filtered = latest[~latest["group_name"].isin(exclude)]
        if not filtered.empty:
            latest = filtered
    if preset == "all":
        return (
            frame.groupby("group_name", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)["group_name"]
            .tolist()
        )
    if preset == "top_10":
        return latest["group_name"].head(10).tolist()
    if preset == "custom" and custom_groups:
        valid_groups = set(latest["group_name"].tolist())
        selected = [group for group in custom_groups if group in valid_groups]
        if selected:
            return selected
    return latest["group_name"].head(5).tolist()


def maturity_subset_options(subset_by: str, start_date: str, end_date: str, current_values: list[str] | None) -> tuple[list[dict[str, str]], list[str], str]:
    frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    if subset_by == "market":
        return [], [], "All holdings in the selected date range."

    if subset_by == "type":
        working = frame[frame["type"] != ""].groupby("type", as_index=False)["value"].sum().sort_values("value", ascending=False)
        valid_values = working["type"].tolist()
        options = [{"label": value, "value": value} for value in valid_values]
        selected = sanitize_selection(current_values, valid_values) or valid_values[: min(5, len(valid_values))]
        return options, selected, "Subset is applied on HoldingList.Type before recomputing bucket shares and weighted-average maturity."

    if subset_by == "category":
        working = frame[frame["category"] != ""].groupby("category", as_index=False)["value"].sum().sort_values("value", ascending=False)
        valid_values = working["category"].tolist()
        options = [{"label": issuer_group_label('category', value), "value": value} for value in valid_values]
        selected = sanitize_selection(current_values, valid_values) or valid_values[: min(5, len(valid_values))]
        return options, selected, "Subset is applied on HoldingList.Category before recomputing bucket shares and weighted-average maturity."

    if subset_by == "fund":
        working = frame[frame["fund"] != ""].groupby("fund", as_index=False)["value"].sum().sort_values("value", ascending=False)
        valid_values = working["fund"].tolist()
        options = [{"label": wrap_legend_label(value, width=34).replace("<br>", "\n"), "value": value} for value in valid_values]
        selected = sanitize_selection(current_values, valid_values) or valid_values[:1]
        return options, selected, "Subset is applied to one or more funds before recomputing bucket shares and weighted-average maturity."

    if subset_by == "issuer":
        working = (
            frame[frame["issuer_norm"] != ""]
            .groupby(["issuer_norm", "issuer_display"], as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)
            .drop_duplicates("issuer_norm")
        )
        valid_values = working["issuer_norm"].tolist()
        options = [{"label": wrap_legend_label(row["issuer_display"], width=38).replace("<br>", "\n"), "value": row["issuer_norm"]} for _, row in working.iterrows()]
        selected = sanitize_selection(current_values, valid_values) or valid_values[:1]
        return options, selected, "Subset is applied to one or more issuers before recomputing bucket shares and weighted-average maturity."

    return [], [], "Unsupported subset."


def filter_maturity_subset(frame: pd.DataFrame, subset_by: str, selected_values: list[str] | None) -> pd.DataFrame:
    if subset_by == "market" or not selected_values:
        return frame.copy()
    if subset_by == "type":
        return frame[frame["type"].isin(selected_values)].copy()
    if subset_by == "category":
        return frame[frame["category"].isin(selected_values)].copy()
    if subset_by == "fund":
        return frame[frame["fund"].isin(selected_values)].copy()
    if subset_by == "issuer":
        return frame[frame["issuer_norm"].isin(selected_values)].copy()
    return frame.copy()


def graph_component(graph_id: str, height_px: int = 620, *, scroll_zoom: bool = True) -> dcc.Graph:
    config = dict(GRAPH_CONFIG)
    config["scrollZoom"] = scroll_zoom
    return dcc.Graph(id=graph_id, className="trend-graph", config=config, style={"height": f"{height_px}px"})


def trace_values(trace, attr: str) -> list:
    values = getattr(trace, attr, None)
    if values is None:
        return []
    if isinstance(values, dict) and {"dtype", "bdata"} <= set(values.keys()):
        try:
            decoded = base64.b64decode(values["bdata"])
            return np.frombuffer(decoded, dtype=np.dtype(values["dtype"])).tolist()
        except Exception:
            return []
    return list(values)


def sync_x_zoom_autorange(figure: dict | None, relayout_data: dict | None):
    if not figure or not relayout_data:
        return no_update
    x_keys = [key for key in relayout_data if key.startswith("xaxis.")]
    if not x_keys:
        return no_update

    updated = go.Figure(figure, skip_invalid=True)
    x_range = None
    if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
        x_range = [relayout_data["xaxis.range[0]"], relayout_data["xaxis.range[1]"]]
        updated.update_xaxes(range=x_range)
    elif "xaxis.range" in relayout_data:
        x_range = relayout_data["xaxis.range"]
        updated.update_xaxes(range=x_range)
    elif relayout_data.get("xaxis.autorange"):
        updated.update_xaxes(autorange=True)

    y_range = None
    if x_range:
        x_start = pd.to_datetime(x_range[0], errors="coerce")
        x_end = pd.to_datetime(x_range[1], errors="coerce")
        if pd.notna(x_start) and pd.notna(x_end):
            stacked_totals: dict[pd.Timestamp, float] = {}
            visible_values: list[float] = []
            has_stacked_area = False
            for trace in updated.data:
                trace_x = pd.to_datetime(trace_values(trace, "x"), errors="coerce")
                trace_y = pd.to_numeric(trace_values(trace, "y"), errors="coerce")
                if len(trace_x) == 0 or len(trace_y) == 0:
                    continue
                point_count = min(len(trace_x), len(trace_y))
                trace_x = trace_x[:point_count]
                trace_y = trace_y[:point_count]
                mask = (trace_x >= x_start) & (trace_x <= x_end) & pd.notna(trace_y)
                if not mask.any():
                    continue
                if getattr(trace, "stackgroup", None):
                    has_stacked_area = True
                    for x_value, y_value in zip(trace_x[mask], trace_y[mask]):
                        if pd.notna(x_value) and pd.notna(y_value):
                            stacked_totals[pd.Timestamp(x_value)] = stacked_totals.get(pd.Timestamp(x_value), 0.0) + float(y_value)
                else:
                    visible_values.extend(float(value) for value in trace_y[mask] if pd.notna(value))
            if has_stacked_area and stacked_totals:
                visible_values = list(stacked_totals.values())
            if visible_values:
                y_min = min(visible_values)
                y_max = max(visible_values)
                pad = (y_max - y_min) * 0.08 if y_max > y_min else max(abs(y_max) * 0.1, 1.0)
                y_floor = min(0.0, y_min - pad) if y_min >= 0 else y_min - pad
                y_range = [y_floor, y_max + pad]
    else:
        visible_values = []
        for trace in updated.data:
            trace_y = pd.to_numeric(trace_values(trace, "y"), errors="coerce")
            visible_values.extend(float(value) for value in trace_y if pd.notna(value))
        if visible_values:
            y_min = min(visible_values)
            y_max = max(visible_values)
            pad = (y_max - y_min) * 0.08 if y_max > y_min else max(abs(y_max) * 0.1, 1.0)
            y_floor = min(0.0, y_min - pad) if y_min >= 0 else y_min - pad
            y_range = [y_floor, y_max + pad]

    for axis_name in updated.layout:
        if str(axis_name).startswith("yaxis"):
            updated.layout[axis_name].fixedrange = True
            if y_range:
                updated.layout[axis_name].range = y_range
                updated.layout[axis_name].autorange = False
            else:
                updated.layout[axis_name].autorange = True

    return updated


def complete_group_frame(frame: pd.DataFrame, all_dates: pd.Index, groups: list[str], value_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["as_of_date", "group_name", value_col])
    pivot = frame.pivot_table(index="as_of_date", columns="group_name", values=value_col, aggfunc="sum")
    pivot = pivot.reindex(pd.Index(all_dates, name="as_of_date"))
    if groups:
        pivot = pivot.reindex(columns=groups, fill_value=0.0)
    pivot = pivot.fillna(0.0)
    return pivot.stack().reset_index(name=value_col)


def resolve_snapshot_date(preferred_date: str | pd.Timestamp | None, available_dates: pd.Series | pd.Index) -> pd.Timestamp:
    preferred = pd.to_datetime(preferred_date, errors="coerce")
    dates = pd.Index(pd.to_datetime(pd.Series(available_dates), errors="coerce").dropna().unique()).sort_values()
    if dates.empty:
        return pd.NaT
    if pd.isna(preferred):
        return pd.Timestamp(dates[-1])
    eligible = dates[dates <= preferred]
    if len(eligible) > 0:
        return pd.Timestamp(eligible[-1])
    return pd.Timestamp(dates[0])


def build_breakdown_timeline_figure(frame: pd.DataFrame, selected_date: pd.Timestamp) -> go.Figure:
    totals = (
        frame.groupby("as_of_date", as_index=False)["value"]
        .sum()
        .sort_values("as_of_date")
        .assign(value_bn=lambda df: df["value"] / 1e9)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=totals["as_of_date"],
            y=totals["value_bn"],
            mode="lines+markers",
            name="Total Holdings",
            line={"color": "#16304A", "width": 3},
            marker={"size": 7, "color": "#94A3B8"},
            fill="tozeroy",
            fillcolor="rgba(22,48,74,0.08)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}bn<extra></extra>",
        )
    )
    if pd.notna(selected_date):
        selected_row = totals[totals["as_of_date"] == selected_date]
        if not selected_row.empty:
            fig.add_trace(
                go.Scatter(
                    x=selected_row["as_of_date"],
                    y=selected_row["value_bn"],
                    mode="markers",
                    name="Selected month",
                    marker={"size": 14, "color": "#D97706", "line": {"color": "#FFF7ED", "width": 2}},
                    hovertemplate="Selected<br>%{x|%Y-%m-%d}<br>%{y:,.1f}bn<extra></extra>",
                )
            )
    fig.update_layout(title="Total Holdings | Click a month to update the snapshot below", title_x=0.03, showlegend=False, transition={"duration": 200})
    fig = apply_historical_chart_style(fig, y_title="Total Holdings (USD bn)", height=220, compact=False, legend_title=None)
    fig.update_xaxes(rangeslider={"visible": False}, fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def add_selected_date_vline(fig: go.Figure, selected_date: pd.Timestamp | None) -> go.Figure:
    if pd.isna(selected_date):
        return fig
    fig.add_vline(
        x=pd.Timestamp(selected_date),
        line_width=2,
        line_dash="dash",
        line_color="#D97706",
        opacity=0.95,
    )
    return fig


def build_snapshot_pie_figure(
    frame: pd.DataFrame,
    *,
    dataset: str,
    selected_date: pd.Timestamp,
    groups: list[str],
) -> go.Figure:
    snapshot = frame[frame["as_of_date"] == selected_date].copy()
    if groups:
        snapshot = snapshot[snapshot["group_name"].isin(groups)]
    snapshot = snapshot.sort_values("value", ascending=False)
    if snapshot.empty:
        empty = go.Figure()
        empty.update_layout(title=f"{OVERVIEW_DATASET_LABELS[dataset]} snapshot: no data")
        empty.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white", margin={"l": 18, "r": 220, "t": 64, "b": 18})
        return empty
    total = snapshot["value"].sum()
    snapshot["share"] = snapshot["value"] / total if total else 0.0
    color_basis = groups if groups else snapshot["group_name"].tolist()
    color_map = dataset_color_map(color_basis)
    snapshot["display_label"] = snapshot["group_name"].map(lambda label: wrap_legend_label(str(label)))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=snapshot["display_label"],
                values=snapshot["value"],
                sort=False,
                hole=0.0,
                marker={"colors": [color_map[group] for group in snapshot["group_name"]], "line": {"color": "white", "width": 1}},
                domain={"x": [0.08, 0.74], "y": [0.04, 0.96]},
                customdata=snapshot["group_name"],
                text=[f"{value:.1%}" if value >= 0.01 else "" for value in snapshot["share"]],
                textinfo="text",
                hovertemplate="%{customdata}<br>%{value:,.0f}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"{OVERVIEW_DATASET_LABELS[dataset]} snapshot | {selected_date.date()}",
        height=460,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 28, "r": 180, "t": 64, "b": 18},
        font={"family": "Arial", "size": 13, "color": "#16304A"},
        title_x=0.04,
        legend={
            "orientation": "v",
            "x": 0.86,
            "y": 1,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(244,247,251,0.92)",
            "bordercolor": "rgba(22,48,74,0.08)",
            "borderwidth": 1,
            "font": {"size": 12},
        },
        transition={"duration": 250},
    )
    return fig


def build_market_figure(frame: pd.DataFrame, groups: list[str], metric: str, chart_mode: str, title: str) -> go.Figure:
    return build_market_figure_with_options(frame, groups, metric, chart_mode, title, normalize_share=False)


def build_market_figure_with_options(
    frame: pd.DataFrame,
    groups: list[str],
    metric: str,
    chart_mode: str,
    title: str,
    *,
    normalize_share: bool = False,
) -> go.Figure:
    if frame.empty or not groups:
        fig = go.Figure()
        fig.update_layout(title=title)
        return apply_historical_chart_style(
            fig,
            y_title="USD bn" if metric == "value" else "Share of total",
            y_tickformat=",.1f" if metric == "value" else ".0%",
        )
    plot_frame = frame[frame["group_name"].isin(groups)].copy()
    complete_value_col = "share" if metric == "share" and not normalize_share else "value"
    plot_frame = complete_group_frame(plot_frame, pd.Index(sorted(frame["as_of_date"].unique())), groups, complete_value_col)
    if metric == "value":
        plot_frame["metric_value"] = plot_frame["value"] / 1e9
    else:
        if normalize_share:
            totals = plot_frame.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "selected_total"})
            plot_frame = plot_frame.merge(totals, on="as_of_date", how="left")
            plot_frame["metric_value"] = np.where(plot_frame["selected_total"] > 0, plot_frame["value"] / plot_frame["selected_total"], 0.0)
        else:
            plot_frame["metric_value"] = plot_frame["share"]
    color_map = dataset_color_map(groups)

    if chart_mode == "area":
        fig = px.area(plot_frame, x="as_of_date", y="metric_value", color="group_name", color_discrete_map=color_map)
    else:
        fig = px.line(plot_frame, x="as_of_date", y="metric_value", color="group_name", color_discrete_map=color_map, markers=True)

    original_names = groups[:]
    wrapped_name_map = {name: wrap_legend_label(name, width=20) for name in original_names}
    for trace in fig.data:
        original_name = str(trace.name)
        trace.name = wrapped_name_map.get(original_name, wrap_legend_label(original_name, width=20))
        trace.meta = original_name
        trace.hovertemplate = trace_hover_template(metric)

    fig.update_layout(title=title)
    return apply_historical_chart_style(
        fig,
        y_title="USD bn" if metric == "value" else "Share of total",
        y_tickformat=",.1f" if metric == "value" else ".0%",
    )


def maturity_bucket_labels(days: pd.Series) -> pd.Series:
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
    return pd.cut(days, bins=bins, labels=labels)


def build_latest_table(frame: pd.DataFrame, groups: list[str], metric: str) -> list[dict[str, str]]:
    latest_date = frame["as_of_date"].max()
    latest = frame[(frame["as_of_date"] == latest_date) & (frame["group_name"].isin(groups))].sort_values("value", ascending=False).copy()
    latest["display_value"] = latest[metric if metric == "share" else "value"].map(lambda value: format_value(metric, value))
    latest["as_of_date"] = latest["as_of_date"].dt.strftime("%Y-%m-%d")
    return latest[["group_name", "display_value", "as_of_date"]].rename(
        columns={"group_name": "Group", "display_value": "Latest", "as_of_date": "As of"}
    ).to_dict("records")


def build_snapshot_table(
    frame: pd.DataFrame,
    *,
    selected_date: pd.Timestamp,
    groups: list[str],
    label_column: str,
    label_header: str,
) -> list[dict[str, str]]:
    snapshot = frame[frame["as_of_date"] == selected_date].copy()
    if groups:
        snapshot = snapshot[snapshot["group_name"].isin(groups)]
    snapshot = snapshot.sort_values("value", ascending=False)
    if snapshot.empty:
        return []
    snapshot["Value"] = snapshot["value"].map(lambda value: format_value("value", value))
    snapshot["Share"] = snapshot["share"].map(lambda value: format_value("share", value))
    snapshot["As of"] = selected_date.strftime("%Y-%m-%d")
    snapshot[label_header] = snapshot["group_name"]
    return snapshot[[label_header, "Value", "Share", "As of"]].to_dict("records")


def summary_cards(items: list[tuple[str, str]]) -> list[html.Div]:
    return [
        html.Div(
            className="summary-card",
            children=[
                html.Div(label, className="card-label"),
                html.Div(value, className="card-value" if len(str(value)) < 24 else "card-value small"),
            ],
        )
        for label, value in items
    ]


def page_shell(title: str, subtitle: str, controls: html.Div, content: html.Div) -> html.Div:
    return html.Div(
        className="page-shell",
        children=[
            html.Div(className="page-header", children=[html.H2(title, className="page-title"), html.P(subtitle, className="page-subtitle")]),
            html.Div(className="content-grid", children=[controls, content]),
        ],
    )


def render_nav(pathname: str) -> html.Div:
    return html.Div(
        className="nav-bar",
        children=[
            dcc.Link(label, href=href, className=f"nav-link{' active' if pathname == href else ''}")
            for href, label in NAV_ITEMS
        ],
    )


def overview_layout() -> html.Div:
    controls = html.Div(
        className="sidebar",
        children=[
            html.Div("Overview controls", className="panel-title"),
            html.Label("Metric", className="control-label"),
            dcc.RadioItems(id="mo-metric", options=METRIC_OPTIONS, value="value", className="radio-grid"),
            html.Label("Chart style", className="control-label"),
            dcc.RadioItems(id="mo-chart", options=CHART_OPTIONS, value="line", className="radio-grid"),
            html.Label("Groups per chart", className="control-label"),
            dcc.RadioItems(id="mo-preset", options=OVERVIEW_PRESET_OPTIONS, value="top_5", className="radio-stack"),
            html.Label("Date range", className="control-label"),
            dcc.DatePickerRange(
                id="mo-date-range",
                min_date_allowed=GLOBAL_MIN_DATE,
                max_date_allowed=GLOBAL_MAX_DATE,
                start_date=GLOBAL_MIN_DATE,
                end_date=GLOBAL_MAX_DATE,
                display_format="YYYY-MM-DD",
                number_of_months_shown=2,
            ),
            html.Div(
                className="dataset-description",
                children="Lightweight overview built from precomputed trend data for Top Issuers and Top Funds. Use Market Breakdown for workbook-style HoldingList filters.",
            ),
        ],
    )
    return page_shell(
        "Market Overview",
        "Top issuer and top fund trend views using the lighter precomputed trend layer, so the page loads and updates faster.",
        controls,
        html.Div(
            className="main-panel",
            children=[
                html.Div(id="mo-cards", className="summary-cards"),
                graph_component("mo-issuers-graph", 620),
                graph_component("mo-funds-graph", 620),
            ],
        ),
    )


def breakdown_layout() -> html.Div:
    filter_controls = []
    for field, label, component_id in OVERVIEW_FILTER_SPECS:
        filter_controls.extend(
            [
                html.Label(label, className="control-label"),
                dcc.Dropdown(id=component_id, options=[], value=[], multi=True, placeholder=f"Filter by {label.lower()}"),
            ]
        )
    controls = html.Div(
        className="sidebar",
        children=[
            html.Div("Breakdown controls", className="panel-title"),
            html.Label("Metric", className="control-label"),
            dcc.RadioItems(id="ov-metric", options=METRIC_OPTIONS, value="value", className="radio-grid"),
            html.Label("Chart style", className="control-label"),
            dcc.RadioItems(id="ov-chart", options=CHART_OPTIONS, value="line", className="radio-grid"),
            html.Label("Groups per chart", className="control-label"),
            dcc.RadioItems(id="ov-preset", options=BREAKDOWN_PRESET_OPTIONS, value="all", className="radio-stack"),
            html.Label("Date range", className="control-label"),
            dcc.DatePickerRange(
                id="ov-date-range",
                min_date_allowed=GLOBAL_MIN_DATE,
                max_date_allowed=GLOBAL_MAX_DATE,
                start_date=GLOBAL_MIN_DATE,
                end_date=GLOBAL_MAX_DATE,
                display_format="YYYY-MM-DD",
                number_of_months_shown=2,
            ),
            *filter_controls,
            html.Div(id="ov-description", className="dataset-description"),
        ],
    )
    return page_shell(
        "Market Breakdown",
        "Portfolio composition and country breakout trends, split out into a separate page but driven by the same filter stack.",
        controls,
        html.Div(
            className="main-panel",
            children=[
                dcc.Store(id="ov-selection-store"),
                html.Div(id="ov-cards", className="summary-cards"),
                dcc.Loading(graph_component("ov-primary-graph", 620), type="circle", color="#0F766E"),
                dcc.Loading(graph_component("ov-secondary-graph", 620), type="circle", color="#1D4ED8"),
                dcc.Loading(graph_component("ov-timeline-graph", 220, scroll_zoom=False), type="circle", color="#16304A"),
                html.Div(
                    className="graph-grid",
                    children=[
                        dcc.Loading(graph_component("ov-composition-pie", 460, scroll_zoom=False), type="circle", color="#0F766E"),
                        dcc.Loading(graph_component("ov-country-pie", 460, scroll_zoom=False), type="circle", color="#1D4ED8"),
                    ],
                ),
                dcc.Loading(graph_component("ov-maturity-graph", 460, scroll_zoom=False), type="circle", color="#D97706"),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="ov-funds-title", children="Top 10 Funds", className="panel-title"),
                                dash_table.DataTable(
                                    id="ov-funds-table",
                                    columns=[{"name": "Fund", "id": "Fund"}, {"name": "Value", "id": "Value"}, {"name": "Share", "id": "Share"}],
                                    data=[],
                                    page_size=10,
                                    style_as_list_view=True,
                                    style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"},
                                    style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="ov-issuers-title", children="Top 50 Issuers", className="panel-title"),
                                dash_table.DataTable(
                                    id="ov-issuers-table",
                                    columns=[{"name": "Issuer", "id": "Issuer"}, {"name": "Value", "id": "Value"}, {"name": "Share", "id": "Share"}],
                                    data=[],
                                    page_size=50,
                                    style_as_list_view=True,
                                    style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"},
                                    style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def country_layout() -> html.Div:
    return page_shell(
        "Country",
        "Country and region exposures rebuilt directly from HoldingList, with a separate time selector and snapshot pies.",
        html.Div(
            className="sidebar",
            children=[
                html.Div("Country controls", className="panel-title"),
                html.Label("Metric", className="control-label"),
                dcc.RadioItems(id="ct-metric", options=METRIC_OPTIONS, value="share", className="radio-grid"),
                html.Label("Chart style", className="control-label"),
                dcc.RadioItems(id="ct-chart", options=CHART_OPTIONS, value="line", className="radio-grid"),
                html.Label("Group preset", className="control-label"),
                dcc.RadioItems(id="ct-preset", options=PRESET_OPTIONS, value="custom", className="radio-stack"),
                html.Label("Date range", className="control-label"),
                dcc.DatePickerRange(id="ct-date-range", min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE, start_date=GLOBAL_MIN_DATE, end_date=GLOBAL_MAX_DATE, display_format="YYYY-MM-DD", number_of_months_shown=2),
                html.Label("Countries", className="control-label"),
                dcc.Dropdown(id="ct-groups", options=[], value=[], multi=True, placeholder="Select countries"),
                html.Div(id="ct-description", className="dataset-description"),
            ],
        ),
        html.Div(
            className="main-panel",
            children=[
                dcc.Store(id="ct-selection-store"),
                html.Div(id="ct-cards", className="summary-cards"),
                graph_component("ct-graph", 620),
                graph_component("ct-region-graph", 520),
                graph_component("ct-timeline-graph", 220, scroll_zoom=False),
                html.Div(
                    className="graph-grid",
                    children=[
                        graph_component("ct-country-pie", 460, scroll_zoom=False),
                        graph_component("ct-region-pie", 460, scroll_zoom=False),
                    ],
                ),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="ct-country-title", children="Country breakout snapshot", className="panel-title"),
                                dash_table.DataTable(
                                    id="ct-country-table",
                                    columns=[{"name": "Country", "id": "Country"}, {"name": "Value", "id": "Value"}, {"name": "Share", "id": "Share"}, {"name": "As of", "id": "As of"}],
                                    data=[],
                                    page_action="none",
                                    style_as_list_view=True,
                                    style_table={"overflowY": "visible", "height": "auto"},
                                    style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"},
                                    style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"},
                                ),
                            ],
                        ),
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="ct-region-title", children="Region breakout snapshot", className="panel-title"),
                                dash_table.DataTable(
                                    id="ct-region-table",
                                    columns=[{"name": "Region", "id": "Region"}, {"name": "Value", "id": "Value"}, {"name": "Share", "id": "Share"}, {"name": "As of", "id": "As of"}],
                                    data=[],
                                    page_size=10,
                                    style_as_list_view=True,
                                    style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"},
                                    style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def maturity_layout() -> html.Div:
    return page_shell(
        "Maturity",
        "Market maturity buckets and weighted-average maturity lines, both rebuilt directly from HoldingList dates.",
        html.Div(
            className="sidebar",
            children=[
                html.Div("Maturity controls", className="panel-title"),
                html.Label("Metric", className="control-label"),
                dcc.RadioItems(id="mt-metric", options=METRIC_OPTIONS, value="share", className="radio-grid"),
                html.Label("Chart style", className="control-label"),
                dcc.RadioItems(id="mt-chart", options=CHART_OPTIONS, value="area", className="radio-grid"),
                html.Label("Subset by", className="control-label"),
                dcc.RadioItems(
                    id="mt-subset-by",
                    options=[
                        {"label": "All holdings", "value": "market"},
                        {"label": "Type", "value": "type"},
                        {"label": "Category", "value": "category"},
                        {"label": "Issuer", "value": "issuer"},
                        {"label": "Fund", "value": "fund"},
                    ],
                    value="market",
                    className="radio-stack",
                ),
                html.Label("Date range", className="control-label"),
                dcc.DatePickerRange(id="mt-date-range", min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE, start_date=GLOBAL_MIN_DATE, end_date=GLOBAL_MAX_DATE, display_format="YYYY-MM-DD", number_of_months_shown=2),
                html.Label("Subset values", className="control-label"),
                dcc.Dropdown(id="mt-subset-values", options=[], value=[], multi=True, placeholder="Select subset values", className="compact-multi-dropdown"),
                html.Label("Buckets", className="control-label"),
                dcc.Dropdown(id="mt-groups", options=[{"label": group, "value": group} for group in MATURITY_GROUPS], value=MATURITY_GROUPS, multi=True),
                html.Div(id="mt-description", className="dataset-description", children="The bucket chart uses the same FundCharts tenor buckets, but every point is recomputed from HoldingList maturity dates. The second chart shows principal-weighted WA maturity and WA final maturity for the selected subset."),
            ],
        ),
        html.Div(
            className="main-panel",
            children=[
                html.Div(id="mt-cards", className="summary-cards"),
                graph_component("mt-buckets-graph", 620),
                graph_component("mt-lines-graph", 460),
                html.Div(
                    className="table-panel",
                    children=[
                        html.Div("Latest maturity snapshot", className="panel-title"),
                        dash_table.DataTable(id="mt-table", columns=[{"name": "Group", "id": "Group"}, {"name": "Latest", "id": "Latest"}, {"name": "As of", "id": "As of"}], data=[], style_as_list_view=True, style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"}, style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"}),
                    ],
                ),
            ],
        ),
    )


def issuer_layout() -> html.Div:
    return page_shell(
        "Issuer Deep Dive",
        "Search an issuer, snap to an event window, and inspect exposure, product mix, and maturity through time.",
        html.Div(
            className="sidebar",
            children=[
                html.Div("Issuer controls", className="panel-title"),
                html.Label("Issuer", className="control-label"),
                dcc.Dropdown(id="is-issuer", options=ISSUER_DROPDOWN_OPTIONS, value="preset::credit_suisse", clearable=False),
                html.Label("Event window", className="control-label"),
                dcc.Dropdown(id="is-window", options=EVENT_PRESET_OPTIONS, value="cs_stress", clearable=False),
                html.Label("Date range", className="control-label"),
                dcc.DatePickerRange(id="is-date-range", min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE, start_date=DEFAULT_ISSUER_START, end_date=DEFAULT_ISSUER_END, display_format="YYYY-MM-DD", number_of_months_shown=2),
                html.Label("Type preset", className="control-label"),
                dcc.RadioItems(id="is-type-preset", options=ISSUER_PRESET_OPTIONS, value="all", className="radio-stack"),
                html.Label("Types", className="control-label"),
                dcc.Dropdown(id="is-types", options=[], value=[], multi=True, placeholder="Select types"),
                html.Label("Category preset", className="control-label"),
                dcc.RadioItems(id="is-category-preset", options=ISSUER_PRESET_OPTIONS, value="all", className="radio-stack"),
                html.Label("Categories", className="control-label"),
                dcc.Dropdown(id="is-categories", options=[], value=[], multi=True, placeholder="Select categories"),
                html.Div(id="is-description", className="dataset-description"),
            ],
        ),
        html.Div(
            className="main-panel",
            children=[
                dcc.Store(id="is-selection-store"),
                html.Div(id="is-cards", className="summary-cards"),
                graph_component("is-exposure-graph", 620),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(children=[graph_component("is-type-value", 460), graph_component("is-type-share", 460)]),
                        html.Div(children=[graph_component("is-category-value", 460), graph_component("is-category-share", 460)]),
                    ],
                ),
                graph_component("is-maturity-graph", 460),
                graph_component("is-maturity-distribution", 420, scroll_zoom=False),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="is-type-title", children="Type snapshot", className="panel-title"),
                                dash_table.DataTable(id="is-type-table", columns=[{"name": "Type", "id": "Type"}, {"name": "Value", "id": "Value"}, {"name": "Issuer share", "id": "Issuer share"}, {"name": "As of", "id": "As of"}], data=[], style_as_list_view=True, style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"}, style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"}),
                            ],
                        ),
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="is-category-title", children="Category snapshot", className="panel-title"),
                                dash_table.DataTable(id="is-category-table", columns=[{"name": "Category", "id": "Category"}, {"name": "Value", "id": "Value"}, {"name": "Issuer share", "id": "Issuer share"}, {"name": "As of", "id": "As of"}], data=[], style_as_list_view=True, style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"}, style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def fund_layout() -> html.Div:
    return page_shell(
        "Fund Deep Dive",
        "Select a fund, set a window, and inspect assets, product mix, and maturity through time.",
        html.Div(
            className="sidebar",
            children=[
                html.Div("Fund controls", className="panel-title"),
                html.Label("Fund", className="control-label"),
                dcc.Dropdown(id="fs-fund", options=FUND_DROPDOWN_OPTIONS, value=DEFAULT_FUND_SELECTION, clearable=False),
                html.Label("Event window", className="control-label"),
                dcc.Dropdown(id="fs-window", options=EVENT_PRESET_OPTIONS, value="last_24m", clearable=False),
                html.Label("Date range", className="control-label"),
                dcc.DatePickerRange(id="fs-date-range", min_date_allowed=GLOBAL_MIN_DATE, max_date_allowed=GLOBAL_MAX_DATE, start_date=DEFAULT_ISSUER_START, end_date=DEFAULT_ISSUER_END, display_format="YYYY-MM-DD", number_of_months_shown=2),
                html.Label("Type preset", className="control-label"),
                dcc.RadioItems(id="fs-type-preset", options=ISSUER_PRESET_OPTIONS, value="all", className="radio-stack"),
                html.Label("Types", className="control-label"),
                dcc.Dropdown(id="fs-types", options=[], value=[], multi=True, placeholder="Select types"),
                html.Label("Category preset", className="control-label"),
                dcc.RadioItems(id="fs-category-preset", options=ISSUER_PRESET_OPTIONS, value="all", className="radio-stack"),
                html.Label("Categories", className="control-label"),
                dcc.Dropdown(id="fs-categories", options=[], value=[], multi=True, placeholder="Select categories"),
                html.Div(id="fs-description", className="dataset-description"),
            ],
        ),
        html.Div(
            className="main-panel",
            children=[
                dcc.Store(id="fs-selection-store"),
                html.Div(id="fs-cards", className="summary-cards"),
                graph_component("fs-exposure-graph", 620),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(children=[graph_component("fs-type-value", 460), graph_component("fs-type-share", 460)]),
                        html.Div(children=[graph_component("fs-category-value", 460), graph_component("fs-category-share", 460)]),
                    ],
                ),
                graph_component("fs-maturity-graph", 460),
                graph_component("fs-maturity-distribution", 420, scroll_zoom=False),
                html.Div(
                    className="graph-grid",
                    children=[
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="fs-type-title", children="Type snapshot", className="panel-title"),
                                dash_table.DataTable(id="fs-type-table", columns=[{"name": "Type", "id": "Type"}, {"name": "Value", "id": "Value"}, {"name": "Fund share", "id": "Fund share"}, {"name": "As of", "id": "As of"}], data=[], style_as_list_view=True, style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"}, style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"}),
                            ],
                        ),
                        html.Div(
                            className="table-panel",
                            children=[
                                html.Div(id="fs-category-title", children="Category snapshot", className="panel-title"),
                                dash_table.DataTable(id="fs-category-table", columns=[{"name": "Category", "id": "Category"}, {"name": "Value", "id": "Value"}, {"name": "Fund share", "id": "Fund share"}, {"name": "As of", "id": "As of"}], data=[], style_as_list_view=True, style_header={"backgroundColor": "#F4F7FB", "fontWeight": "600", "color": "#16304A", "border": "none"}, style_cell={"padding": "10px 12px", "border": "none", "backgroundColor": "white", "fontFamily": "Arial", "fontSize": 13, "color": "#213547"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def render_page(pathname: str) -> html.Div:
    if pathname == "/market-breakdown":
        return breakdown_layout()
    if pathname == "/country":
        return country_layout()
    if pathname == "/maturity":
        return maturity_layout()
    if pathname == "/issuer":
        return issuer_layout()
    if pathname == "/fund":
        return fund_layout()
    return overview_layout()


def build_app_shell(pathname: str = "/") -> html.Div:
    return html.Div(
        className="app-shell",
        children=[
            html.Div(
                className="hero",
                children=[
                    html.Div(
                        [
                            html.Div(
                                className="brand-lockup",
                                children=[
                                    html.Img(src=app.get_asset_url("crane-mark.svg"), className="brand-logo", alt="Crane Insight logo"),
                                    html.Div(
                                        className="brand-copy",
                                        children=[
                                            html.Div("CRANE INSIGHT", className="brand-wordmark"),
                                            html.Div("Money Fund Holdings Intelligence", className="brand-tagline"),
                                        ],
                                    ),
                                ],
                            ),
                            html.H1("Money Fund Holdings Trend Explorer", className="hero-title"),
                            html.P(
                                "Interactive research workspace for turning Crane monthly hold reports into continuous trend views across issuer exposure, fund positioning, product mix, country allocation, and maturity structure.",
                                className="hero-subtitle",
                            ),
                        ]
                    ),
                    html.Div(
                        className="hero-note",
                        children=[
                            html.Div("Coverage", className="hero-note-label"),
                            html.Div("Crane monthly hold reports | 2011-03 to 2026-03", className="hero-note-value"),
                            html.Div("Modules", className="hero-note-label"),
                            html.Div("Market, Country, Maturity, Issuer, Fund deep dives", className="hero-note-value"),
                        ],
                    ),
                ],
            ),
            render_nav(pathname),
            html.Div(id="page-container", children=render_page(pathname)),
        ],
    )


app.layout = html.Div([dcc.Location(id="url"), html.Div(id="app-shell")])


def resolve_issuer_selection(selection: str) -> tuple[str, list[str]]:
    if selection in ISSUER_PRESETS:
        preset = ISSUER_PRESETS[selection]
        norms = []
        for pattern in preset["patterns"]:
            if preset.get("match_mode") == "startswith":
                mask = ISSUER_LOOKUP["issuer_norm"].str.startswith(pattern)
            else:
                mask = ISSUER_LOOKUP["issuer_norm"].str.contains(pattern, regex=False)
            norms.extend(ISSUER_LOOKUP.loc[mask, "issuer_norm"].tolist())
        norms = sorted(set(norms))
        return preset["label"], norms
    if selection.startswith("norm::"):
        norm = selection.split("norm::", 1)[1]
        match = ISSUER_LOOKUP.loc[ISSUER_LOOKUP["issuer_norm"] == norm]
        label = match["issuer_display"].iloc[0] if not match.empty else norm
        return label, [norm]
    return selection, []


def inferred_window(selection: str, preset: str) -> tuple[str, str]:
    if preset == "full_sample":
        return str(GLOBAL_MIN_DATE), str(GLOBAL_MAX_DATE)
    if preset == "last_24m":
        end = pd.Timestamp(GLOBAL_MAX_DATE)
        start = end - pd.DateOffset(months=24)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    if preset in EVENT_WINDOWS:
        return EVENT_WINDOWS[preset]
    if preset == "auto":
        if selection in ISSUER_PRESETS:
            auto = ISSUER_PRESETS[selection]["auto_window"]
            if auto in EVENT_WINDOWS:
                return EVENT_WINDOWS[auto]
        return inferred_window(selection, "last_24m")
    return str(GLOBAL_MIN_DATE), str(GLOBAL_MAX_DATE)


def issuer_group_source(grouping: str) -> pd.DataFrame:
    return ISSUER_CATEGORY if grouping == "category" else ISSUER_TYPE


def issuer_group_label(grouping: str, group_name: str) -> str:
    if grouping == "category":
        return CATEGORY_SHORT_LABELS.get(group_name, group_name)
    return group_name


def issuer_group_options(grouping: str, norms: list[str], start_date: str, end_date: str, preset: str, current_groups: list[str] | None) -> tuple[list[dict[str, str]], list[str]]:
    source = issuer_group_source(grouping)
    frame = source[(source["issuer_norm"].isin(norms)) & (source["as_of_date"] >= pd.Timestamp(start_date)) & (source["as_of_date"] <= pd.Timestamp(end_date))].copy()
    if frame.empty:
        return [], []
    frame = frame.groupby(["as_of_date", "category" if grouping == "category" else "group_name"], as_index=False)["value"].sum()
    name_col = "category" if grouping == "category" else "group_name"
    ranking = frame.groupby(name_col, as_index=False)["value"].sum().sort_values("value", ascending=False)
    ranking["label"] = ranking[name_col].map(lambda value: issuer_group_label(grouping, value))
    options = [{"label": f"{label}", "value": raw} for raw, label in zip(ranking[name_col], ranking["label"])]
    valid_groups = set(ranking[name_col].tolist())
    if preset == "custom" and current_groups is not None:
        selected = [group for group in current_groups if group in valid_groups]
    elif preset == "all":
        selected = ranking[name_col].tolist()
    else:
        selected = ranking[name_col].head(5).tolist()
    return options, selected


def build_issuer_exposure_figure(norms: list[str], start_date: str, end_date: str, issuer_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    dates = ALL_DATES[(ALL_DATES >= pd.Timestamp(start_date)) & (ALL_DATES <= pd.Timestamp(end_date))]
    holdings = ISSUER_EXPOSURE[(ISSUER_EXPOSURE["issuer_norm"].isin(norms)) & (ISSUER_EXPOSURE["as_of_date"] >= pd.Timestamp(start_date)) & (ISSUER_EXPOSURE["as_of_date"] <= pd.Timestamp(end_date))]
    holdings = (
        holdings.groupby("as_of_date", as_index=False)
        .agg(value=("value", "sum"), market_share=("market_share", "sum"))
        .set_index("as_of_date")
        .reindex(pd.Index(dates, name="as_of_date"))
        .fillna(0.0)
        .reset_index()
    )
    selected_date = resolve_snapshot_date(selected_date, holdings["as_of_date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=holdings["as_of_date"], y=holdings["value"] / 1e9, mode="lines+markers", name="Matched holdings exposure", line={"color": "#1D4ED8", "width": 3}))
    selected_row = holdings[holdings["as_of_date"] == selected_date]
    if not selected_row.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_row["as_of_date"],
                y=selected_row["value"] / 1e9,
                mode="markers",
                name="Selected month",
                marker={"size": 13, "color": "#D97706", "line": {"color": "#FFF7ED", "width": 2}},
                showlegend=False,
                hovertemplate="Selected<br>%{x|%Y-%m-%d}<br>%{y:,.1f}bn<extra></extra>",
            )
        )
    for trace in fig.data:
        if getattr(trace, "showlegend", True):
            trace.meta = str(trace.name)
            trace.name = wrap_legend_label(str(trace.name), width=26)
            trace.hovertemplate = "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f}bn<extra></extra>"
    fig.update_layout(title=f"{issuer_label}: matched exposure through time")
    fig = apply_historical_chart_style(fig, y_title="Exposure (USD bn)", legend_title=None, legend_orientation="v")
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_layout(margin={"l": 34, "r": 260, "t": 68, "b": 56}, legend={"x": 1.12})
    return fig


def build_issuer_mix_figure(norms: list[str], grouping: str, groups: list[str], start_date: str, end_date: str, metric: str, issuer_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    source = issuer_group_source(grouping)
    name_col = "category" if grouping == "category" else "group_name"
    frame = source[(source["issuer_norm"].isin(norms)) & (source["as_of_date"] >= pd.Timestamp(start_date)) & (source["as_of_date"] <= pd.Timestamp(end_date))].copy()
    if groups:
        frame = frame[frame[name_col].isin(groups)]
    singular_label, _ = grouping_labels(grouping)
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{issuer_label}: no matching {singular_label.lower()} data in selected window")
        return fig
    grouped = frame.groupby(["as_of_date", name_col], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "issuer_total"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["metric_value"] = grouped["value"] / 1e9 if metric == "value" else grouped["value"] / grouped["issuer_total"]
    label_map = {raw: issuer_group_label(grouping, raw) for raw in grouped[name_col].unique()}
    grouped["group_label"] = grouped[name_col].map(label_map)
    color_map = dataset_color_map(sorted(grouped["group_label"].unique()))
    complete_input = grouped[["as_of_date", "group_label", "metric_value"]].rename(columns={"group_label": "group_name", "metric_value": "plot_value"})
    complete = complete_group_frame(complete_input, pd.Index(sorted(grouped["as_of_date"].unique())), sorted(grouped["group_label"].unique()), "plot_value")
    fig = px.area(complete, x="as_of_date", y="plot_value", color="group_name", color_discrete_map=color_map)
    for trace in fig.data:
        original_name = str(trace.name).replace("<br>", " ")
        trace.meta = original_name
        trace.name = wrap_legend_label(original_name, width=16)
        trace.hovertemplate = trace_hover_template(metric)
    fig.update_layout(title=f"{issuer_label}: {singular_label} mix ({'USD bn' if metric == 'value' else '% of issuer exposure'})")
    fig = apply_historical_chart_style(
        fig,
        y_title="USD bn" if metric == "value" else "Share of issuer",
        y_tickformat=",.1f" if metric == "value" else ".0%",
        compact=True,
    )
    fig.update_xaxes(rangeslider={"visible": False})
    fig = add_selected_date_vline(fig, selected_date)
    fig.update_layout(
        margin={"l": 34, "r": 18, "t": 68, "b": 56},
        legend={"x": 1.22, "y": 1, "xanchor": "left", "yanchor": "top"},
    )
    return fig


def build_issuer_maturity_figure(norms: list[str], start_date: str, end_date: str, issuer_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    frame = ISSUER_MATURITY[(ISSUER_MATURITY["issuer_norm"].isin(norms)) & (ISSUER_MATURITY["as_of_date"] >= pd.Timestamp(start_date)) & (ISSUER_MATURITY["as_of_date"] <= pd.Timestamp(end_date))].copy()
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{issuer_label}: no maturity data in selected window")
        return fig
    grouped = frame.groupby("as_of_date", as_index=False).agg(
        weight_sum=("weight_sum", "sum"),
        weighted_maturity_component=("weighted_maturity_component", "sum"),
        weighted_final_component=("weighted_final_component", "sum"),
    )
    grouped["wa_maturity_days"] = grouped["weighted_maturity_component"] / grouped["weight_sum"]
    grouped["wa_final_maturity_days"] = grouped["weighted_final_component"] / grouped["weight_sum"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["as_of_date"], y=grouped["wa_maturity_days"], mode="lines+markers", name="WA maturity", line={"color": "#0F766E", "width": 3}))
    fig.add_trace(go.Scatter(x=grouped["as_of_date"], y=grouped["wa_final_maturity_days"], mode="lines+markers", name="WA final maturity", line={"color": "#7C3AED", "width": 3}))
    for trace in fig.data:
        trace.meta = str(trace.name)
        trace.hovertemplate = "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f} days<extra></extra>"
    fig.update_layout(title=f"{issuer_label}: maturity profile")
    fig = apply_historical_chart_style(fig, y_title="Days", compact=True, legend_title=None, legend_orientation="v")
    fig.update_xaxes(rangeslider={"visible": False})
    fig = add_selected_date_vline(fig, selected_date)
    return fig


def issuer_latest_table(norms: list[str], grouping: str, groups: list[str], start_date: str, end_date: str) -> list[dict[str, str]]:
    return issuer_snapshot_table(norms, grouping, groups, start_date, end_date, None)


def issuer_snapshot_table(
    norms: list[str],
    grouping: str,
    groups: list[str],
    start_date: str,
    end_date: str,
    selected_date: str | pd.Timestamp | None,
) -> list[dict[str, str]]:
    source = issuer_group_source(grouping)
    name_col = "category" if grouping == "category" else "group_name"
    frame = source[(source["issuer_norm"].isin(norms)) & (source["as_of_date"] >= pd.Timestamp(start_date)) & (source["as_of_date"] <= pd.Timestamp(end_date))].copy()
    if groups:
        frame = frame[frame[name_col].isin(groups)]
    if frame.empty:
        return []
    grouped = frame.groupby(["as_of_date", name_col], as_index=False)["value"].sum()
    snapshot_date = resolve_snapshot_date(selected_date, grouped["as_of_date"])
    latest = grouped[grouped["as_of_date"] == snapshot_date].sort_values("value", ascending=False).copy()
    latest_total = latest["value"].sum()
    latest["issuer_share"] = latest["value"] / latest_total if latest_total else 0.0
    singular_label, _ = grouping_labels(grouping)
    latest[singular_label] = latest[name_col].map(lambda value: issuer_group_label(grouping, value))
    latest["Value"] = latest["value"].map(lambda value: format_value("value", value))
    latest["Issuer share"] = latest["issuer_share"].map(lambda value: format_value("share", value))
    latest["As of"] = snapshot_date.strftime("%Y-%m-%d")
    return latest[[singular_label, "Value", "Issuer share", "As of"]].to_dict("records")


def build_issuer_maturity_distribution_figure(
    norms: list[str],
    type_groups: list[str],
    category_groups: list[str],
    start_date: str,
    end_date: str,
    selected_date: pd.Timestamp | None,
) -> go.Figure:
    subset = OVERVIEW_HOLDINGS[
        (OVERVIEW_HOLDINGS["issuer_norm"].isin(norms))
        & (OVERVIEW_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (OVERVIEW_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    snapshot_date = resolve_snapshot_date(selected_date, subset["as_of_date"])
    subset = subset[subset["as_of_date"] == snapshot_date].copy()
    if type_groups:
        subset = subset[subset["type"].isin(type_groups)]
    if category_groups:
        subset = subset[subset["category"].isin(category_groups)]
    subset = subset.dropna(subset=["maturity_date"])
    if subset.empty:
        empty = go.Figure()
        empty.update_layout(title=f"Maturity Distribution | {snapshot_date.date() if pd.notna(snapshot_date) else 'n/a'}: no matching rows")
        empty.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
        return empty

    days = (pd.to_datetime(subset["maturity_date"], errors="coerce") - subset["as_of_date"]).dt.days.clip(lower=0)
    subset["maturity_bucket"] = maturity_bucket_labels(days)
    maturity = subset.dropna(subset=["maturity_bucket"]).groupby("maturity_bucket", as_index=False)["value"].sum()
    maturity["share"] = maturity["value"] / maturity["value"].sum()
    bucket_order = ["Overnight", "2 - 7 Days", "8 - 30 Days", "31 - 60 Days", "61 - 90 Days", "91 - 180 Days", "181 - 365 Days", "366 + Days"]
    maturity["maturity_bucket"] = pd.Categorical(maturity["maturity_bucket"], categories=bucket_order, ordered=True)
    maturity = maturity.sort_values("maturity_bucket")
    fig = go.Figure(
        data=[
            go.Bar(
                x=maturity["maturity_bucket"].astype(str),
                y=maturity["share"],
                marker={"color": "#3B82F6", "line": {"color": "#1D4ED8", "width": 1.2}},
                text=[f"{value:.2%}" for value in maturity["share"]],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"Maturity Distribution | {snapshot_date.date()}",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 24, "r": 24, "t": 64, "b": 36},
        font={"family": "Arial", "size": 13, "color": "#16304A"},
        height=420,
        transition={"duration": 250},
    )
    fig.update_xaxes(title="Days", showgrid=False, fixedrange=True)
    fig.update_yaxes(title="Share of selected issuer subset", tickformat=".0%", showgrid=True, gridcolor="rgba(22,48,74,0.08)", fixedrange=True)
    return fig


def resolve_fund_selection(selection: str | None) -> tuple[str, list[str]]:
    if not selection:
        return "No fund selected", []
    return selection, [selection]


def fund_group_options(grouping: str, funds: list[str], start_date: str, end_date: str, preset: str, current_groups: list[str] | None) -> tuple[list[dict[str, str]], list[str]]:
    name_col = "category" if grouping == "category" else "type"
    frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    frame = frame[frame[name_col] != ""]
    if frame.empty:
        return [], []
    ranking = frame.groupby(name_col, as_index=False)["value"].sum().sort_values("value", ascending=False)
    ranking["label"] = ranking[name_col].map(lambda value: issuer_group_label(grouping, value))
    options = [{"label": label, "value": raw} for raw, label in zip(ranking[name_col], ranking["label"])]
    valid_groups = set(ranking[name_col].tolist())
    if preset == "custom" and current_groups is not None:
        selected = [group for group in current_groups if group in valid_groups]
    elif preset == "all":
        selected = ranking[name_col].tolist()
    else:
        selected = ranking[name_col].head(5).tolist()
    return options, selected


def build_fund_exposure_figure(funds: list[str], start_date: str, end_date: str, fund_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    dates = ALL_DATES[(ALL_DATES >= pd.Timestamp(start_date)) & (ALL_DATES <= pd.Timestamp(end_date))]
    holdings = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ]
    holdings = (
        holdings.groupby("as_of_date", as_index=False)
        .agg(value=("value", "sum"))
        .set_index("as_of_date")
        .reindex(pd.Index(dates, name="as_of_date"))
        .fillna(0.0)
        .reset_index()
    )
    selected_date = resolve_snapshot_date(selected_date, holdings["as_of_date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=holdings["as_of_date"], y=holdings["value"] / 1e9, mode="lines+markers", name="Fund assets", line={"color": "#1D4ED8", "width": 3}))
    selected_row = holdings[holdings["as_of_date"] == selected_date]
    if not selected_row.empty:
        fig.add_trace(
            go.Scatter(
                x=selected_row["as_of_date"],
                y=selected_row["value"] / 1e9,
                mode="markers",
                name="Selected month",
                marker={"size": 13, "color": "#D97706", "line": {"color": "#FFF7ED", "width": 2}},
                showlegend=False,
                hovertemplate="Selected<br>%{x|%Y-%m-%d}<br>%{y:,.1f}bn<extra></extra>",
            )
        )
    for trace in fig.data:
        if getattr(trace, "showlegend", True):
            trace.meta = str(trace.name)
            trace.name = wrap_legend_label(str(trace.name), width=24)
            trace.hovertemplate = "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f}bn<extra></extra>"
    fig.update_layout(title=f"{fund_label}: fund assets through time")
    fig = apply_historical_chart_style(fig, y_title="Assets (USD bn)", legend_title=None, legend_orientation="v")
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_layout(margin={"l": 34, "r": 240, "t": 68, "b": 56}, legend={"x": 1.08})
    return fig


def build_fund_mix_figure(funds: list[str], grouping: str, groups: list[str], start_date: str, end_date: str, metric: str, fund_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    name_col = "category" if grouping == "category" else "type"
    frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    frame = frame[frame[name_col] != ""]
    if groups:
        frame = frame[frame[name_col].isin(groups)]
    singular_label, _ = grouping_labels(grouping)
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{fund_label}: no matching {singular_label.lower()} data in selected window")
        return fig
    grouped = frame.groupby(["as_of_date", name_col], as_index=False)["value"].sum()
    totals = grouped.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "fund_total"})
    grouped = grouped.merge(totals, on="as_of_date", how="left")
    grouped["metric_value"] = grouped["value"] / 1e9 if metric == "value" else grouped["value"] / grouped["fund_total"]
    label_map = {raw: issuer_group_label(grouping, raw) for raw in grouped[name_col].unique()}
    grouped["group_label"] = grouped[name_col].map(label_map)
    color_map = dataset_color_map(sorted(grouped["group_label"].unique()))
    complete_input = grouped[["as_of_date", "group_label", "metric_value"]].rename(columns={"group_label": "group_name", "metric_value": "plot_value"})
    complete = complete_group_frame(complete_input, pd.Index(sorted(grouped["as_of_date"].unique())), sorted(grouped["group_label"].unique()), "plot_value")
    fig = px.area(complete, x="as_of_date", y="plot_value", color="group_name", color_discrete_map=color_map)
    for trace in fig.data:
        original_name = str(trace.name).replace("<br>", " ")
        trace.meta = original_name
        trace.name = wrap_legend_label(original_name, width=16)
        trace.hovertemplate = trace_hover_template(metric)
    fig.update_layout(title=f"{fund_label}: {singular_label} mix ({'USD bn' if metric == 'value' else '% of fund assets'})")
    fig = apply_historical_chart_style(
        fig,
        y_title="USD bn" if metric == "value" else "Share of fund",
        y_tickformat=",.1f" if metric == "value" else ".0%",
        compact=True,
    )
    fig.update_xaxes(rangeslider={"visible": False})
    fig = add_selected_date_vline(fig, selected_date)
    fig.update_layout(margin={"l": 34, "r": 18, "t": 68, "b": 56}, legend={"x": 1.22, "y": 1, "xanchor": "left", "yanchor": "top"})
    return fig


def build_fund_maturity_figure(funds: list[str], start_date: str, end_date: str, fund_label: str, selected_date: pd.Timestamp | None) -> go.Figure:
    frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    frame = frame.dropna(subset=["weight"])
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{fund_label}: no maturity data in selected window")
        return fig
    grouped = frame.groupby("as_of_date", as_index=False).agg(
        weight_sum=("weight", "sum"),
        weighted_maturity_component=("weighted_maturity_component", "sum"),
        weighted_final_component=("weighted_final_component", "sum"),
    )
    grouped["wa_maturity_days"] = grouped["weighted_maturity_component"] / grouped["weight_sum"]
    grouped["wa_final_maturity_days"] = grouped["weighted_final_component"] / grouped["weight_sum"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["as_of_date"], y=grouped["wa_maturity_days"], mode="lines+markers", name="WA maturity", line={"color": "#0F766E", "width": 3}))
    fig.add_trace(go.Scatter(x=grouped["as_of_date"], y=grouped["wa_final_maturity_days"], mode="lines+markers", name="WA final maturity", line={"color": "#7C3AED", "width": 3}))
    for trace in fig.data:
        trace.meta = str(trace.name)
        trace.hovertemplate = "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f} days<extra></extra>"
    fig.update_layout(title=f"{fund_label}: maturity profile")
    fig = apply_historical_chart_style(fig, y_title="Days", compact=True, legend_title=None, legend_orientation="v")
    fig.update_xaxes(rangeslider={"visible": False})
    fig = add_selected_date_vline(fig, selected_date)
    return fig


def fund_snapshot_table(
    funds: list[str],
    grouping: str,
    groups: list[str],
    start_date: str,
    end_date: str,
    selected_date: str | pd.Timestamp | None,
) -> list[dict[str, str]]:
    name_col = "category" if grouping == "category" else "type"
    frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    frame = frame[frame[name_col] != ""]
    if groups:
        frame = frame[frame[name_col].isin(groups)]
    if frame.empty:
        return []
    grouped = frame.groupby(["as_of_date", name_col], as_index=False)["value"].sum()
    snapshot_date = resolve_snapshot_date(selected_date, grouped["as_of_date"])
    latest = grouped[grouped["as_of_date"] == snapshot_date].sort_values("value", ascending=False).copy()
    latest_total = latest["value"].sum()
    latest["fund_share"] = latest["value"] / latest_total if latest_total else 0.0
    singular_label, _ = grouping_labels(grouping)
    latest[singular_label] = latest[name_col].map(lambda value: issuer_group_label(grouping, value))
    latest["Value"] = latest["value"].map(lambda value: format_value("value", value))
    latest["Fund share"] = latest["fund_share"].map(lambda value: format_value("share", value))
    latest["As of"] = snapshot_date.strftime("%Y-%m-%d")
    return latest[[singular_label, "Value", "Fund share", "As of"]].to_dict("records")


def build_fund_maturity_distribution_figure(
    funds: list[str],
    type_groups: list[str],
    category_groups: list[str],
    start_date: str,
    end_date: str,
    selected_date: pd.Timestamp | None,
) -> go.Figure:
    subset = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    snapshot_date = resolve_snapshot_date(selected_date, subset["as_of_date"])
    subset = subset[subset["as_of_date"] == snapshot_date].copy()
    if type_groups:
        subset = subset[subset["type"].isin(type_groups)]
    if category_groups:
        subset = subset[subset["category"].isin(category_groups)]
    subset = subset.dropna(subset=["maturity_date"])
    if subset.empty:
        empty = go.Figure()
        empty.update_layout(title=f"Maturity Distribution | {snapshot_date.date() if pd.notna(snapshot_date) else 'n/a'}: no matching rows")
        empty.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
        return empty
    days = (pd.to_datetime(subset["maturity_date"], errors="coerce") - subset["as_of_date"]).dt.days.clip(lower=0)
    subset["maturity_bucket"] = maturity_bucket_labels(days)
    maturity = subset.dropna(subset=["maturity_bucket"]).groupby("maturity_bucket", as_index=False)["value"].sum()
    maturity["share"] = maturity["value"] / maturity["value"].sum()
    bucket_order = ["Overnight", "2 - 7 Days", "8 - 30 Days", "31 - 60 Days", "61 - 90 Days", "91 - 180 Days", "181 - 365 Days", "366 + Days"]
    maturity["maturity_bucket"] = pd.Categorical(maturity["maturity_bucket"], categories=bucket_order, ordered=True)
    maturity = maturity.sort_values("maturity_bucket")
    fig = go.Figure(
        data=[
            go.Bar(
                x=maturity["maturity_bucket"].astype(str),
                y=maturity["share"],
                marker={"color": "#3B82F6", "line": {"color": "#1D4ED8", "width": 1.2}},
                text=[f"{value:.2%}" for value in maturity["share"]],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=f"Maturity Distribution | {snapshot_date.date()}", paper_bgcolor="white", plot_bgcolor="white", margin={"l": 24, "r": 24, "t": 64, "b": 36}, font={"family": "Arial", "size": 13, "color": "#16304A"}, height=420, transition={"duration": 250})
    fig.update_xaxes(title="Days", showgrid=False, fixedrange=True)
    fig.update_yaxes(title="Share of selected fund subset", tickformat=".0%", showgrid=True, gridcolor="rgba(22,48,74,0.08)", fixedrange=True)
    return fig


@app.callback(Output("app-shell", "children"), Input("url", "pathname"))
def route_page(pathname: str):
    return build_app_shell(pathname or "/").children


@app.callback(
    Output("mo-cards", "children"),
    Output("mo-issuers-graph", "figure"),
    Output("mo-funds-graph", "figure"),
    Input("mo-metric", "value"),
    Input("mo-chart", "value"),
    Input("mo-preset", "value"),
    Input("mo-date-range", "start_date"),
    Input("mo-date-range", "end_date"),
)
def update_market_overview(metric: str, chart_mode: str, preset: str, start_date: str, end_date: str):
    issuer_frame = filter_frame("issuers", start_date, end_date)
    fund_frame = filter_frame("funds", start_date, end_date)

    issuer_groups = selected_groups_for_frame("issuers", issuer_frame, preset, None)
    fund_groups = selected_groups_for_frame("funds", fund_frame, preset, None)

    issuer_fig = build_market_figure_with_options(
        issuer_frame,
        issuer_groups,
        metric,
        chart_mode,
        f"Top Issuers historical trend ({'USD bn' if metric == 'value' else 'normalized % share'})",
        normalize_share=(metric == "share"),
    )
    fund_fig = build_market_figure(
        fund_frame,
        fund_groups,
        metric,
        chart_mode,
        f"Top Funds historical trend ({'USD bn' if metric == 'value' else '% share'})",
    )

    latest_date = issuer_frame["as_of_date"].max() if not issuer_frame.empty else None
    issuer_latest = issuer_frame[(issuer_frame["as_of_date"] == latest_date) & (issuer_frame["group_name"].isin(issuer_groups))]
    fund_latest = fund_frame[(fund_frame["as_of_date"] == latest_date) & (fund_frame["group_name"].isin(fund_groups))]
    issuer_total = issuer_latest["value"].sum() if metric == "value" else (1.0 if issuer_latest["value"].sum() > 0 else 0.0)
    cards = summary_cards(
        [
            ("Latest month", str(latest_date.date()) if latest_date is not None else "n/a"),
            ("Issuer groups shown", str(len(issuer_groups))),
            ("Fund groups shown", str(len(fund_groups))),
            ("Issuer total", format_value(metric, issuer_total)),
            ("Fund total", format_value(metric, fund_latest["value"].sum() if metric == "value" else fund_latest["share"].sum())),
        ]
    )
    return cards, issuer_fig, fund_fig


@app.callback(
    Output("ov-filter-fund-type", "options"),
    Output("ov-filter-fund-type", "value"),
    Output("ov-filter-fund-family", "options"),
    Output("ov-filter-fund-family", "value"),
    Output("ov-filter-fund", "options"),
    Output("ov-filter-fund", "value"),
    Output("ov-filter-category", "options"),
    Output("ov-filter-category", "value"),
    Output("ov-filter-type", "options"),
    Output("ov-filter-type", "value"),
    Output("ov-filter-country", "options"),
    Output("ov-filter-country", "value"),
    Output("ov-filter-region", "options"),
    Output("ov-filter-region", "value"),
    Output("ov-filter-sector", "options"),
    Output("ov-filter-sector", "value"),
    Output("ov-description", "children"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def sync_overview_groups(
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    current_filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    filter_options, sanitized_filters, holdings_frame = overview_filter_state(start_date, end_date, current_filters)
    active_filters = [label for field, label, _ in OVERVIEW_FILTER_SPECS if sanitized_filters[field]]
    group_counts = {OVERVIEW_DATASET_LABELS[key]: build_overview_frame(key, start_date, end_date, sanitized_filters)["group_name"].nunique() for key in OVERVIEW_DATASETS}
    description = (
        "HoldingList-driven overview. "
        f"{len(holdings_frame):,} filtered rows across {holdings_frame['as_of_date'].nunique():,} months; "
        + "; ".join(f"{label}: {count:,} groups" for label, count in group_counts.items())
        + ". "
        f"Active workbook-style filters: {', '.join(active_filters) if active_filters else 'none'}."
    )
    outputs = []
    for field, _, _ in OVERVIEW_FILTER_SPECS:
        outputs.extend([filter_options[field], sanitized_filters[field]])
    outputs.append(description)
    return tuple(outputs)


@app.callback(
    Output("ov-cards", "children"),
    Input("ov-metric", "value"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_cards(
    metric: str,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, holdings_frame = overview_filter_state(start_date, end_date, filters)
    latest_date = holdings_frame["as_of_date"].max() if not holdings_frame.empty else None
    cards = summary_cards([
        ("Latest month", str(latest_date.date()) if latest_date is not None else "n/a"),
        ("Filtered rows", f"{len(holdings_frame):,}"),
        ("Funds", f"{holdings_frame['fund'].nunique():,}" if not holdings_frame.empty else "0"),
        ("Issuers", f"{holdings_frame['issuer'].nunique():,}" if not holdings_frame.empty else "0"),
        ("Countries", f"{holdings_frame['country'].nunique():,}" if not holdings_frame.empty else "0"),
    ])
    return cards


@app.callback(
    Output("ov-primary-graph", "figure"),
    Input("ov-metric", "value"),
    Input("ov-chart", "value"),
    Input("ov-preset", "value"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_primary_graph(
    metric: str,
    chart_mode: str,
    preset: str,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, _ = overview_filter_state(start_date, end_date, filters)
    dataset = "composition"
    frame = build_overview_frame(dataset, start_date, end_date, sanitized_filters)
    selected_groups = selected_groups_from_latest(frame, preset, None)
    title = f"{OVERVIEW_DATASET_LABELS[dataset]} historical trend ({'USD bn' if metric == 'value' else '% share'})"
    if frame.empty or not selected_groups:
        empty = go.Figure()
        empty.update_layout(title=f"{OVERVIEW_DATASET_LABELS[dataset]}: no data for current filters")
        return apply_historical_chart_style(empty, y_title="USD bn" if metric == "value" else "Share of total")
    fig = build_market_figure(frame, selected_groups, metric, chart_mode, title)
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_layout(transition={"duration": 250})
    return fig


@app.callback(
    Output("ov-secondary-graph", "figure"),
    Input("ov-metric", "value"),
    Input("ov-chart", "value"),
    Input("ov-preset", "value"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_secondary_graph(
    metric: str,
    chart_mode: str,
    preset: str,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, _ = overview_filter_state(start_date, end_date, filters)
    dataset = "country"
    frame = build_overview_frame(dataset, start_date, end_date, sanitized_filters)
    selected_groups = selected_groups_from_latest(frame, preset, None)
    title = f"{OVERVIEW_DATASET_LABELS[dataset]} historical trend ({'USD bn' if metric == 'value' else '% share'})"
    if frame.empty or not selected_groups:
        empty = go.Figure()
        empty.update_layout(title=f"{OVERVIEW_DATASET_LABELS[dataset]}: no data for current filters")
        return apply_historical_chart_style(empty, y_title="USD bn" if metric == "value" else "Share of total")
    fig = build_market_figure(frame, selected_groups, metric, chart_mode, title)
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_layout(transition={"duration": 250})
    return fig


@app.callback(
    Output("ov-selection-store", "data"),
    Input("ov-primary-graph", "clickData"),
    Input("ov-secondary-graph", "clickData"),
    Input("ov-timeline-graph", "clickData"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    State("ov-selection-store", "data"),
    prevent_initial_call=False,
)
def update_breakdown_selection(
    primary_click: dict | None,
    secondary_click: dict | None,
    timeline_click: dict | None,
    start_date: str,
    end_date: str,
    current_selection: dict | None,
):
    triggered = ctx.triggered_id
    if triggered == "ov-primary-graph" and primary_click and primary_click.get("points"):
        return {"dataset": "composition", "as_of_date": primary_click["points"][0]["x"]}
    if triggered == "ov-secondary-graph" and secondary_click and secondary_click.get("points"):
        return {"dataset": "country", "as_of_date": secondary_click["points"][0]["x"]}
    if triggered == "ov-timeline-graph" and timeline_click and timeline_click.get("points"):
        return {
            "dataset": (current_selection or {}).get("dataset", "composition"),
            "as_of_date": timeline_click["points"][0]["x"],
        }
    default_date = resolve_snapshot_date(end_date, OVERVIEW_HOLDINGS["as_of_date"])
    return {
        "dataset": (current_selection or {}).get("dataset", "composition"),
        "as_of_date": default_date.strftime("%Y-%m-%d") if pd.notna(default_date) else end_date,
    }


@app.callback(
    Output("ov-timeline-graph", "figure"),
    Input("ov-selection-store", "data"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_timeline_graph(
    selection: dict | None,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, filtered_holdings = overview_filter_state(start_date, end_date, filters)
    filtered_holdings = apply_overview_filters(start_date, end_date, sanitized_filters)
    if filtered_holdings.empty:
        empty = go.Figure()
        empty.update_layout(title="Selected time point: no data")
        empty = apply_historical_chart_style(empty, y_title="Filtered universe (USD bn)", height=260, compact=True, legend_title=None)
        empty.update_xaxes(rangeslider={"visible": False})
        return empty
    selected_date = resolve_snapshot_date((selection or {}).get("as_of_date", end_date), filtered_holdings["as_of_date"])
    return build_breakdown_timeline_figure(filtered_holdings, selected_date)


@app.callback(
    Output("ov-composition-pie", "figure"),
    Output("ov-country-pie", "figure"),
    Input("ov-selection-store", "data"),
    Input("ov-preset", "value"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_snapshot_pies(
    selection: dict | None,
    preset: str,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, filtered_holdings = overview_filter_state(start_date, end_date, filters)
    filtered_holdings = apply_overview_filters(start_date, end_date, sanitized_filters)
    selected_date = resolve_snapshot_date((selection or {}).get("as_of_date", end_date), filtered_holdings["as_of_date"])

    composition_frame = build_overview_frame("composition", start_date, end_date, sanitized_filters)
    country_frame = build_overview_frame("country", start_date, end_date, sanitized_filters)
    composition_groups = selected_groups_from_latest(composition_frame, preset, None)
    country_groups = selected_groups_from_latest(country_frame, preset, None)
    return (
        build_snapshot_pie_figure(composition_frame, dataset="composition", selected_date=selected_date, groups=composition_groups),
        build_snapshot_pie_figure(country_frame, dataset="country", selected_date=selected_date, groups=country_groups),
    )


@app.callback(
    Output("ov-maturity-graph", "figure"),
    Output("ov-funds-table", "data"),
    Output("ov-issuers-table", "data"),
    Output("ov-funds-title", "children"),
    Output("ov-issuers-title", "children"),
    Input("ov-selection-store", "data"),
    Input("ov-preset", "value"),
    Input("ov-date-range", "start_date"),
    Input("ov-date-range", "end_date"),
    Input("ov-filter-fund-type", "value"),
    Input("ov-filter-fund-family", "value"),
    Input("ov-filter-fund", "value"),
    Input("ov-filter-category", "value"),
    Input("ov-filter-type", "value"),
    Input("ov-filter-country", "value"),
    Input("ov-filter-region", "value"),
    Input("ov-filter-sector", "value"),
)
def update_breakdown_detail_panel(
    selection: dict | None,
    preset: str,
    start_date: str,
    end_date: str,
    fund_type_values: list[str] | None,
    fund_family_values: list[str] | None,
    fund_values: list[str] | None,
    category_values: list[str] | None,
    type_values: list[str] | None,
    country_values: list[str] | None,
    region_values: list[str] | None,
    sector_values: list[str] | None,
):
    filters = {
        "fund_type": fund_type_values or [],
        "fund_family": fund_family_values or [],
        "fund": fund_values or [],
        "category": category_values or [],
        "type": type_values or [],
        "country": country_values or [],
        "region": region_values or [],
        "sector": sector_values or [],
    }
    _, sanitized_filters, filtered_holdings = overview_filter_state(start_date, end_date, filters)
    dataset = (selection or {}).get("dataset", "composition")
    selected_date = resolve_snapshot_date((selection or {}).get("as_of_date", end_date), filtered_holdings["as_of_date"])
    if pd.isna(selected_date):
        selected_date = resolve_snapshot_date(end_date, OVERVIEW_HOLDINGS["as_of_date"])

    overview_frame = build_overview_frame(dataset, start_date, end_date, sanitized_filters)
    selected_groups = selected_groups_from_latest(overview_frame, preset, None)
    group_column = OVERVIEW_GROUP_COLUMNS[dataset]
    funds_title = f"Top 10 Funds | {selected_date.date()} | {OVERVIEW_DATASET_LABELS[dataset]}"
    issuers_title = f"Top 50 Issuers | {selected_date.date()} | {OVERVIEW_DATASET_LABELS[dataset]}"

    subset = filtered_holdings[filtered_holdings["as_of_date"] == selected_date].copy()
    if selected_groups:
        subset = subset[subset[group_column].isin(selected_groups)]
    subset = subset.dropna(subset=["maturity_date"])

    if subset.empty:
        empty = go.Figure()
        empty.update_layout(title=f"Maturity Distribution ({OVERVIEW_DATASET_LABELS[dataset]}, {selected_date.date()}): no matching rows")
        empty = apply_historical_chart_style(empty, y_title="Share of selected subset", height=440, legend_title=None, compact=True, legend_orientation="h")
        empty.update_xaxes(rangeslider={"visible": False})
        empty.update_xaxes(fixedrange=True)
        empty.update_yaxes(fixedrange=True)
        return empty, [], [], funds_title, issuers_title

    days = (pd.to_datetime(subset["maturity_date"], errors="coerce") - subset["as_of_date"]).dt.days.clip(lower=0)
    subset["maturity_bucket"] = maturity_bucket_labels(days)
    maturity = subset.dropna(subset=["maturity_bucket"]).groupby("maturity_bucket", as_index=False)["value"].sum()
    maturity["share"] = maturity["value"] / maturity["value"].sum()
    bucket_order = ["Overnight", "2 - 7 Days", "8 - 30 Days", "31 - 60 Days", "61 - 90 Days", "91 - 180 Days", "181 - 365 Days", "366 + Days"]
    maturity["maturity_bucket"] = pd.Categorical(maturity["maturity_bucket"], categories=bucket_order, ordered=True)
    maturity = maturity.sort_values("maturity_bucket")
    maturity_fig = go.Figure(
        data=[
            go.Bar(
                x=maturity["maturity_bucket"].astype(str),
                y=maturity["share"],
                marker={"color": "#3B82F6", "line": {"color": "#1D4ED8", "width": 1.2}},
                text=[f"{value:.2%}" for value in maturity["share"]],
                textposition="outside",
                hovertemplate="%{x}<br>%{y:.2%}<extra></extra>",
            )
        ]
    )
    maturity_fig.update_layout(
        title=f"Maturity Distribution | {OVERVIEW_DATASET_LABELS[dataset]} | {selected_date.date()}",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 24, "r": 24, "t": 64, "b": 36},
        font={"family": "Arial", "size": 13, "color": "#16304A"},
        height=440,
        transition={"duration": 250},
    )
    maturity_fig.update_xaxes(title="Days", showgrid=False, fixedrange=True)
    maturity_fig.update_yaxes(title="Share of selected subset", tickformat=".0%", showgrid=True, gridcolor="rgba(22,48,74,0.08)", fixedrange=True)

    total_value = subset["value"].sum()
    funds = (
        subset.groupby("fund", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    )
    funds["Share"] = funds["value"] / total_value
    funds["Value"] = funds["value"].map(lambda value: format_value("value", value))
    funds["Share"] = funds["Share"].map(lambda value: format_value("share", value))
    funds = funds.rename(columns={"fund": "Fund"})[["Fund", "Value", "Share"]].to_dict("records")

    issuers = (
        subset.groupby("issuer", as_index=False)["value"].sum().sort_values("value", ascending=False).head(50)
    )
    issuers["Share"] = issuers["value"] / total_value
    issuers["Value"] = issuers["value"].map(lambda value: format_value("value", value))
    issuers["Share"] = issuers["Share"].map(lambda value: format_value("share", value))
    issuers = issuers.rename(columns={"issuer": "Issuer"})[["Issuer", "Value", "Share"]].to_dict("records")

    return maturity_fig, funds, issuers, funds_title, issuers_title


@app.callback(
    Output("ct-groups", "options"),
    Output("ct-groups", "value"),
    Output("ct-description", "children"),
    Input("ct-preset", "value"),
    Input("ct-date-range", "start_date"),
    Input("ct-date-range", "end_date"),
    Input("ct-groups", "value"),
)
def sync_country_groups(preset: str, start_date: str, end_date: str, current_groups: list[str] | None):
    frame = build_holdings_group_frame("country", start_date, end_date, exclude_groups=DEFAULT_EXCLUDE["country"], unknown_label="Unspecified Country")
    if frame.empty:
        return [], [], "HoldingList-based country view: no data for the current date range."
    all_groups = (
        frame.groupby("group_name", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)["group_name"]
        .tolist()
    )
    options = [{"label": group, "value": group} for group in all_groups]
    if preset == "custom":
        selected = sanitize_selection(current_groups, all_groups)
        if safe_triggered_id() != "ct-groups" and not selected:
            selected = all_groups
    else:
        selected = selected_groups_from_latest(frame, "top_10" if preset == "top_10" else "top_5", current_groups)
    description = (
        f"HoldingList-based country view. {frame['group_name'].nunique():,} countries available between "
        f"{frame['as_of_date'].min().date()} and {frame['as_of_date'].max().date()}. "
        "Country shares are recomputed from monthly HoldingList totals. Months before 2011-06 do not carry country tags in the raw HoldingList, so they are shown as Unspecified Country."
    )
    return options, selected, description


@app.callback(
    Output("ct-preset", "value", allow_duplicate=True),
    Input("ct-groups", "value"),
    State("ct-preset", "value"),
    State("ct-date-range", "start_date"),
    State("ct-date-range", "end_date"),
    prevent_initial_call=True,
)
def sync_country_preset_with_groups(current_groups: list[str] | None, preset: str, start_date: str, end_date: str):
    if preset == "custom":
        return no_update
    frame = build_holdings_group_frame("country", start_date, end_date, exclude_groups=DEFAULT_EXCLUDE["country"], unknown_label="Unspecified Country")
    expected = selected_groups_from_latest(frame, "top_10" if preset == "top_10" else "top_5", None)
    if set(current_groups or []) != set(expected):
        return "custom"
    return no_update


@app.callback(
    Output("ct-cards", "children"),
    Output("ct-graph", "figure"),
    Output("ct-region-graph", "figure"),
    Input("ct-preset", "value"),
    Input("ct-metric", "value"),
    Input("ct-chart", "value"),
    Input("ct-date-range", "start_date"),
    Input("ct-date-range", "end_date"),
    Input("ct-groups", "value"),
)
def update_country(preset: str, metric: str, chart_mode: str, start_date: str, end_date: str, groups: list[str]):
    frame = build_holdings_group_frame("country", start_date, end_date, exclude_groups=DEFAULT_EXCLUDE["country"], unknown_label="Unspecified Country")
    region_frame = build_holdings_group_frame("region", start_date, end_date, unknown_label="Unspecified Region")
    if frame.empty or region_frame.empty:
        empty = go.Figure()
        empty.update_layout(title="No data for the selected date range")
        empty = apply_historical_chart_style(empty, y_title="USD bn" if metric == "value" else "Share of total")
        empty.update_xaxes(rangeslider={"visible": False})
        return summary_cards([("Latest month", "n/a"), ("Selected countries", "0"), ("Regions shown", "0"), ("Country total", "n/a"), ("Region total", "n/a")]), empty, empty
    if preset == "custom":
        selected_groups = groups or (
            frame.groupby("group_name", as_index=False)["value"]
            .sum()
            .sort_values("value", ascending=False)["group_name"]
            .tolist()
        )
    else:
        selected_groups = groups or selected_groups_from_latest(frame, "top_10" if preset == "top_10" else "top_5", groups)
    selected_region_groups = selected_groups_from_latest(region_frame, "all", None)
    latest_date = frame["as_of_date"].max()
    latest = frame[(frame["as_of_date"] == latest_date) & (frame["group_name"].isin(selected_groups))]
    latest_regions = region_frame[(region_frame["as_of_date"] == latest_date) & (region_frame["group_name"].isin(selected_region_groups))]
    latest_total = latest["value"].sum() if metric == "value" else latest["share"].sum()
    region_total = latest_regions["value"].sum() if metric == "value" else latest_regions["share"].sum()
    cards = summary_cards([
        ("Latest month", str(latest_date.date())),
        ("Selected countries", str(len(selected_groups))),
        ("Regions shown", str(len(selected_region_groups))),
        ("Country total", format_value(metric, latest_total)),
        ("Region total", format_value(metric, region_total)),
    ])
    country_fig = build_market_figure(frame, selected_groups, metric, chart_mode, f"Country exposure historical trend ({'USD bn' if metric == 'value' else '% share'})")
    country_fig.update_xaxes(rangeslider={"visible": False})
    region_fig = build_market_figure(region_frame, selected_region_groups, metric, chart_mode, f"Region exposure historical trend ({'USD bn' if metric == 'value' else '% share'})")
    region_fig.update_xaxes(rangeslider={"visible": False})
    return cards, country_fig, region_fig


@app.callback(
    Output("ct-selection-store", "data"),
    Input("ct-graph", "clickData"),
    Input("ct-region-graph", "clickData"),
    Input("ct-timeline-graph", "clickData"),
    Input("ct-date-range", "start_date"),
    Input("ct-date-range", "end_date"),
    prevent_initial_call=False,
)
def update_country_selection(country_click: dict | None, region_click: dict | None, timeline_click: dict | None, start_date: str, end_date: str):
    triggered = ctx.triggered_id
    if triggered == "ct-graph" and country_click and country_click.get("points"):
        return {"as_of_date": country_click["points"][0]["x"]}
    if triggered == "ct-region-graph" and region_click and region_click.get("points"):
        return {"as_of_date": region_click["points"][0]["x"]}
    if triggered == "ct-timeline-graph" and timeline_click and timeline_click.get("points"):
        return {"as_of_date": timeline_click["points"][0]["x"]}
    default_date = resolve_snapshot_date(end_date, OVERVIEW_HOLDINGS["as_of_date"])
    return {"as_of_date": default_date.strftime("%Y-%m-%d") if pd.notna(default_date) else end_date}


@app.callback(
    Output("ct-timeline-graph", "figure"),
    Output("ct-country-pie", "figure"),
    Output("ct-region-pie", "figure"),
    Output("ct-country-table", "data"),
    Output("ct-region-table", "data"),
    Output("ct-country-title", "children"),
    Output("ct-region-title", "children"),
    Input("ct-selection-store", "data"),
    Input("ct-date-range", "start_date"),
    Input("ct-date-range", "end_date"),
    Input("ct-groups", "value"),
)
def update_country_snapshot_panel(selection: dict | None, start_date: str, end_date: str, groups: list[str] | None):
    country_frame = build_holdings_group_frame("country", start_date, end_date, exclude_groups=DEFAULT_EXCLUDE["country"], unknown_label="Unspecified Country")
    region_frame = build_holdings_group_frame("region", start_date, end_date, unknown_label="Unspecified Region")
    if country_frame.empty or region_frame.empty:
        empty = go.Figure()
        empty.update_layout(title="No data for the selected date range")
        empty.update_layout(height=460, paper_bgcolor="white", plot_bgcolor="white")
        return empty, empty, empty, [], [], "Country breakout snapshot", "Region breakout snapshot"
    selected_date = resolve_snapshot_date((selection or {}).get("as_of_date", end_date), country_frame["as_of_date"])
    timeline_fig = build_breakdown_timeline_figure(
        OVERVIEW_HOLDINGS[(OVERVIEW_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date)) & (OVERVIEW_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))],
        selected_date,
    )
    country_groups = groups or (
        country_frame.groupby("group_name", as_index=False)["value"]
        .sum()
        .sort_values("value", ascending=False)["group_name"]
        .tolist()
    )
    region_groups = selected_groups_from_latest(region_frame, "all", None)
    country_pie = build_snapshot_pie_figure(country_frame, dataset="country", selected_date=selected_date, groups=country_groups)
    region_pie = build_snapshot_pie_figure(region_frame, dataset="region", selected_date=selected_date, groups=region_groups)
    country_table = build_snapshot_table(country_frame, selected_date=selected_date, groups=country_groups, label_column="group_name", label_header="Country")
    region_table = build_snapshot_table(region_frame, selected_date=selected_date, groups=region_groups, label_column="group_name", label_header="Region")
    return (
        timeline_fig,
        country_pie,
        region_pie,
        country_table,
        region_table,
        f"Country breakout snapshot | {selected_date.date()}",
        f"Region breakout snapshot | {selected_date.date()}",
    )


@app.callback(
    Output("mt-subset-values", "options"),
    Output("mt-subset-values", "value"),
    Output("mt-description", "children"),
    Input("mt-subset-by", "value"),
    Input("mt-date-range", "start_date"),
    Input("mt-date-range", "end_date"),
    Input("mt-subset-values", "value"),
)
def sync_maturity_subset_values(subset_by: str, start_date: str, end_date: str, current_values: list[str] | None):
    return maturity_subset_options(subset_by, start_date, end_date, current_values)


@app.callback(
    Output("mt-cards", "children"),
    Output("mt-buckets-graph", "figure"),
    Output("mt-lines-graph", "figure"),
    Output("mt-table", "data"),
    Input("mt-metric", "value"),
    Input("mt-chart", "value"),
    Input("mt-subset-by", "value"),
    Input("mt-subset-values", "value"),
    Input("mt-date-range", "start_date"),
    Input("mt-date-range", "end_date"),
    Input("mt-groups", "value"),
)
def update_maturity(metric: str, chart_mode: str, subset_by: str, subset_values: list[str], start_date: str, end_date: str, groups: list[str]):
    detail_frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    detail_frame = filter_maturity_subset(detail_frame, subset_by, subset_values)
    detail_frame = detail_frame.dropna(subset=["as_of_date", "value", "maturity_date"])
    if detail_frame.empty:
        empty = go.Figure()
        empty.update_layout(title="No maturity data for the selected subset")
        empty = apply_historical_chart_style(empty, y_title="USD bn" if metric == "value" else "Share of total")
        return summary_cards([("Latest month", "n/a"), ("Selected subset", "0"), ("Selected total", "0.0bn"), ("Latest WA maturity", "n/a"), ("Latest WA final", "n/a")]), empty, empty, []

    days = (detail_frame["maturity_date"] - detail_frame["as_of_date"]).dt.days.clip(lower=0)
    detail_frame["group_name"] = maturity_bucket_labels(days)
    detail_frame = detail_frame.dropna(subset=["group_name"])
    frame = detail_frame.groupby(["as_of_date", "group_name"], as_index=False)["value"].sum()
    totals = frame.groupby("as_of_date", as_index=False)["value"].sum().rename(columns={"value": "total_value"})
    frame = frame.merge(totals, on="as_of_date", how="left")
    frame["share"] = frame["value"] / frame["total_value"]
    selected_groups = groups or MATURITY_GROUPS
    latest_date = frame["as_of_date"].max()
    latest = frame[(frame["as_of_date"] == latest_date) & (frame["group_name"].isin(selected_groups))]
    latest_metric_total = latest["value"].sum() if metric == "value" else latest["share"].sum()
    latest_line = detail_frame.groupby("as_of_date", as_index=False).agg(
        weight_sum=("weight", "sum"),
        weighted_maturity_component=("weighted_maturity_component", "sum"),
        weighted_final_component=("weighted_final_component", "sum"),
    ).sort_values("as_of_date")
    latest_line["wa_maturity_days"] = latest_line["weighted_maturity_component"] / latest_line["weight_sum"]
    latest_line["wa_final_maturity_days"] = latest_line["weighted_final_component"] / latest_line["weight_sum"]
    latest_row = latest_line.iloc[-1]
    subset_label = {
        "market": "All holdings",
        "type": f"{len(subset_values or [])} types",
        "category": f"{len(subset_values or [])} categories",
        "issuer": f"{len(subset_values or [])} issuers",
        "fund": f"{len(subset_values or [])} funds",
    }.get(subset_by, "Custom subset")
    cards = summary_cards([
        ("Latest month", str(latest_date.date())),
        ("Selected subset", subset_label),
        ("Selected total", format_value(metric, latest_metric_total)),
        ("Latest WA maturity", f"{latest_row['wa_maturity_days']:.1f} days"),
        ("Latest WA final", f"{latest_row['wa_final_maturity_days']:.1f} days"),
    ])
    title_prefix = {
        "market": "Market",
        "type": "Type-filtered",
        "category": "Category-filtered",
        "issuer": "Issuer-filtered",
        "fund": "Fund-filtered",
    }.get(subset_by, "Subset")
    bucket_fig = build_market_figure(frame, selected_groups, metric, chart_mode, f"{title_prefix} maturity bucket trend ({'USD bn' if metric == 'value' else '% share'})")
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=latest_line["as_of_date"], y=latest_line["wa_maturity_days"], mode="lines+markers", name="WA maturity", line={"color": "#0F766E", "width": 3}))
    line_fig.add_trace(go.Scatter(x=latest_line["as_of_date"], y=latest_line["wa_final_maturity_days"], mode="lines+markers", name="WA final maturity", line={"color": "#7C3AED", "width": 3}))
    for trace in line_fig.data:
        trace.meta = str(trace.name)
        trace.hovertemplate = "%{x|%Y-%m-%d}<br>%{meta}: %{y:,.1f} days<extra></extra>"
    line_fig.update_layout(title=f"{title_prefix} weighted-average maturity profile")
    line_fig = apply_historical_chart_style(line_fig, y_title="Days", compact=True, legend_title=None, legend_orientation="v")
    return cards, bucket_fig, line_fig, build_latest_table(frame, selected_groups, metric)


@app.callback(
    Output("is-date-range", "start_date"),
    Output("is-date-range", "end_date"),
    Input("is-issuer", "value"),
    Input("is-window", "value"),
)
def update_issuer_window(selection: str, preset: str):
    if preset == "custom":
        return no_update, no_update
    start_date, end_date = inferred_window(selection, preset)
    return start_date, end_date


@app.callback(
    Output("is-type-preset", "value", allow_duplicate=True),
    Input("is-types", "value"),
    State("is-issuer", "value"),
    State("is-type-preset", "value"),
    State("is-date-range", "start_date"),
    State("is-date-range", "end_date"),
    prevent_initial_call=True,
)
def sync_issuer_type_preset_with_groups(current_groups: list[str] | None, selection: str, preset: str, start_date: str, end_date: str):
    if preset == "custom":
        return no_update
    _, norms = resolve_issuer_selection(selection)
    if not norms:
        return no_update
    _, expected = issuer_group_options("type", norms, start_date, end_date, preset, None)
    if set(current_groups or []) != set(expected):
        return "custom"
    return no_update


@app.callback(
    Output("is-category-preset", "value", allow_duplicate=True),
    Input("is-categories", "value"),
    State("is-issuer", "value"),
    State("is-category-preset", "value"),
    State("is-date-range", "start_date"),
    State("is-date-range", "end_date"),
    prevent_initial_call=True,
)
def sync_issuer_category_preset_with_groups(current_groups: list[str] | None, selection: str, preset: str, start_date: str, end_date: str):
    if preset == "custom":
        return no_update
    _, norms = resolve_issuer_selection(selection)
    if not norms:
        return no_update
    _, expected = issuer_group_options("category", norms, start_date, end_date, preset, None)
    if set(current_groups or []) != set(expected):
        return "custom"
    return no_update


@app.callback(
    Output("is-types", "options"),
    Output("is-types", "value"),
    Input("is-issuer", "value"),
    Input("is-type-preset", "value"),
    Input("is-date-range", "start_date"),
    Input("is-date-range", "end_date"),
    Input("is-types", "value"),
)
def sync_issuer_type_groups(selection: str, preset: str, start_date: str, end_date: str, current_groups: list[str] | None):
    _, norms = resolve_issuer_selection(selection)
    options, selected = issuer_group_options("type", norms, start_date, end_date, preset, current_groups)
    if preset == "custom" and safe_triggered_id() == "is-types":
        valid_values = [option["value"] for option in options]
        selected = sanitize_selection(current_groups, valid_values)
    return options, selected


@app.callback(
    Output("is-categories", "options"),
    Output("is-categories", "value"),
    Input("is-issuer", "value"),
    Input("is-category-preset", "value"),
    Input("is-date-range", "start_date"),
    Input("is-date-range", "end_date"),
    Input("is-categories", "value"),
)
def sync_issuer_category_groups(selection: str, preset: str, start_date: str, end_date: str, current_groups: list[str] | None):
    _, norms = resolve_issuer_selection(selection)
    options, selected = issuer_group_options("category", norms, start_date, end_date, preset, current_groups)
    if preset == "custom" and safe_triggered_id() == "is-categories":
        valid_values = [option["value"] for option in options]
        selected = sanitize_selection(current_groups, valid_values)
    return options, selected


@app.callback(
    Output("is-description", "children"),
    Input("is-issuer", "value"),
    Input("is-date-range", "start_date"),
    Input("is-date-range", "end_date"),
)
def update_issuer_description(selection: str, start_date: str, end_date: str):
    issuer_label, norms = resolve_issuer_selection(selection)
    if not norms:
        return "No matching issuer keys were found for this selection."
    matched_labels = ISSUER_LOOKUP.loc[ISSUER_LOOKUP["issuer_norm"].isin(norms), "issuer_display"].head(6).tolist()
    suffix = "" if len(norms) <= 6 else " ..."
    return (
        f"{issuer_label}: matched {len(norms)} issuer key(s). Examples: {', '.join(matched_labels)}{suffix}. "
        "Both Type and Category controls are built across the full selected window, not just the last month, "
        "so event windows like Credit Suisse can still surface repo / CP / CD buckets that roll off by the end of the window. "
        "The exposure chart shows the matched HoldingList series only."
    )


@app.callback(
    Output("is-selection-store", "data"),
    Input("is-exposure-graph", "clickData"),
    Input("is-type-value", "clickData"),
    Input("is-type-share", "clickData"),
    Input("is-category-value", "clickData"),
    Input("is-category-share", "clickData"),
    Input("is-maturity-graph", "clickData"),
    Input("is-date-range", "end_date"),
    prevent_initial_call=False,
)
def update_issuer_selection(exposure_click: dict | None, type_value_click: dict | None, type_share_click: dict | None, category_value_click: dict | None, category_share_click: dict | None, maturity_click: dict | None, end_date: str):
    triggered = ctx.triggered_id
    clicked = {
        "is-exposure-graph": exposure_click,
        "is-type-value": type_value_click,
        "is-type-share": type_share_click,
        "is-category-value": category_value_click,
        "is-category-share": category_share_click,
        "is-maturity-graph": maturity_click,
    }.get(triggered)
    if clicked and clicked.get("points"):
        return {"as_of_date": clicked["points"][0]["x"]}
    default_date = resolve_snapshot_date(end_date, ALL_DATES)
    return {"as_of_date": default_date.strftime("%Y-%m-%d") if pd.notna(default_date) else end_date}


@app.callback(
    Output("is-cards", "children"),
    Output("is-exposure-graph", "figure"),
    Output("is-type-value", "figure"),
    Output("is-type-share", "figure"),
    Output("is-category-value", "figure"),
    Output("is-category-share", "figure"),
    Output("is-maturity-graph", "figure"),
    Output("is-maturity-distribution", "figure"),
    Output("is-type-table", "data"),
    Output("is-category-table", "data"),
    Output("is-type-title", "children"),
    Output("is-category-title", "children"),
    Input("is-issuer", "value"),
    Input("is-date-range", "start_date"),
    Input("is-date-range", "end_date"),
    Input("is-types", "value"),
    Input("is-categories", "value"),
    Input("is-selection-store", "data"),
)
def update_issuer_page(selection: str, start_date: str, end_date: str, type_groups: list[str], category_groups: list[str], snapshot_selection: dict | None):
    issuer_label, norms = resolve_issuer_selection(selection)
    if not norms:
        empty = go.Figure()
        empty.update_layout(title="No matching issuer data")
        return (
            summary_cards([("Issuer", issuer_label), ("Matched keys", "0"), ("Window", f"{start_date} to {end_date}")]),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
            [],
            "Type snapshot",
            "Category snapshot",
        )

    dates = ALL_DATES[(ALL_DATES >= pd.Timestamp(start_date)) & (ALL_DATES <= pd.Timestamp(end_date))]
    exposure_frame = ISSUER_EXPOSURE[(ISSUER_EXPOSURE["issuer_norm"].isin(norms)) & (ISSUER_EXPOSURE["as_of_date"] >= pd.Timestamp(start_date)) & (ISSUER_EXPOSURE["as_of_date"] <= pd.Timestamp(end_date))]
    exposure_series = exposure_frame.groupby("as_of_date", as_index=False).agg(value=("value", "sum"), market_share=("market_share", "sum")).set_index("as_of_date").reindex(dates).fillna(0.0).reset_index()
    peak_row = exposure_series.sort_values("value", ascending=False).iloc[0]
    start_value = exposure_series.iloc[0]["value"]
    end_value = exposure_series.iloc[-1]["value"]
    delta = end_value - start_value
    maturity_frame = ISSUER_MATURITY[(ISSUER_MATURITY["issuer_norm"].isin(norms)) & (ISSUER_MATURITY["as_of_date"] >= pd.Timestamp(start_date)) & (ISSUER_MATURITY["as_of_date"] <= pd.Timestamp(end_date))]
    if maturity_frame.empty:
        latest_wam = float("nan")
        latest_wafm = float("nan")
    else:
        maturity_grouped = maturity_frame.groupby("as_of_date", as_index=False).agg(weight_sum=("weight_sum", "sum"), weighted_maturity_component=("weighted_maturity_component", "sum"), weighted_final_component=("weighted_final_component", "sum"))
        maturity_grouped["wa_maturity_days"] = maturity_grouped["weighted_maturity_component"] / maturity_grouped["weight_sum"]
        maturity_grouped["wa_final_maturity_days"] = maturity_grouped["weighted_final_component"] / maturity_grouped["weight_sum"]
        latest_wam = maturity_grouped.iloc[-1]["wa_maturity_days"]
        latest_wafm = maturity_grouped.iloc[-1]["wa_final_maturity_days"]

    cards = summary_cards([
        ("Issuer", issuer_label),
        ("Matched keys", str(len(norms))),
        ("Peak exposure", format_value("value", peak_row["value"])),
        ("Window change", f"{delta / 1e9:,.1f}bn"),
        ("Latest WA maturity", f"{latest_wam:.1f} days" if pd.notna(latest_wam) else "n/a"),
        ("Latest WA final", f"{latest_wafm:.1f} days" if pd.notna(latest_wafm) else "n/a"),
    ])
    selected_date = resolve_snapshot_date((snapshot_selection or {}).get("as_of_date", end_date), dates)
    exposure_fig = build_issuer_exposure_figure(norms, start_date, end_date, issuer_label, selected_date)
    type_value_fig = build_issuer_mix_figure(norms, "type", type_groups or [], start_date, end_date, "value", issuer_label, selected_date)
    type_share_fig = build_issuer_mix_figure(norms, "type", type_groups or [], start_date, end_date, "share", issuer_label, selected_date)
    category_value_fig = build_issuer_mix_figure(norms, "category", category_groups or [], start_date, end_date, "value", issuer_label, selected_date)
    category_share_fig = build_issuer_mix_figure(norms, "category", category_groups or [], start_date, end_date, "share", issuer_label, selected_date)
    maturity_fig = build_issuer_maturity_figure(norms, start_date, end_date, issuer_label, selected_date)
    maturity_distribution_fig = build_issuer_maturity_distribution_figure(norms, type_groups or [], category_groups or [], start_date, end_date, selected_date)
    type_table = issuer_snapshot_table(norms, "type", type_groups or [], start_date, end_date, selected_date)
    category_table = issuer_snapshot_table(norms, "category", category_groups or [], start_date, end_date, selected_date)
    return (
        cards,
        exposure_fig,
        type_value_fig,
        type_share_fig,
        category_value_fig,
        category_share_fig,
        maturity_fig,
        maturity_distribution_fig,
        type_table,
        category_table,
        f"Type snapshot | {selected_date.date()}",
        f"Category snapshot | {selected_date.date()}",
    )


@app.callback(
    Output("fs-date-range", "start_date"),
    Output("fs-date-range", "end_date"),
    Input("fs-fund", "value"),
    Input("fs-window", "value"),
)
def update_fund_window(selection: str, preset: str):
    if preset == "custom":
        return no_update, no_update
    start_date, end_date = inferred_window(selection, preset)
    return start_date, end_date


@app.callback(
    Output("fs-type-preset", "value", allow_duplicate=True),
    Input("fs-types", "value"),
    State("fs-fund", "value"),
    State("fs-type-preset", "value"),
    State("fs-date-range", "start_date"),
    State("fs-date-range", "end_date"),
    prevent_initial_call=True,
)
def sync_fund_type_preset_with_groups(current_groups: list[str] | None, selection: str, preset: str, start_date: str, end_date: str):
    if preset == "custom":
        return no_update
    _, funds = resolve_fund_selection(selection)
    if not funds:
        return no_update
    _, expected = fund_group_options("type", funds, start_date, end_date, preset, None)
    if set(current_groups or []) != set(expected):
        return "custom"
    return no_update


@app.callback(
    Output("fs-category-preset", "value", allow_duplicate=True),
    Input("fs-categories", "value"),
    State("fs-fund", "value"),
    State("fs-category-preset", "value"),
    State("fs-date-range", "start_date"),
    State("fs-date-range", "end_date"),
    prevent_initial_call=True,
)
def sync_fund_category_preset_with_groups(current_groups: list[str] | None, selection: str, preset: str, start_date: str, end_date: str):
    if preset == "custom":
        return no_update
    _, funds = resolve_fund_selection(selection)
    if not funds:
        return no_update
    _, expected = fund_group_options("category", funds, start_date, end_date, preset, None)
    if set(current_groups or []) != set(expected):
        return "custom"
    return no_update


@app.callback(
    Output("fs-types", "options"),
    Output("fs-types", "value"),
    Input("fs-fund", "value"),
    Input("fs-type-preset", "value"),
    Input("fs-date-range", "start_date"),
    Input("fs-date-range", "end_date"),
    Input("fs-types", "value"),
)
def sync_fund_type_groups(selection: str, preset: str, start_date: str, end_date: str, current_groups: list[str] | None):
    _, funds = resolve_fund_selection(selection)
    options, selected = fund_group_options("type", funds, start_date, end_date, preset, current_groups)
    if preset == "custom" and safe_triggered_id() == "fs-types":
        valid_values = [option["value"] for option in options]
        selected = sanitize_selection(current_groups, valid_values)
    return options, selected


@app.callback(
    Output("fs-categories", "options"),
    Output("fs-categories", "value"),
    Input("fs-fund", "value"),
    Input("fs-category-preset", "value"),
    Input("fs-date-range", "start_date"),
    Input("fs-date-range", "end_date"),
    Input("fs-categories", "value"),
)
def sync_fund_category_groups(selection: str, preset: str, start_date: str, end_date: str, current_groups: list[str] | None):
    _, funds = resolve_fund_selection(selection)
    options, selected = fund_group_options("category", funds, start_date, end_date, preset, current_groups)
    if preset == "custom" and safe_triggered_id() == "fs-categories":
        valid_values = [option["value"] for option in options]
        selected = sanitize_selection(current_groups, valid_values)
    return options, selected


@app.callback(
    Output("fs-description", "children"),
    Input("fs-fund", "value"),
    Input("fs-date-range", "start_date"),
    Input("fs-date-range", "end_date"),
)
def update_fund_description(selection: str, start_date: str, end_date: str):
    fund_label, funds = resolve_fund_selection(selection)
    if not funds:
        return "No matching fund was found for this selection."
    subset = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ]
    family = subset["fund_family"].replace("", pd.NA).dropna().mode()
    fund_type = subset["fund_type"].replace("", pd.NA).dropna().mode()
    family_text = family.iloc[0] if not family.empty else "n/a"
    type_text = fund_type.iloc[0] if not fund_type.empty else "n/a"
    return (
        f"{fund_label}: fund-family view centered on a single fund. "
        f"Fund family {family_text}; fund type {type_text}. "
        "Both Type and Category controls are built across the full selected window, not just the last month, "
        "so temporary buckets do not disappear from the selector after they roll off."
    )


@app.callback(
    Output("fs-selection-store", "data"),
    Input("fs-exposure-graph", "clickData"),
    Input("fs-type-value", "clickData"),
    Input("fs-type-share", "clickData"),
    Input("fs-category-value", "clickData"),
    Input("fs-category-share", "clickData"),
    Input("fs-maturity-graph", "clickData"),
    Input("fs-date-range", "end_date"),
    prevent_initial_call=False,
)
def update_fund_selection(exposure_click: dict | None, type_value_click: dict | None, type_share_click: dict | None, category_value_click: dict | None, category_share_click: dict | None, maturity_click: dict | None, end_date: str):
    triggered = ctx.triggered_id
    clicked = {
        "fs-exposure-graph": exposure_click,
        "fs-type-value": type_value_click,
        "fs-type-share": type_share_click,
        "fs-category-value": category_value_click,
        "fs-category-share": category_share_click,
        "fs-maturity-graph": maturity_click,
    }.get(triggered)
    if clicked and clicked.get("points"):
        return {"as_of_date": clicked["points"][0]["x"]}
    default_date = resolve_snapshot_date(end_date, ALL_DATES)
    return {"as_of_date": default_date.strftime("%Y-%m-%d") if pd.notna(default_date) else end_date}


@app.callback(
    Output("fs-cards", "children"),
    Output("fs-exposure-graph", "figure"),
    Output("fs-type-value", "figure"),
    Output("fs-type-share", "figure"),
    Output("fs-category-value", "figure"),
    Output("fs-category-share", "figure"),
    Output("fs-maturity-graph", "figure"),
    Output("fs-maturity-distribution", "figure"),
    Output("fs-type-table", "data"),
    Output("fs-category-table", "data"),
    Output("fs-type-title", "children"),
    Output("fs-category-title", "children"),
    Input("fs-fund", "value"),
    Input("fs-date-range", "start_date"),
    Input("fs-date-range", "end_date"),
    Input("fs-types", "value"),
    Input("fs-categories", "value"),
    Input("fs-selection-store", "data"),
)
def update_fund_page(selection: str, start_date: str, end_date: str, type_groups: list[str], category_groups: list[str], snapshot_selection: dict | None):
    fund_label, funds = resolve_fund_selection(selection)
    if not funds:
        empty = go.Figure()
        empty.update_layout(title="No matching fund data")
        return (
            summary_cards([("Fund", fund_label), ("Matched funds", "0"), ("Window", f"{start_date} to {end_date}")]),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
            [],
            "Type snapshot",
            "Category snapshot",
        )

    dates = ALL_DATES[(ALL_DATES >= pd.Timestamp(start_date)) & (ALL_DATES <= pd.Timestamp(end_date))]
    fund_frame = FUND_DETAIL_HOLDINGS[
        (FUND_DETAIL_HOLDINGS["fund"].isin(funds))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] >= pd.Timestamp(start_date))
        & (FUND_DETAIL_HOLDINGS["as_of_date"] <= pd.Timestamp(end_date))
    ].copy()
    exposure_series = (
        fund_frame.groupby("as_of_date", as_index=False)
        .agg(value=("value", "sum"))
        .set_index("as_of_date")
        .reindex(dates)
        .fillna(0.0)
        .reset_index()
    )
    peak_row = exposure_series.sort_values("value", ascending=False).iloc[0]
    start_value = exposure_series.iloc[0]["value"]
    end_value = exposure_series.iloc[-1]["value"]
    delta = end_value - start_value

    maturity_frame = fund_frame.dropna(subset=["weight"]).copy()
    if maturity_frame.empty:
        latest_wam = float("nan")
        latest_wafm = float("nan")
    else:
        maturity_grouped = maturity_frame.groupby("as_of_date", as_index=False).agg(
            weight_sum=("weight", "sum"),
            weighted_maturity_component=("weighted_maturity_component", "sum"),
            weighted_final_component=("weighted_final_component", "sum"),
        )
        maturity_grouped["wa_maturity_days"] = maturity_grouped["weighted_maturity_component"] / maturity_grouped["weight_sum"]
        maturity_grouped["wa_final_maturity_days"] = maturity_grouped["weighted_final_component"] / maturity_grouped["weight_sum"]
        latest_wam = maturity_grouped.iloc[-1]["wa_maturity_days"]
        latest_wafm = maturity_grouped.iloc[-1]["wa_final_maturity_days"]

    cards = summary_cards([
        ("Fund", fund_label),
        ("Matched funds", str(len(funds))),
        ("Peak assets", format_value("value", peak_row["value"])),
        ("Window change", f"{delta / 1e9:,.1f}bn"),
        ("Latest WA maturity", f"{latest_wam:.1f} days" if pd.notna(latest_wam) else "n/a"),
        ("Latest WA final", f"{latest_wafm:.1f} days" if pd.notna(latest_wafm) else "n/a"),
    ])
    selected_date = resolve_snapshot_date((snapshot_selection or {}).get("as_of_date", end_date), dates)
    exposure_fig = build_fund_exposure_figure(funds, start_date, end_date, fund_label, selected_date)
    type_value_fig = build_fund_mix_figure(funds, "type", type_groups or [], start_date, end_date, "value", fund_label, selected_date)
    type_share_fig = build_fund_mix_figure(funds, "type", type_groups or [], start_date, end_date, "share", fund_label, selected_date)
    category_value_fig = build_fund_mix_figure(funds, "category", category_groups or [], start_date, end_date, "value", fund_label, selected_date)
    category_share_fig = build_fund_mix_figure(funds, "category", category_groups or [], start_date, end_date, "share", fund_label, selected_date)
    maturity_fig = build_fund_maturity_figure(funds, start_date, end_date, fund_label, selected_date)
    maturity_distribution_fig = build_fund_maturity_distribution_figure(funds, type_groups or [], category_groups or [], start_date, end_date, selected_date)
    type_table = fund_snapshot_table(funds, "type", type_groups or [], start_date, end_date, selected_date)
    category_table = fund_snapshot_table(funds, "category", category_groups or [], start_date, end_date, selected_date)
    return (
        cards,
        exposure_fig,
        type_value_fig,
        type_share_fig,
        category_value_fig,
        category_share_fig,
        maturity_fig,
        maturity_distribution_fig,
        type_table,
        category_table,
        f"Type snapshot | {selected_date.date()}",
        f"Category snapshot | {selected_date.date()}",
    )


for graph_id in [
    "mo-issuers-graph",
    "mo-funds-graph",
    "ov-primary-graph",
    "ov-secondary-graph",
    "ct-graph",
    "ct-region-graph",
    "mt-buckets-graph",
    "mt-lines-graph",
    "is-exposure-graph",
    "is-type-value",
    "is-type-share",
    "is-category-value",
    "is-category-share",
    "is-maturity-graph",
    "fs-exposure-graph",
    "fs-type-value",
    "fs-type-share",
    "fs-category-value",
    "fs-category-share",
    "fs-maturity-graph",
]:

    @app.callback(
        Output(graph_id, "figure", allow_duplicate=True),
        Input(graph_id, "relayoutData"),
        State(graph_id, "figure"),
        prevent_initial_call=True,
    )
    def _sync_graph_zoom(relayout_data, figure, _graph_id=graph_id):
        return sync_x_zoom_autorange(figure, relayout_data)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
