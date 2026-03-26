from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "processed_data"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"

FUNDING_CATEGORIES = {
    "Asset Backed Commercial Paper",
    "Certificate of Deposit",
    "Financial Company Commercial Paper",
    "Non-Negotiable Time Deposit",
    "Insurance Company Funding Agreement",
    "Other Repurchase Agreement",
    "U.S. Government Agency Repurchase Agreement",
    "U.S. Treasury Repurchase Agreement",
}

COMPOSITION_SUMMARY_CATEGORIES = {
    "Agency",
    "CD",
    "CP",
    "Grand Total",
    "Other",
    "Repo",
    "Total Holdings",
    "Treasury",
}

CATEGORY_SHORT_LABELS = {
    "Asset Backed Commercial Paper": "ABCP",
    "Bank Notes": "Bank Notes",
    "Certificate of Deposit": "CD",
    "Commercial Paper": "CP",
    "Financial Company Commercial Paper": "Financial CP",
    "Government Agency Repurchase Agreement": "Agency Repo",
    "Government Agency Repurchase Agreements": "Agency Repo",
    "Government Agency Debt": "Agency Debt",
    "Insurance Company Funding Agreement": "Funding Agreement",
    "Investment Company": "Investment Co.",
    "Non-Financial Company Commercial Paper": "Non-Financial CP",
    "Non-Negotiable Time Deposit": "NTD",
    "Non-U.S. Sovereign, Sub-Sovereign and Supra-National Debt": "Non-U.S. Sov.",
    "Other Commercial Paper": "Other CP",
    "Other Asset Backed Securities": "Other ABS",
    "Other Instrument": "Other Instrument",
    "Other Instrument (Time Deposit)": "Time Deposit",
    "Other matched categories": "Other",
    "Other Municipal Debt": "Other Muni Debt",
    "Other Municipal Security": "Other Muni Security",
    "Other Note": "Other Note",
    "Other Repurchase Agreement": "Other Repo",
    "Tender Option Bond": "TOB",
    "U.S. Government Agency Debt": "Agency Debt",
    "U.S. Government Agency Repurchase Agreement": "Agency Repo",
    "U.S. Treasury Debt": "Treasury Debt",
    "U.S. Treasury Repurchase Agreement": "Treasury Repo",
    "Treasury Repurchase Agreement": "Treasury Repo",
    "Variable Rate Demand Note": "VRDN",
}

CATEGORY_COLORS = {
    "Asset Backed Commercial Paper": "#E377C2",
    "Bank Notes": "#9C755F",
    "Certificate of Deposit": "#D62728",
    "Commercial Paper": "#FF7F0E",
    "Financial Company Commercial Paper": "#FF9D0A",
    "Government Agency Repurchase Agreement": "#2CA02C",
    "Government Agency Repurchase Agreements": "#2CA02C",
    "Government Agency Debt": "#8C564B",
    "Insurance Company Funding Agreement": "#76B7B2",
    "Investment Company": "#B07AA1",
    "Non-Financial Company Commercial Paper": "#EDC948",
    "Non-Negotiable Time Deposit": "#BCBD22",
    "Non-U.S. Sovereign, Sub-Sovereign and Supra-National Debt": "#8C564B",
    "Other Commercial Paper": "#F4A259",
    "Other Asset Backed Securities": "#AF7AA1",
    "Other Instrument": "#7F7F7F",
    "Other Instrument (Time Deposit)": "#BAB0AC",
    "Other matched categories": "#C7C7C7",
    "Other Municipal Debt": "#59A14F",
    "Other Municipal Security": "#8CD17D",
    "Other Note": "#4E5D6C",
    "Other Repurchase Agreement": "#17BECF",
    "Tender Option Bond": "#9467BD",
    "U.S. Government Agency Debt": "#8C564B",
    "U.S. Government Agency Repurchase Agreement": "#2CA02C",
    "U.S. Treasury Debt": "#1F4E79",
    "U.S. Treasury Repurchase Agreement": "#1F77B4",
    "Treasury Repurchase Agreement": "#1F77B4",
    "Variable Rate Demand Note": "#9467BD",
}

TYPE_ORDER = ["Repo", "Treasury", "Agency", "CP", "CD", "VRDN", "Other", "Municipal Bond"]
TYPE_SHORT_LABELS = {
    "Repo": "Repo",
    "Treasury": "Treasury",
    "Agency": "Agency",
    "CP": "CP",
    "CD": "CD",
    "VRDN": "VRDN",
    "Other": "Other",
    "Municipal Bond": "Municipal Bond",
}
TYPE_COLORS = {
    "Repo": "#1F77B4",
    "Treasury": "#FF7F0E",
    "Agency": "#2CA02C",
    "CP": "#D62728",
    "CD": "#17BECF",
    "VRDN": "#9467BD",
    "Other": "#7F7F7F",
    "Municipal Bond": "#BCBD22",
}

CS_PATTERN = re.compile(r"credit suisse|credit suisse firstboston|csfb", re.IGNORECASE)
BAC_PATTERN = re.compile(r"bank of america|banc of america", re.IGNORECASE)
CD_PATTERN = re.compile(r"certificate of deposit", re.IGNORECASE)
REGIONAL_BANK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"first republic",
        r"western alliance",
        r"pacwest",
        r"comerica",
        r"regions bank",
        r"regions financial",
        r"keybank",
        r"citizens bank",
        r"citizens financial",
        r"zions",
        r"fifth third",
        r"huntington",
        r"truist",
        r"m&t bank",
        r"webster bank",
        r"east west bank",
        r"bokf",
        r"bok financial",
    ]
]
