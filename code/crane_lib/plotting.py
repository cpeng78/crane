from __future__ import annotations

import os

from .constants import ROOT


os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def billions(series: pd.Series) -> pd.Series:
    return series / 1e9


def style_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str = "USD bn",
    xlabel: str = "As-of date (month-end)",
    date_format: str = "%Y-%m",
) -> None:
    ax.set_title(title, fontsize=12, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.tick_params(axis="x", rotation=45)


def set_month_ticks(
    ax: plt.Axes,
    dates: pd.Series | pd.Index,
    fmt: str = "%Y-%m-%d",
    every: int = 1,
) -> None:
    tick_dates = pd.to_datetime(pd.Series(dates)).dropna().drop_duplicates().sort_values()
    if every > 1:
        tick_dates = tick_dates.iloc[::every]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels([d.strftime(fmt) for d in tick_dates], rotation=45, ha="right")
