# Crane Dash App

Pure Plotly Dash app for exploring the historical trend view behind the `FundCharts` tab in the Crane monthly hold reports.

## What it covers

The app is now split into four pages:

- `Market Overview`
- `Country`
- `Maturity`
- `Issuer Deep Dive`

Across those pages, it translates the main first-tab chart families into historical time series:

- `Top Issuers`
- `Composition`
- `Maturity Buckets`
- `By Country`
- `Top Funds`

The `Issuer Deep Dive` page adds:

- issuer search with alias-aware presets for `Credit Suisse`, `Bank of America / BAC`, `Deutsche Bank / DB`, and `UBS`
- event window presets such as `CS stress`, `Rate hikes`, `DB 2016`, and `BAC 2011`
- exposure, product mix, and maturity views through time

Interactive controls include:

- interactive date-range selection
- top-N or custom group selection
- `USD bn` and `% share` metrics
- line and stacked-area modes

## Data sources

The app reads from the processed parquet files already created in this repo:

- `/Users/cp/Projects/crane/processed_data/issuers.parquet`
- `/Users/cp/Projects/crane/processed_data/composition.parquet`
- `/Users/cp/Projects/crane/processed_data/country.parquet`
- `/Users/cp/Projects/crane/processed_data/holdings.parquet`

It materializes dashboard-ready trend tables into:

- `/Users/cp/Projects/crane/dashapp/data/trend_data.parquet`
- `/Users/cp/Projects/crane/dashapp/data/dataset_meta.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_lookup.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_exposure.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_summary.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_category.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_type.parquet`
- `/Users/cp/Projects/crane/dashapp/data/issuer_maturity.parquet`
- `/Users/cp/Projects/crane/dashapp/data/market_maturity.parquet`

## Run

From the project root:

```bash
/Users/cp/Projects/crane/.venv/bin/python /Users/cp/Projects/crane/dashapp/prepare_dash_data.py
/Users/cp/Projects/crane/.venv/bin/python /Users/cp/Projects/crane/dashapp/app.py
```

Then open:

```text
http://127.0.0.1:8052
```

## Notes

- `Composition` and `By Country` include workbook roll-up rows in the raw source. The app keeps them available in the group selector but excludes them from the default view so the chart opens on more informative detailed lines.
- `Maturity Buckets` are rebuilt from `HoldingList` maturity dates using the same bucket structure shown in the workbook.
- The issuer page uses precomputed monthly issuer aggregates so the event-window views stay responsive inside Dash.
- Alias-aware issuer presets are intentionally narrower than free-form substring search, so they behave more like a clean issuer lens and less like a raw text search over every security description.
