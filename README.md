# crane

Plotly Dash app for historical money fund holdings analysis built from Crane monthly hold reports.

## Included

- `dashapp/`: multi-page Dash application
- `code/crane_lib/`: shared helpers used by the app data-prep layer

## Not Included

This repository intentionally does **not** include raw data or processed data outputs:

- `data/`
- `processed_data/`
- `dashapp/data/`
- `results/`
- `report/`

## Run

1. Prepare the processed parquet inputs locally.
2. Build the dashboard-ready trend tables:

```bash
python dashapp/prepare_dash_data.py
```

3. Start the app:

```bash
python dashapp/app.py
```

4. Open:

```text
http://127.0.0.1:8052
```

## Notes

- The app is designed to sit alongside locally prepared Crane hold report datasets.
- Data files are excluded from version control to keep the repository lightweight and private-data safe.
