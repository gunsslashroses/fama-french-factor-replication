# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python/Jupyter data science project for replicating Fama-French SMB/HML factors from raw WRDS data (Compustat, CRSP, CCM link table). The main artifact is `ff_factor_replication.ipynb`.

All custom logic (helpers, pipeline steps) lives inside the notebook — do not extract into a separate package.

### Running JupyterLab

```bash
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token="" --ServerApp.password=""
```

Jupyter binaries install to `~/.local/bin`; ensure `PATH` includes it.

### Executing notebooks headlessly

```bash
jupyter nbconvert --to notebook --execute ff_factor_replication.ipynb --output executed.ipynb
```

The notebook requires either WRDS credentials or local data files in `data/`. Without either, execution will fail at the data-loading cell.

### Running tests

```bash
pytest tests/ -v
```

Smoke tests extract helper functions from the notebook and validate core logic (BE construction, breakpoints, portfolio labels, SMB/HML formulas, syntax). No WRDS access needed.

### Key packages

`pandas`, `numpy`, `matplotlib`, `scipy`, `statsmodels`, `wrds`, `pandas-datareader`, `jupyter` — see `requirements.txt`.

### Data source

The notebook supports two modes set via `DATA_SOURCE` in the config cell:
- `'wrds'` — live SQL queries (needs WRDS username/password)
- `'local'` — reads Parquet/CSV from `data/` directory

### Gotchas

- The `build_notebook.py` script is gitignored; it's only used to regenerate the notebook programmatically.
- CRSP `prc` can be negative (bid/ask midpoint flag) — always use `abs(prc)` for market equity.
- The `count >= 1` filter is 0-indexed, meaning at least 2 years of Compustat history.
