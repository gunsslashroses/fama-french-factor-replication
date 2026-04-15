# Fama-French SMB & HML Factor Replication

Replication of the **SMB** (Small Minus Big) and **HML** (High Minus Low) factors
from Fama and French (1993), "Common Risk Factors in the Returns on Stocks and Bonds",
*Journal of Financial Economics*, 33(1), 3-56.

## Overview

This notebook-first project builds the Fama-French factors from **raw WRDS data**:

| WRDS Table | Contents |
|------------|----------|
| `comp.funda` | Compustat annual fundamentals (book equity) |
| `crsp.msf` + `crsp.msenames` | CRSP monthly stock file with exchange/share codes |
| `crsp.msedelist` | CRSP delisting returns |
| `crsp.ccmxpf_linktable` | CRSP-Compustat merged link table |

The replicated factors are compared against the **official Ken French factors**
from his [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure data source

Open `ff_factor_replication.ipynb` and set the configuration cell:

```python
DATA_SOURCE = 'wrds'        # or 'local'
START_DATE  = '2000-01-01'  # adjust as needed
```

**Option A — WRDS live query** (requires a [WRDS account](https://wrds-www.wharton.upenn.edu/)):
- Set `DATA_SOURCE = 'wrds'`
- Optionally set `WRDS_USERNAME = 'your_id'`

**Option B — Local files**:
- Set `DATA_SOURCE = 'local'`
- Place Parquet or CSV files in `data/` and update the path variables

### 3. Run the notebook

```bash
jupyter lab ff_factor_replication.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute ff_factor_replication.ipynb
```

## Pipeline Structure

1. **Compustat block** — preferred stock fallback hierarchy, book equity, positive-BE filter
2. **CRSP block** — delisting returns, retadj, market equity, permco aggregation, Dec ME, July-June weights
3. **CCM block** — link Compustat to CRSP, compute book-to-market
4. **NYSE breakpoints** — median ME (size), 30th/70th B/M percentiles
5. **Portfolio assignment** — 2×3 sorts (SL, SM, SH, BL, BM, BH)
6. **Factor construction** — value-weighted returns → SMB and HML
7. **Comparison** — correlation, beta, R², cointegration, time-series plots

## Project Structure

```
ff_factor_replication.ipynb   # Main notebook (all logic lives here)
requirements.txt              # Python dependencies
pyproject.toml                # Project metadata
data/                         # Raw/cached data files (gitignored)
  README.md                   # Data directory documentation
tests/
  test_notebook_smoke.py      # Smoke tests for notebook helper functions
```

## Testing

```bash
pytest tests/ -v
```

The smoke tests extract helper functions from the notebook and validate:
- Compustat book equity construction
- NYSE breakpoint logic
- June portfolio labels (size and B/M buckets)
- SMB/HML formulas
- All notebook code cells parse with `ast.parse`

## Reference

Fama, E.F. and French, K.R. (1993), "Common Risk Factors in the Returns on
Stocks and Bonds", *Journal of Financial Economics*, 33(1), 3-56.
