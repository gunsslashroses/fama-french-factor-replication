# Data Directory

This directory holds raw and cached data files used by the notebook.
Files here are **gitignored** — they are either queried from WRDS or provided locally.

## Expected Files (when using `DATA_SOURCE = 'local'`)

| File | Description | Source |
|------|-------------|--------|
| `compustat_data.parquet` | Compustat annual fundamentals | `comp.funda` |
| `crsp_data.parquet` | CRSP monthly returns with delisting | `crsp.msf` + `crsp.msenames` + `crsp.msedelist` |
| `ccm_data.parquet` | CRSP-Compustat link table | `crsp.ccmxpf_linktable` |

Files can be Parquet (`.parquet`) or CSV (`.csv`). Update the path variables
in the notebook configuration cell accordingly.

## Generating Local Files

You can save WRDS query results for offline use:

```python
import wrds
conn = wrds.Connection()

comp = conn.raw_sql("SELECT ... FROM comp.funda WHERE ...")
comp.to_parquet('data/compustat_data.parquet')

# Similarly for crsp_data and ccm_data
```
