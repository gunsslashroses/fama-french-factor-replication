# AGENTS.md

## Cursor Cloud specific instructions

This is a Python/Jupyter notebook project for replicating Fama-French SMB and HML factor portfolios from CRSP/Compustat data.

### Project structure
- `ff_factor_replication.ipynb` — Main Jupyter notebook (the sole deliverable)
- `requirements.txt` — Python dependencies
- `data/` — Directory for datasets (CRSP/Compustat)

### Running Jupyter
```
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token="" --ServerApp.password=""
```
Jupyter Lab will be available at `http://localhost:8888`.

### Key libraries
pandas, numpy, matplotlib, statsmodels, scipy, wrds (for WRDS database access).

### Gotchas
- `pip install` places binaries in `~/.local/bin` which may not be on PATH. The update script handles this, but if running manually, use `export PATH="$HOME/.local/bin:$PATH"`.
- The `wrds` package requires WRDS account credentials for actual data access. For local development without WRDS access, sample/mock data can be used instead.
- The notebook file in the repo is currently a placeholder; it needs valid `.ipynb` JSON to be opened in Jupyter.
