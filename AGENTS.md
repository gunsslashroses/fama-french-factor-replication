# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

This is a Python/Jupyter data science project for replicating Fama-French SMB/HML factors from CRSP/Compustat data. The main artifact is `ff_factor_replication.ipynb`.

### Running JupyterLab

```
jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token="" --ServerApp.password=""
```

Jupyter binaries install to `~/.local/bin`; ensure `PATH` includes it (`export PATH="$HOME/.local/bin:$PATH"`).

### Executing notebooks headlessly

```
jupyter nbconvert --to notebook --execute <notebook>.ipynb --output <output>.ipynb
```

### Key packages

pandas, numpy, matplotlib, scipy, statsmodels, wrds — installed via `pip install`.

### Lint / format

No linter or formatter is configured in this repository yet. If one is added, check `requirements.txt` or a config file (e.g. `pyproject.toml`, `.flake8`, `ruff.toml`).

### Testing

No test framework is configured. If tests are added, look for a `tests/` directory and a pytest/unittest configuration.
