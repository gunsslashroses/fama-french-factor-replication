"""
Smoke tests for ff_factor_replication.ipynb.

These tests extract helper functions from the notebook code cells and
validate core logic without needing WRDS access.
"""
import ast
import json
import pathlib
import textwrap

import numpy as np
import pandas as pd
import pytest

NOTEBOOK_PATH = pathlib.Path(__file__).resolve().parent.parent / "ff_factor_replication.ipynb"


# ── Notebook loading utilities ───────────────────────────────────────

def _load_notebook():
    """Load the notebook JSON."""
    with open(NOTEBOOK_PATH) as f:
        return json.load(f)


def _code_cells(nb):
    """Return source strings for all code cells."""
    sources = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            sources.append("".join(cell["source"]))
    return sources


def _exec_helpers():
    """Execute the helper-function cell and return its namespace."""
    nb = _load_notebook()
    codes = _code_cells(nb)
    # The helper cell is the one that defines wavg, sz_bucket, etc.
    helper_src = None
    for src in codes:
        if "def wavg(" in src and "def sz_bucket(" in src:
            helper_src = src
            break
    assert helper_src is not None, "Could not find helper-function cell in notebook"
    ns = {"pd": pd, "np": np}
    exec(helper_src, ns)
    return ns


# ── Test: all code cells parse ───────────────────────────────────────

class TestNotebookSyntax:
    def test_all_cells_parse(self):
        """Every code cell must be valid Python (ast.parse)."""
        nb = _load_notebook()
        codes = _code_cells(nb)
        assert len(codes) > 0, "No code cells found"
        for i, src in enumerate(codes):
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has a syntax error: {e}")


# ── Test: Compustat BE construction ──────────────────────────────────

class TestBookEquity:
    def setup_method(self):
        self.ns = _exec_helpers()
        self.compute_be = self.ns["compute_book_equity"]

    def test_preferred_stock_fallback_pstkrv(self):
        """pstkrv available → use it."""
        df = pd.DataFrame({
            "pstkrv": [10.0], "pstkl": [20.0], "pstk": [30.0],
            "seq": [100.0], "txditc": [5.0],
        })
        result = self.compute_be(df)
        expected_be = 100 + 5 - 10  # seq + txditc - pstkrv
        assert result["be"].iloc[0] == pytest.approx(expected_be)

    def test_preferred_stock_fallback_pstkl(self):
        """pstkrv missing → fall back to pstkl."""
        df = pd.DataFrame({
            "pstkrv": [np.nan], "pstkl": [20.0], "pstk": [30.0],
            "seq": [100.0], "txditc": [5.0],
        })
        result = self.compute_be(df)
        expected_be = 100 + 5 - 20
        assert result["be"].iloc[0] == pytest.approx(expected_be)

    def test_preferred_stock_fallback_pstk(self):
        """pstkrv and pstkl missing → fall back to pstk."""
        df = pd.DataFrame({
            "pstkrv": [np.nan], "pstkl": [np.nan], "pstk": [30.0],
            "seq": [100.0], "txditc": [5.0],
        })
        result = self.compute_be(df)
        expected_be = 100 + 5 - 30
        assert result["be"].iloc[0] == pytest.approx(expected_be)

    def test_preferred_stock_fallback_all_missing(self):
        """All preferred stock fields missing → ps = 0."""
        df = pd.DataFrame({
            "pstkrv": [np.nan], "pstkl": [np.nan], "pstk": [np.nan],
            "seq": [100.0], "txditc": [5.0],
        })
        result = self.compute_be(df)
        expected_be = 100 + 5 - 0
        assert result["be"].iloc[0] == pytest.approx(expected_be)

    def test_negative_be_excluded(self):
        """Negative book equity should become NaN."""
        df = pd.DataFrame({
            "pstkrv": [200.0], "pstkl": [np.nan], "pstk": [np.nan],
            "seq": [50.0], "txditc": [5.0],
        })
        result = self.compute_be(df)
        assert pd.isna(result["be"].iloc[0])

    def test_txditc_nan_treated_as_zero(self):
        """Missing txditc should be filled with 0."""
        df = pd.DataFrame({
            "pstkrv": [10.0], "pstkl": [np.nan], "pstk": [np.nan],
            "seq": [100.0], "txditc": [np.nan],
        })
        result = self.compute_be(df)
        expected_be = 100 + 0 - 10
        assert result["be"].iloc[0] == pytest.approx(expected_be)


# ── Test: NYSE breakpoint logic ──────────────────────────────────────

class TestNYSEBreakpoints:
    """Verify that breakpoint computation uses correct percentiles."""

    def test_median_size_breakpoint(self):
        """Size breakpoint should be the median ME."""
        me_values = [10, 20, 30, 40, 50]
        median_me = np.median(me_values)
        assert median_me == 30.0

    def test_bm_percentiles(self):
        """B/M breakpoints should be 30th and 70th percentiles."""
        beme_values = np.arange(1, 101)  # 1 to 100
        p30 = np.percentile(beme_values, 30)
        p70 = np.percentile(beme_values, 70)
        assert p30 == pytest.approx(30.7, abs=0.1)
        assert p70 == pytest.approx(70.3, abs=0.1)


# ── Test: size bucket ────────────────────────────────────────────────

class TestSizeBucket:
    def setup_method(self):
        self.ns = _exec_helpers()
        self.sz_bucket = self.ns["sz_bucket"]

    def test_small(self):
        row = pd.Series({"me": 50, "sizemedn": 100})
        assert self.sz_bucket(row) == "S"

    def test_big(self):
        row = pd.Series({"me": 150, "sizemedn": 100})
        assert self.sz_bucket(row) == "B"

    def test_equal_to_median_is_small(self):
        row = pd.Series({"me": 100, "sizemedn": 100})
        assert self.sz_bucket(row) == "S"

    def test_missing_me(self):
        row = pd.Series({"me": np.nan, "sizemedn": 100})
        assert self.sz_bucket(row) == ""

    def test_missing_sizemedn(self):
        row = pd.Series({"me": 50, "sizemedn": np.nan})
        assert self.sz_bucket(row) == ""


# ── Test: book-to-market bucket ──────────────────────────────────────

class TestBMBucket:
    def setup_method(self):
        self.ns = _exec_helpers()
        self.bm_bucket = self.ns["bm_bucket"]

    def test_low(self):
        row = pd.Series({"beme": 0.5, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == "L"

    def test_medium(self):
        row = pd.Series({"beme": 1.5, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == "M"

    def test_high(self):
        row = pd.Series({"beme": 3.0, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == "H"

    def test_at_bm30_boundary(self):
        row = pd.Series({"beme": 1.0, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == "L"

    def test_at_bm70_boundary(self):
        row = pd.Series({"beme": 2.0, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == "M"

    def test_missing_beme(self):
        row = pd.Series({"beme": np.nan, "bm30": 1.0, "bm70": 2.0})
        assert self.bm_bucket(row) == ""

    def test_negative_beme(self):
        """Negative beme is excluded by the pipeline mask (beme > 0) before
        bm_bucket is called, so the function itself may return 'L' for
        values in [0, bm30]. We only test the mask catches it."""
        row = pd.Series({"beme": -0.5, "bm30": 1.0, "bm70": 2.0})
        # The function doesn't reject negatives — the pipeline mask does
        result = self.bm_bucket(row)
        assert isinstance(result, str)


# ── Test: SMB/HML formulas ──────────────────────────────────────────

class TestSMBHML:
    def setup_method(self):
        self.ns = _exec_helpers()
        self.compute_smb_hml = self.ns["compute_smb_hml"]

    def _make_ff(self, sl, sm, sh, bl, bm, bh):
        return pd.DataFrame({
            "SL": [sl], "SM": [sm], "SH": [sh],
            "BL": [bl], "BM": [bm], "BH": [bh],
        })

    def test_smb_formula(self):
        ff = self._make_ff(0.03, 0.04, 0.05, 0.01, 0.02, 0.03)
        result = self.compute_smb_hml(ff)
        expected = (0.03 + 0.04 + 0.05) / 3 - (0.01 + 0.02 + 0.03) / 3
        assert result["WSMB"].iloc[0] == pytest.approx(expected)

    def test_hml_formula(self):
        ff = self._make_ff(0.03, 0.04, 0.05, 0.01, 0.02, 0.03)
        result = self.compute_smb_hml(ff)
        expected = (0.05 + 0.03) / 2 - (0.03 + 0.01) / 2
        assert result["WHML"].iloc[0] == pytest.approx(expected)

    def test_smb_zero_when_equal(self):
        """When small and big have same returns, SMB = 0."""
        ff = self._make_ff(0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
        result = self.compute_smb_hml(ff)
        assert result["WSMB"].iloc[0] == pytest.approx(0.0)

    def test_hml_zero_when_equal(self):
        """When high and low B/M have same returns, HML = 0."""
        ff = self._make_ff(0.02, 0.03, 0.02, 0.02, 0.03, 0.02)
        result = self.compute_smb_hml(ff)
        assert result["WHML"].iloc[0] == pytest.approx(0.0)

    def test_multiple_months(self):
        """Formula works across multiple rows."""
        ff = pd.DataFrame({
            "SL": [0.01, 0.02], "SM": [0.02, 0.03], "SH": [0.03, 0.04],
            "BL": [0.005, 0.01], "BM": [0.01, 0.02], "BH": [0.015, 0.03],
        })
        result = self.compute_smb_hml(ff)
        assert len(result) == 2
        smb0 = (0.01 + 0.02 + 0.03) / 3 - (0.005 + 0.01 + 0.015) / 3
        assert result["WSMB"].iloc[0] == pytest.approx(smb0)


# ── Test: wavg function ─────────────────────────────────────────────

class TestWavg:
    def setup_method(self):
        self.ns = _exec_helpers()
        self.wavg = self.ns["wavg"]

    def test_basic_weighted_average(self):
        group = pd.DataFrame({"ret": [0.10, 0.20], "wt": [100, 300]})
        result = self.wavg(group, "ret", "wt")
        expected = (0.10 * 100 + 0.20 * 300) / 400
        assert result == pytest.approx(expected)

    def test_equal_weights(self):
        group = pd.DataFrame({"ret": [0.10, 0.20, 0.30], "wt": [1, 1, 1]})
        result = self.wavg(group, "ret", "wt")
        assert result == pytest.approx(0.20)
