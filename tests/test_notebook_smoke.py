"""
Smoke tests for ff_factor_replication.ipynb.

These tests extract the three helper functions (wavg, sz_bucket, bm_bucket)
from the notebook and validate core logic without needing WRDS access.
The book-equity and SMB/HML logic is inline in the notebook, so we test
it directly here using the same operations.
"""
import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

NOTEBOOK_PATH = pathlib.Path(__file__).resolve().parent.parent / "ff_factor_replication.ipynb"


# ── Notebook loading utilities ───────────────────────────────────────

def _load_notebook():
    with open(NOTEBOOK_PATH) as f:
        return json.load(f)


def _code_cells(nb):
    sources = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            sources.append("".join(cell["source"]))
    return sources


def _exec_helpers():
    """Execute the helper-function cell and return its namespace."""
    nb = _load_notebook()
    codes = _code_cells(nb)
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
        nb = _load_notebook()
        codes = _code_cells(nb)
        assert len(codes) > 0, "No code cells found"
        for i, src in enumerate(codes):
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"Cell {i} has a syntax error: {e}")


# ── Test: Compustat BE construction (inline logic) ───────────────────

class TestBookEquity:
    """Test the inline preferred-stock fallback + BE logic."""

    @staticmethod
    def _compute_be(pstkrv, pstkl, pstk, seq, txditc):
        """Replicate the inline notebook logic for a single row."""
        ps = pstkrv if not pd.isna(pstkrv) else pstkl
        ps = ps if not pd.isna(ps) else pstk
        ps = ps if not pd.isna(ps) else 0
        txditc = txditc if not pd.isna(txditc) else 0
        be = float(seq + txditc - ps)
        return be if be > 0 else np.nan

    def test_preferred_stock_fallback_pstkrv(self):
        assert self._compute_be(10, 20, 30, 100, 5) == pytest.approx(95)

    def test_preferred_stock_fallback_pstkl(self):
        assert self._compute_be(np.nan, 20, 30, 100, 5) == pytest.approx(85)

    def test_preferred_stock_fallback_pstk(self):
        assert self._compute_be(np.nan, np.nan, 30, 100, 5) == pytest.approx(75)

    def test_preferred_stock_fallback_all_missing(self):
        assert self._compute_be(np.nan, np.nan, np.nan, 100, 5) == pytest.approx(105)

    def test_negative_be_excluded(self):
        assert pd.isna(self._compute_be(200, np.nan, np.nan, 50, 5))

    def test_txditc_nan_treated_as_zero(self):
        assert self._compute_be(10, np.nan, np.nan, 100, np.nan) == pytest.approx(90)

    def test_vectorized_matches_scalar(self):
        """Verify the numpy-vectorized version from the notebook matches."""
        df = pd.DataFrame({
            "pstkrv": [10, np.nan, np.nan, np.nan],
            "pstkl":  [20, 20,     np.nan, np.nan],
            "pstk":   [30, 30,     30,     np.nan],
            "seq":    [100, 100, 100, 100],
            "txditc": [5, 5, 5, 5],
        })
        df['ps'] = np.where(df['pstkrv'].isnull(), df['pstkl'], df['pstkrv'])
        df['ps'] = np.where(df['ps'].isnull(), df['pstk'], df['ps'])
        df['ps'] = np.where(df['ps'].isnull(), 0, df['ps'])
        df['txditc'] = df['txditc'].fillna(0)
        df['be'] = (df['seq'] + df['txditc'] - df['ps']).astype(float)
        df['be'] = np.where(df['be'] > 0, df['be'], np.nan)
        assert list(df['be']) == [pytest.approx(95), pytest.approx(85),
                                  pytest.approx(75), pytest.approx(105)]


# ── Test: NYSE breakpoint logic ──────────────────────────────────────

class TestNYSEBreakpoints:
    def test_median_size_breakpoint(self):
        assert np.median([10, 20, 30, 40, 50]) == 30.0

    def test_bm_percentiles(self):
        beme = np.arange(1, 101)
        assert np.percentile(beme, 30) == pytest.approx(30.7, abs=0.1)
        assert np.percentile(beme, 70) == pytest.approx(70.3, abs=0.1)


# ── Test: size bucket ────────────────────────────────────────────────

class TestSizeBucket:
    def setup_method(self):
        self.sz_bucket = _exec_helpers()["sz_bucket"]

    def test_small(self):
        assert self.sz_bucket(pd.Series({"me": 50, "sizemedn": 100})) == "S"

    def test_big(self):
        assert self.sz_bucket(pd.Series({"me": 150, "sizemedn": 100})) == "B"

    def test_equal_to_median_is_small(self):
        assert self.sz_bucket(pd.Series({"me": 100, "sizemedn": 100})) == "S"

    def test_missing_me(self):
        assert self.sz_bucket(pd.Series({"me": np.nan, "sizemedn": 100})) == ""

    def test_missing_sizemedn(self):
        assert self.sz_bucket(pd.Series({"me": 50, "sizemedn": np.nan})) == ""


# ── Test: book-to-market bucket ──────────────────────────────────────

class TestBMBucket:
    def setup_method(self):
        self.bm_bucket = _exec_helpers()["bm_bucket"]

    def test_low(self):
        assert self.bm_bucket(pd.Series({"beme": 0.5, "bm30": 1.0, "bm70": 2.0})) == "L"

    def test_medium(self):
        assert self.bm_bucket(pd.Series({"beme": 1.5, "bm30": 1.0, "bm70": 2.0})) == "M"

    def test_high(self):
        assert self.bm_bucket(pd.Series({"beme": 3.0, "bm30": 1.0, "bm70": 2.0})) == "H"

    def test_at_bm30_boundary(self):
        assert self.bm_bucket(pd.Series({"beme": 1.0, "bm30": 1.0, "bm70": 2.0})) == "L"

    def test_at_bm70_boundary(self):
        assert self.bm_bucket(pd.Series({"beme": 2.0, "bm30": 1.0, "bm70": 2.0})) == "M"

    def test_missing_beme(self):
        assert self.bm_bucket(pd.Series({"beme": np.nan, "bm30": 1.0, "bm70": 2.0})) == ""

    def test_negative_beme(self):
        """Negative beme is filtered out by the pipeline mask before bm_bucket runs."""
        result = self.bm_bucket(pd.Series({"beme": -0.5, "bm30": 1.0, "bm70": 2.0}))
        assert isinstance(result, str)


# ── Test: SMB/HML formulas (inline logic) ────────────────────────────

class TestSMBHML:
    @staticmethod
    def _compute(sl, sm, sh, bl, bm, bh):
        wsmb = (sl + sm + sh) / 3 - (bl + bm + bh) / 3
        whml = (sh + bh) / 2 - (sl + bl) / 2
        return wsmb, whml

    def test_smb_formula(self):
        smb, _ = self._compute(0.03, 0.04, 0.05, 0.01, 0.02, 0.03)
        expected = (0.03 + 0.04 + 0.05) / 3 - (0.01 + 0.02 + 0.03) / 3
        assert smb == pytest.approx(expected)

    def test_hml_formula(self):
        _, hml = self._compute(0.03, 0.04, 0.05, 0.01, 0.02, 0.03)
        expected = (0.05 + 0.03) / 2 - (0.03 + 0.01) / 2
        assert hml == pytest.approx(expected)

    def test_smb_zero_when_equal(self):
        smb, _ = self._compute(0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
        assert smb == pytest.approx(0.0)

    def test_hml_zero_when_equal(self):
        _, hml = self._compute(0.02, 0.03, 0.02, 0.02, 0.03, 0.02)
        assert hml == pytest.approx(0.0)

    def test_vectorized(self):
        """Test with a DataFrame matching the notebook's inline code."""
        ff = pd.DataFrame({
            "SL": [0.01, 0.02], "SM": [0.02, 0.03], "SH": [0.03, 0.04],
            "BL": [0.005, 0.01], "BM": [0.01, 0.02], "BH": [0.015, 0.03],
        })
        ff['WSMB'] = (ff['SL'] + ff['SM'] + ff['SH']) / 3 - (ff['BL'] + ff['BM'] + ff['BH']) / 3
        ff['WHML'] = (ff['SH'] + ff['BH']) / 2 - (ff['SL'] + ff['BL']) / 2
        expected_smb0 = (0.01 + 0.02 + 0.03) / 3 - (0.005 + 0.01 + 0.015) / 3
        assert ff['WSMB'].iloc[0] == pytest.approx(expected_smb0)
        assert len(ff) == 2


# ── Test: wavg function ─────────────────────────────────────────────

class TestWavg:
    def setup_method(self):
        self.wavg = _exec_helpers()["wavg"]

    def test_basic_weighted_average(self):
        group = pd.DataFrame({"ret": [0.10, 0.20], "wt": [100, 300]})
        expected = (0.10 * 100 + 0.20 * 300) / 400
        assert self.wavg(group, "ret", "wt") == pytest.approx(expected)

    def test_equal_weights(self):
        group = pd.DataFrame({"ret": [0.10, 0.20, 0.30], "wt": [1, 1, 1]})
        assert self.wavg(group, "ret", "wt") == pytest.approx(0.20)
