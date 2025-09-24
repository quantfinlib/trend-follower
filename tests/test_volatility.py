"""Tests suite for volatility functions in trendfollower.core.volatility."""


import numpy as np
import polars as pl
import pytest 

from trendfollower.core.volatility import (
    lag1_diff, 
    relative_return, 
    true_range, 
    relative_true_range,
    ma_true_range,
    ma_relative_true_range,
    ma_true_range_from_hlc,
    ma_relative_true_range_from_hlc,
    ewma_relative_true_range,
    ewma_relative_true_range_from_hlc
)

from .test_filters import SMOOTHING_PARS, VARS, sample_long_series_with_input_variance


# Test Configuration Constants
RNG = np.random.default_rng(seed=42)
PERIODS = [2, 10, 100]  # Different periods for moving average tests


def hlc_per_variance(var: float) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Generate high, low, close price series with specified variance.
    
    Parameters
    ----------
    var : float
        The desired variance of the close price series.
        
    Returns
    -------
    tuple[pl.Series, pl.Series, pl.Series]
        High, low, and close price series.
    """
    close = sample_long_series_with_input_variance(var)
    high = pl.Series(RNG.uniform(1.01, 1.02, size=close.shape)) * close
    low = pl.Series(RNG.uniform(0.95, 0.96, size=close.shape)) * close
    return high, low, close


class TestBasicVolatilityMeasures:
    """Test cases for basic volatility measures: lag1_diff and relative_return."""

    def test_lag1_diff_mean(self):
        """Test that lag1_diff has zero mean for random data."""
        _, _, close = hlc_per_variance(0.1)
        result = lag1_diff(close).mean()
        assert result == pytest.approx(0, abs=1e-2)

    @pytest.mark.parametrize("var", VARS)
    def test_lag1_diff_std(self, var):
        """Test that lag1_diff has correct standard deviation."""
        _, _, close = hlc_per_variance(var)
        result = lag1_diff(close).std()
        expected = np.sqrt(2 * var)
        assert result == pytest.approx(expected, abs=1e-2)

    def test_relative_return_logic(self):
        """Test the mathematical logic of relative_return calculation."""
        s = pl.Series(np.arange(1, 1000))
        actual = relative_return(s).to_numpy()[1:]
        expected = 1. / np.linspace(1, 998, 998)
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)


class TestTrueRange:
    """Test cases for true range calculations."""

    def test_true_range_properties(self):
        """Test that true range satisfies certain mathematical properties."""
        high, low, close = hlc_per_variance(0.1)
        actual = true_range(high, low, close).to_numpy()
        
        s1 = (high - low).abs().to_numpy()
        s2 = (high - close.shift(1)).abs().to_numpy()
        s3 = (low - close.shift(1)).abs().to_numpy()

        # True range should be non-negative
        assert np.all(actual >= 0), "True range should be non-negative"
        
        # True range should be at least as large as each component
        assert np.all(actual >= s1), "True range should be at least as large as high - low"
        assert np.all(actual[1:] >= s2[1:]), (
            "True range should be at least as large as |high - prev_close|"
        )
        assert np.all(actual[1:] >= s3[1:]), (
            "True range should be at least as large as |low - prev_close|"
        )

    def test_relative_true_range_calculation(self):
        """Test that relative true range is correctly calculated."""
        high, low, close = hlc_per_variance(0.1)
        rtr = relative_true_range(high, low, close)
        tr = true_range(high, low, close)
        close_shift = close.shift(1)
        expected = tr / close_shift
        
        np.testing.assert_array_almost_equal(
            rtr.to_numpy(), expected.to_numpy(), decimal=3
        )


class TestMovingAverageTrueRange:
    """Test cases for moving average true range calculations."""

    @pytest.mark.parametrize("period", PERIODS)
    def test_ma_true_range_finite_values(self, period):
        """Test that moving average true range has the correct number of finite values."""
        high, low, close = hlc_per_variance(0.1)
        
        # Test moving average true range from true range directly
        tr = true_range(high, low, close)
        ma_tr = ma_true_range(tr, period=period).to_numpy()
        expected_finite = len(ma_tr) - period + 1
        actual_finite = np.isfinite(ma_tr).sum()
        
        assert actual_finite == expected_finite, (
            f"MA True Range must have {expected_finite} finite values, got {actual_finite}"
        )
        
        # Verify last values are computed correctly
        assert ma_tr[-1] == pytest.approx(np.mean(tr.to_numpy()[-period:]), rel=1e-10)
        assert ma_tr[-2] == pytest.approx(np.mean(tr.to_numpy()[-(period+1):-1]), rel=1e-10)

    @pytest.mark.parametrize("period", PERIODS)
    def test_ma_true_range_from_hlc(self, period):
        """Test moving average true range calculated directly from HLC data."""
        high, low, close = hlc_per_variance(0.1)
        
        ma_tr = ma_true_range_from_hlc(high, low, close, period=period).to_numpy()
        expected_finite = len(ma_tr) - period + 1
        actual_finite = np.isfinite(ma_tr).sum()
        
        assert actual_finite == expected_finite, (
            f"MA True Range from HLC must have {expected_finite} finite values, got {actual_finite}"
        )

    @pytest.mark.parametrize("period", PERIODS)
    def test_ma_relative_true_range(self, period):
        """Test moving average relative true range calculations."""
        high, low, close = hlc_per_variance(0.1)
        
        # Test from relative true range directly
        rtr = relative_true_range(high, low, close)
        ma_rtr = ma_relative_true_range(rtr, period=period).to_numpy()
        expected_finite = len(ma_rtr) - period
        actual_finite = np.isfinite(ma_rtr).sum()
        
        assert actual_finite == expected_finite, (
            f"MA Relative True Range must have {expected_finite} finite values, got {actual_finite}"
        )
        
        # Test from HLC directly
        ma_rtr_hlc = ma_relative_true_range_from_hlc(high, low, close, period=period).to_numpy()
        actual_finite_hlc = np.isfinite(ma_rtr_hlc).sum()
        
        assert actual_finite_hlc == expected_finite, (
            f"MA Relative True Range from HLC must have {expected_finite} finite values, got {actual_finite_hlc}"
        )


class TestEWMATrueRange:
    """Test cases for EWMA true range calculations."""

    @pytest.mark.parametrize("alpha", SMOOTHING_PARS)
    def test_ewma_relative_true_range_properties(self, alpha):
        """Test EWMA of relative true range has correct variance properties."""
        high, low, close = hlc_per_variance(0.1)
        rtr = relative_true_range(high=high, low=low, close=close)
        var_rtr = rtr.var()
        
        # Test EWMA RTR directly from RTR
        ewma_rtr = ewma_relative_true_range(rtr=rtr, alpha=alpha)
        finite_count = np.isfinite(ewma_rtr.to_numpy()).sum()
        expected_finite = len(ewma_rtr) - 1
        
        assert finite_count == expected_finite, (
            f"All elements of EWMA RTR, except the first, must be finite. "
            f"Expected {expected_finite}, got {finite_count}"
        )
        
        # Check variance reduction follows EWMA theory
        var_ewma_rtr = ewma_rtr.var()
        expected_var_ewma_rtr = (1 - alpha) / (1 + alpha) * var_rtr
        
        assert var_ewma_rtr == pytest.approx(expected_var_ewma_rtr, rel=1e-2), (
            f"Variance of EWMA RTR must be approximately "
            f"{(1 - alpha) / (1 + alpha)} times the variance of RTR"
        )

    @pytest.mark.parametrize("alpha", SMOOTHING_PARS)
    def test_ewma_relative_true_range_from_hlc(self, alpha):
        """Test EWMA relative true range calculated directly from HLC data."""
        high, low, close = hlc_per_variance(0.1)
        rtr = relative_true_range(high=high, low=low, close=close)
        var_rtr = rtr.var()
        
        # Test EWMA RTR from HLC
        ewma_rtr_hlc = ewma_relative_true_range_from_hlc(
            high=high, low=low, close=close, alpha=alpha
        )
        finite_count = np.isfinite(ewma_rtr_hlc.to_numpy()).sum()
        expected_finite = len(ewma_rtr_hlc) - 1
        
        assert finite_count == expected_finite, (
            f"All elements of EWMA RTR from HLC, except the first, must be finite. "
            f"Expected {expected_finite}, got {finite_count}"
        )
        
        # Check variance reduction follows EWMA theory
        var_ewma_rtr_hlc = ewma_rtr_hlc.var()
        expected_var_ewma_rtr = (1 - alpha) / (1 + alpha) * var_rtr
        
        assert var_ewma_rtr_hlc == pytest.approx(expected_var_ewma_rtr, rel=1e-2), (
            f"Variance of EWMA RTR from HLC must be approximately "
            f"{(1 - alpha) / (1 + alpha)} times the variance of RTR"
        )