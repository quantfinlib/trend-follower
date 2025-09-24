"""
Test suite for filter functions in trendfollower.core.filters.

This module tests the exponentially weighted moving average (EWMA) filters,
including standard EWMA, variance-preserving EWMA, and long-short variance-preserving EWMA.
"""

from itertools import combinations, product

import numpy as np
import polars as pl
import pytest 

from trendfollower.core.filters import (
    ewma, 
    variance_preserving_ewma, 
    long_short_variance_preserving_ewma
)


# Test Configuration Constants
RNG = np.random.default_rng(seed=42)
SMOOTHING_PARS = 0.1 + 0.1 * np.arange(8)  # [0.1, 0.2, ..., 0.8]
VARS = [0.1, 1.0, 2.0]  # Different variance levels for testing


def sample_long_series_with_input_variance(var: float) -> pl.Series:
    """Generate a sample long series with a specified input variance.

    Parameters
    ----------
    var : float
        The desired variance of the output series.

    Returns
    -------
    pl.Series
        A sample long series with the specified variance.
    """
    samples = RNG.normal(loc=0, scale=np.sqrt(var), size=100000)
    return pl.Series(samples)


@pytest.fixture(scope="module", autouse=True)
def sample_series_per_variance() -> dict[float, pl.Series]:
    """Generate sample series for each variance level."""
    return {var: sample_long_series_with_input_variance(var) for var in VARS}


class TestEWMAFilter:
    """Test cases for the standard EWMA filter."""

    @pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
    def test_mean_preservation(self, sample_series_per_variance, input_var, alpha):
        """Test that EWMA preserves the mean of the input series."""
        series = sample_series_per_variance[input_var]
        filtered = ewma(series, nu=alpha)
        expected = series.mean()
        calculated = filtered.mean()
        
        msg = f"EWMA should preserve mean: expected {expected}, got {calculated}"
        assert calculated == pytest.approx(expected, rel=1e-1), msg

    @pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
    def test_variance_reduction(self, sample_series_per_variance, input_var, alpha):
        """Test that EWMA reduces variance according to the theoretical formula."""
        series = sample_series_per_variance[input_var]
        filtered = ewma(series, nu=alpha)
        
        # Theoretical variance for EWMA: var * (1 - alpha) / (1 + alpha)
        expected = input_var * ((1 - alpha) / (1 + alpha))
        calculated = filtered.var()
        
        msg = (f"EWMA variance should be (1 - alpha) / (1 + alpha) * var: "
               f"expected {expected}, got {calculated}")
        assert calculated == pytest.approx(expected, rel=1e-1), msg


class TestVariancePreservingEWMA:
    """Test cases for the variance-preserving EWMA filter."""

    @pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
    def test_variance_preservation(self, sample_series_per_variance, input_var, alpha):
        """Test that variance-preserving EWMA maintains input variance."""
        series = sample_series_per_variance[input_var]
        filtered = variance_preserving_ewma(series, nu=alpha)
        
        expected = input_var
        calculated = filtered.var()
        
        msg = (f"Variance-preserving EWMA should maintain input variance: "
               f"expected {expected}, got {calculated}")
        assert calculated == pytest.approx(expected, rel=1e-1), msg


class TestLongShortVariancePreservingEWMA:
    """Test cases for the long-short variance-preserving EWMA filter."""

    @pytest.mark.parametrize("input_var, alphas", product(VARS, combinations(SMOOTHING_PARS, 2)))
    def test_variance_preservation(self, sample_series_per_variance, input_var, alphas):
        """Test that long-short variance-preserving EWMA maintains input variance."""
        series = sample_series_per_variance[input_var]
        filtered = long_short_variance_preserving_ewma(
            series, nu1=alphas[0], nu2=alphas[1]
        )
        
        expected = input_var
        calculated = filtered.var()
        
        msg = (f"Long-short variance-preserving EWMA should maintain input variance: "
               f"expected {expected}, got {calculated}")
        assert calculated == pytest.approx(expected, rel=1e-1), msg

    @pytest.mark.parametrize("alpha", SMOOTHING_PARS)
    def test_equal_alphas_error(self, sample_series_per_variance, alpha):
        """Test that equal alpha values raise a ValueError."""
        series = sample_series_per_variance[0.1]
        
        error_msg = ("nu1 and nu2 must be different. When they are equal, the long-short filter is ill-defined.")
        
        with pytest.raises(ValueError, match=error_msg):
            long_short_variance_preserving_ewma(series, nu1=alpha, nu2=alpha)

    