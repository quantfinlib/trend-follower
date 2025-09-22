from itertools import combinations, product

import numpy as np
import polars as pl
import pytest 

from trendfollower.filters import ewma, variance_preserving_ewma, long_short_variance_preserving_ewma


RNG = np.random.default_rng(seed=42)
SMOOTHING_PARS = 0.1 + 0.1*np.arange(8)
VARS = [0.1, 1.0, 2.0]


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
    return {var: sample_long_series_with_input_variance(var) for var in VARS}


@pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
def test_ewma_filter_mean(sample_series_per_variance, input_var, alpha):
    series = sample_series_per_variance[input_var]
    filtered = ewma(series, alpha=alpha)
    expected = series.mean()
    calculated = filtered.mean()
    msg = f"Under ewma transformation, mean remains unchanged, expected {expected}, got {calculated}"
    assert calculated == pytest.approx(expected, rel=1e-1), msg


@pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
def test_ewma_filter_variance(sample_series_per_variance, input_var, alpha):
    series = sample_series_per_variance[input_var]
    filtered = ewma(series, alpha=alpha)
    expected = input_var * ((1 - alpha) / (1 + alpha))
    calculated = filtered.var()
    msg = f"Variance of ewma of a series with input_variance var must be (1 - alpha) / (1 + alpha) * var, expected {expected}, got {calculated}"
    assert calculated == pytest.approx(expected, rel=1e-1), msg


@pytest.mark.parametrize("input_var, alpha", product(VARS, SMOOTHING_PARS))
def test_variance_preserving_ewma(sample_series_per_variance, input_var, alpha):
    series = sample_series_per_variance[input_var]
    filtered = variance_preserving_ewma(series, alpha=alpha)
    expected = input_var
    calculated = filtered.var()
    msg = f"Variance of variance_preserving_ewma of a series with input_variance var must be var, expected {expected}, got {calculated}"
    assert calculated == pytest.approx(expected, rel=1e-1), msg


@pytest.mark.parametrize("input_var, alphas", product(VARS, combinations(SMOOTHING_PARS, 2)))
def test_variance_long_short_variance_preserving_ewma(sample_series_per_variance, input_var, alphas):
    series = sample_series_per_variance[input_var]
    filtered = long_short_variance_preserving_ewma(series, alpha1=alphas[0], alpha2=alphas[1])
    expected = input_var
    calculated = filtered.var()
    msg = f"Variance of long_short_variance_preserving_ewma of a series with input_variance var must be var, expected {expected}, got {calculated}"
    assert calculated == pytest.approx(expected, rel=1e-1), msg


@pytest.mark.parametrize("alpha", SMOOTHING_PARS)
def test_trivial_long_short_variance_preserving_ewma(sample_series_per_variance, alpha):
    series = sample_series_per_variance[0.1]
    with pytest.raises(ValueError, match="alpha1 and alpha2 must be different. When they are equal, the long-short filter is ill-defined."):
        long_short_variance_preserving_ewma(series, alpha1=alpha, alpha2=alpha)

    