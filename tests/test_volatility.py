from itertools import product
from matplotlib import axis
import numpy as np
import polars as pl
import pytest 

from trendfollower.volatility import (
    lag1_diff, 
    relative_return, 
    sigma_t_price, 
    sigma_t_return, 
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

RNG = np.random.default_rng(seed=42)
PERIODS = [2, 10, 100]

def hlc_per_variance(var):
    close = sample_long_series_with_input_variance(var)
    high = pl.Series(RNG.uniform(1.01, 1.02, size=close.shape)) * close
    low = pl.Series(RNG.uniform(0.95, 0.96, size=close.shape)) * close
    return high, low, close


def test_lag1_diff_mean():
    _,_,close = hlc_per_variance(.1)
    assert lag1_diff(close).mean() == pytest.approx(0, abs=1e-2)


@pytest.mark.parametrize("var", VARS)
def test_lag1_diff_std(var):
    _,_,close = hlc_per_variance(var)
    assert lag1_diff(close).std() == pytest.approx(np.sqrt(2 * var), abs=1e-2)


def test_logic_relative_return():
    s = pl.Series(np.arange(1, 1000))
    actual = relative_return(s).to_numpy()[1:]
    desired = 1./np.linspace(1, 998, 998)
    np.testing.assert_array_almost_equal(actual, desired, decimal=3)


def test_true_range():
    high, low, close = hlc_per_variance(0.1)
    actual = true_range(high, low, close).to_numpy()
    s1 = (high - low).abs().to_numpy()
    s2 = (high - close.shift(1)).abs().to_numpy()
    s3 = (low - close.shift(1)).abs().to_numpy()

    assert np.all(actual >= 0), "True range should be non-negative" 
    assert np.all(actual >= s1), "True range should be at least as large as high - low"
    assert np.all(actual[1:] >= s2[1:]), "True range should be at least as large as the absolute value of high - close.shift(1)"
    assert np.all(actual[1:] >= s3[1:]), "True range should be at least as large as the absolute value of low - close.shift(1)"


def test_relative_true_range():
    high, low, close = hlc_per_variance(0.1)
    rtr = relative_true_range(high, low, close)
    tr = true_range(high, low, close)
    close_shift = close.shift(1)
    expected = tr / close_shift
    np.testing.assert_array_almost_equal(rtr.to_numpy(), expected.to_numpy(), decimal=3)
    

@pytest.mark.parametrize("period", PERIODS)
def test_ma_true_range(period):
    high, low, close = hlc_per_variance(0.1)
    # moving average true range from true range directly
    tr = true_range(high, low, close)
    ma_tr = ma_true_range(tr, period=period).to_numpy()
    assert np.isfinite(ma_tr).sum() == len(ma_tr) - period + 1, f"MA True Range must have {len(ma_tr) - period + 1} finite values"
    ma_tr[-1] = np.mean(tr.to_numpy()[-period:])
    ma_tr[-2] = np.mean(tr.to_numpy()[-(period+1):-1])
    # moving average true range from HLC
    ma_tr = ma_true_range_from_hlc(high, low, close, period=period).to_numpy()
    assert np.isfinite(ma_tr).sum() == len(ma_tr) - period + 1, f"MA True Range must have {len(ma_tr) - period + 1} finite values"
    ma_tr[-1] = np.mean(ma_tr[-period:])
    ma_tr[-2] = np.mean(ma_tr[-(period+1):-1])
    # moving relative true range from relative true range directly
    rtr = relative_true_range(high, low, close)
    ma_rtr = ma_relative_true_range(rtr, period=period).to_numpy()
    assert np.isfinite(ma_rtr).sum() == len(ma_rtr) - period, f"MA Relative True Range must have {len(ma_rtr) - period} finite values"
    ma_rtr[-1] = np.mean(ma_rtr[-period:])
    ma_rtr[-2] = np.mean(ma_rtr[-(period+1):-1])
    # moving relative true range from HLC
    ma_rtr = ma_relative_true_range_from_hlc(high, low, close, period=period).to_numpy()
    assert np.isfinite(ma_rtr).sum() == len(ma_rtr) - period, f"MA Relative True Range must have {len(ma_rtr) - period} finite values"
    ma_rtr[-1] = np.mean(ma_rtr[-period:])
    ma_rtr[-2] = np.mean(ma_rtr[-(period+1):-1])


@pytest.mark.parametrize("alpha", SMOOTHING_PARS)
def test_ewma_relative_true_range(alpha):
    high, low, close = hlc_per_variance(0.1)
    rtr = relative_true_range(high=high, low=low, close=close)
    var_rtr = rtr.var()
    # ewma rtr directly from rtr
    ewma_rtr = ewma_relative_true_range(rtr=rtr, alpha=alpha)
    assert np.isfinite(ewma_rtr.to_numpy()).sum() == len(ewma_rtr) - 1, "All elements of EWMA Relative True Range, except the first element, must be finite"
    var_ewma_rtr = ewma_rtr.var()
    expected_var_ewma_rtr = (1 - alpha) / (1 + alpha) * var_rtr
    assert var_ewma_rtr == pytest.approx(expected_var_ewma_rtr, rel=1e-2), f"Variance of EWMA Relative True Range must be approximately {(1 - alpha) / (1 + alpha)} times the variance of Relative True Range"
    # ewma rtr from HLC
    ewma_rtr_hlc = ewma_relative_true_range_from_hlc(high=high, low=low, close=close, alpha=alpha)
    assert np.isfinite(ewma_rtr_hlc.to_numpy()).sum() == len(ewma_rtr_hlc) - 1, "All elements of EWMA Relative True Range from HLC, except the first element, must be finite"
    var_ewma_rtr_hlc = ewma_rtr_hlc.var()
    assert var_ewma_rtr_hlc == pytest.approx(expected_var_ewma_rtr, rel=1e-2), f"Variance of EWMA Relative True Range from HLC must be approximately {(1 - alpha) / (1 + alpha)} times the variance of Relative True Range"