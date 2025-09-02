# Copyright (c) 2025, Mohammadjavad Vakili
# All rights reserved.
#
# This code is licensed under the MIT License.
"""Implementation of return and volatility functions.

Implemented functions:

- `lag1_diff`
- `relative_return`
- `sigma_t_price`
- `sigma_t_return`
- `true_range`
- `average_true_range_from_trt`
- `average_true_range`

"""

import polars as pl

from trendfollower.filters import ewma


def lag1_diff(z: pl.Series) -> pl.Series:
    r"""Calculate the lag-1 difference of a series.

    $$\text{lag-1 diff}(z_t) = d_{t} = z_t - z_{t-1}$$

    Parameters
    ----------
    z : pl.Series
        The input time series.

    Returns
    -------
    pl.Series
        The lag-1 difference of the input series.

    References
    ----------
    .. [1] Eq. (1) Science & Practice of trend-following systems.

    """
    return z - z.shift(1)


def relative_return(z: pl.Series) -> pl.Series:
    r"""Calculate the relative return of a series.

    $$ r_t = \frac{z_t - z_{t-1}}{z_{t-1}} $$

    Parameters
    ----------
    z : pl.Series
        The input time series.

    Returns
    -------
    pl.Series
        The relative return of the input series.

    References
    ----------
    .. [1] Eq. (2) Science & Practice of trend-following systems.

    """
    return z / z.shift(1) - 1


def sigma_t_price(z: pl.Series, alpha: float) -> pl.Series:
    r"""Calculate the volatility of a series using the EWMA method.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    alpha : float
        The smoothing factor (0 < alpha < 1).

    Returns
    -------
    pl.Series
        The volatility of the input series.

    References
    ----------
    .. [1] Eq. (11) Science & Practice of trend-following systems.

    """
    dt_squared = lag1_diff(z) ** 2
    return ewma(dt_squared, alpha) ** 0.5


def sigma_t_return(z: pl.Series, alpha: float) -> pl.Series:
    r"""Calculate the volatility of a series using the EWMA method.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    alpha : float
        The smoothing factor (0 < alpha < 1).

    Returns
    -------
    pl.Series
        The volatility of the input series.

    References
    ----------
    .. [1] Eq. (12) Science & Practice of trend-following systems.

    """
    rt_squared = relative_return(z) ** 2
    return ewma(rt_squared, alpha) ** 0.5


def true_range(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """Compute the true range from high, low, and close series.

    Parameters
    ----------
    high : pl.Series
        The high prices.
    low : pl.Series
        The low prices.
    close : pl.Series
        The close prices.

    Returns
    -------
    pl.Series
        The true range of the input series.

    References
    ----------
    .. [1] Eq. (13) Science & Practice of trend-following systems.

    """
    high_low_t = high - low
    close_t_1 = close.shift(1)
    high_close_t = high - close_t_1
    low_close_t = low - close_t_1
    df = pl.DataFrame({
        "high_low": high_low_t.abs(),
        "high_close": high_close_t.abs(),
        "low_close": low_close_t.abs(),
    })
    return df.select(pl.max_horizontal(["high_low", "high_close", "low_close"])).to_series()


def relative_true_range(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """Compute the relative true range from high, low, and close series.

    Parameters
    ----------
    high : pl.Series
        The high prices.
    low : pl.Series
        The low prices.
    close : pl.Series
        The close prices.

    Returns
    -------
    pl.Series
        The relative true range of the input series.

    References
    ----------
    .. [1] Eq. (15) Science & Practice of trend-following systems.

    """
    tr = true_range(high, low, close)
    return tr / close.shift(1)


def ma_true_range(trt: pl.Series, period: int) -> pl.Series:
    """Compute the average true range from the true range time series over a specified period.

    Parameters
    ----------
    trt : pl.Series
        The true range time series.
    period : int
        The rolling window period.

    Returns
    -------
    pl.Series
        The average true range of the input series.

    """
    return trt.rolling_mean(window_size=period)


def ma_relative_true_range(rtr: pl.Series, period: int) -> pl.Series:
    """Compute the average relative true range from the relative true range time series over a specified period.

    Parameters
    ----------
    rtr : pl.Series
        The relative true range time series.
    period : int
        The rolling window period.

    Returns
    -------
    pl.Series
        The average relative true range of the input series.

    """
    return rtr.rolling_mean(window_size=period)


def ma_true_range_from_hlc(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """Compute rolling average of true range over a specified period from high, low, and close prices.

    Parameters
    ----------
    high : pl.Series
        The high prices.
    low : pl.Series
        The low prices.
    close : pl.Series
        The close prices.
    period : int
        The rolling window period.

    Returns
    -------
    pl.Series
        The average true range of the input series.

    """
    tr = true_range(high, low, close)
    return ma_true_range(tr, period)


def ma_relative_true_range_from_hlc(high: pl.Series, low: pl.Series, close: pl.Series, period: int) -> pl.Series:
    """Compute rolling average of relative true range over a specified period from high, low, and close prices.

    Parameters
    ----------
    high : pl.Series
        The high prices.
    low : pl.Series
        The low prices.
    close : pl.Series
        The close prices.
    period : int
        The rolling window period.

    Returns
    -------
    pl.Series
        The average relative true range of the input series.

    """
    rtr = relative_true_range(high, low, close)
    return ma_relative_true_range(rtr, period)


def ewma_relative_true_range(rtr: pl.Series, alpha: float) -> pl.Series:
    """Compute the exponentially weighted moving average of the relative true range.

    Parameters
    ----------
    rtr : pl.Series
        The relative true range time series.
    alpha : float
        The smoothing factor (between 0 and 1).

    Returns
    -------
    pl.Series
        The exponentially weighted moving average of the relative true range.

    References
    ----------
    .. [1] Eq. (14) Science & Practice of trend-following systems.

    """
    return ewma(rtr, alpha)


def ewma_relative_true_range_from_hlc(high: pl.Series, low: pl.Series, close: pl.Series, alpha: float) -> pl.Series:
    """Compute the exponentially weighted moving average of the relative true range from high, low, and close prices.

    Parameters
    ----------
    high : pl.Series
        The high prices.
    low : pl.Series
        The low prices.
    close : pl.Series
        The close prices.
    alpha : float
        The smoothing factor (between 0 and 1).

    Returns
    -------
    pl.Series
        The exponentially weighted moving average of the relative true range.

    References
    ----------
    .. [1] Eq. (14) Science & Practice of trend-following systems.

    """
    rtr = relative_true_range(high, low, close)
    return ewma(rtr, alpha)
