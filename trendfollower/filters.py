# Copyright (c) 2025, Mohammadjavad Vakili
# All rights reserved.
# This code is licensed under the MIT License.

"""Filter functions for time series data.

Implemented functions:

`ewma`

Exponentially-weighted moving average filter.

`variance_preserving_ewma`

Variance preserving exponential moving average filter.

`long_short_variance_preserving_ewma`

Long-short variance preserving exponential moving average filter.
"""

import polars as pl


def ewma(z: pl.Series, alpha: float) -> pl.Series:
    """Calculate exponential moving average filter of a series.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    alpha : float
        The smoothing factor (0 < alpha < 1).

    Returns
    -------
    pl.Series
        The exponentially weighted moving average of the input series.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html

    """
    return z.ewm_mean(alpha=1 - alpha)


def variance_preserving_ewma(z: pl.Series, alpha: float) -> pl.Series:
    """Calculate variance preserving exponential moving average filter of a series.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    alpha : float
        The smoothing factor (0 < alpha < 1).

    Returns
    -------
    pl.Series
        The variance preserving exponentially weighted moving average of the input series.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html

    """
    return ewma(z=z, alpha=alpha) * ((1 + alpha) / (1 - alpha))**.5


def long_short_variance_preserving_ewma(z: pl.Series, alpha1: float, alpha2: float) -> pl.Series:
    """Calculate long-short variance preserving exponential moving average filter of a series.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    alpha1 : float
        The smoothing factor for the long position (0 < alpha1 < 1).
    alpha2 : float
        The smoothing factor for the short position (0 < alpha2 < 1).

    Returns
    -------
    pl.Series
        The long-short variance preserving exponentially weighted moving average of the input series.

    Raises
    ------
    ValueError
        If alpha1 and alpha2 are equal.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html
    In Eq. 9 of paper, there is a typo: q should be 1 / q.

    """
    if alpha1 == alpha2:
        msg = "alpha1 and alpha2 must be different. When they are equal, the long-short filter is ill-defined."
        raise ValueError(msg)
    q = (1 / (1 - alpha1**2.) + 1 / (1 - alpha2**2.) - 2 / (1 - alpha1 * alpha2)) ** -.5
    l1 = q / (1 - alpha1)
    l2 = q / (1 - alpha2)
    return l1 * ewma(z=z, alpha=alpha1) - l2 * ewma(z=z, alpha=alpha2)
