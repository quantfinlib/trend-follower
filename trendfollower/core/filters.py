# Copyright (c) 2025, Mohammadjavad Vakili
# All rights reserved.
# This code is licensed under the MIT License.

"""Implementation of exponential weighted moving average filters.

Implemented functions:

`ewma`

Exponentially-weighted moving average filter.

`variance_preserving_ewma`

Variance preserving exponential moving average filter.

`long_short_variance_preserving_ewma`

Long-short variance preserving exponential moving average filter.
"""

import polars as pl


def _validate_nu_or_span(nu: float | None, span: int | None) -> float:
    """Validate and convert nu/span parameters to nu.

    Parameters
    ----------
    nu : float | None
        The smoothing parameter (0 < nu < 1).
    span : int | None
        The span of the moving average (positive integer).

    Returns
    -------
    float
        The validated nu parameter.

    Raises
    ------
    ValueError
        If neither nu nor span is provided, or if parameters are invalid.

    """
    if nu is None and span is None:
        msg = "Either 'nu' or 'span' must be provided."
        raise ValueError(msg)
    if nu is not None and not (0 < nu < 1):
        msg = f"Invalid nu={nu}: must be between 0 and 1 (exclusive)."
        raise ValueError(msg)
    if span is not None and (span <= 0 or span != int(span)):
        msg = f"Invalid span={span}: must be a positive integer."
        raise ValueError(msg)
    return nu if nu is not None else 1 - 2 / (span + 1)


def ewma(z: pl.Series, nu: float | None = None, span: int | None = None) -> pl.Series:
    r"""Calculate exponential moving average filter of a series.

    The smoothing parameter `nu` and the span are related by:

    .. math::

        \nu = 1 - \frac{2}{\text{span} + 1}

    If both `nu` and `span` are provided, `nu` takes precedence.
    If neither is provided, a ValueError is raised.

    Parameters
    ----------
    z : pl.Series
        The input time series.
    nu : float | None
        The smoothing parameter.
    span : int | None
        The span of the moving average (number of periods), defaults to None.
        span is related to nu by: nu = 1 - 2 / (span + 1).

    Returns
    -------
    pl.Series
        The exponentially weighted moving average of the input series.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html
    .. [2] Eq. (3) Science & Practice of trend-following systems.
    .. [3] Eq. (6) Science & Practice of trend-following systems.


    """
    nu = _validate_nu_or_span(nu, span)
    return z.ewm_mean(alpha=1 - nu)


def variance_preserving_ewma(z: pl.Series, nu: float | None = None, span: int | None = None) -> pl.Series:
    r"""Calculate variance preserving exponential moving average filter of a series.

    .. math::
        \text{VP-EWMA}(z_t) = \text{EWMA}(z_t) \sqrt{\frac{1 + \nu}{1 - \nu}}

    Parameters
    ----------
    z : pl.Series
        The input time series.
    nu : float | None
        The smoothing factor (0 < nu < 1). This is related
        to polars' ewm_mean's alpha by: alpha = 1 - nu.
    span : int | None
        The span of the moving average (number of periods), defaults to None.
        span is related to nu by: nu = 1 - 2 / (span + 1).
        If both `nu` and `span` are provided, `nu` takes precedence.
        If neither is provided, a ValueError is raised.

    Returns
    -------
    pl.Series
        The variance preserving exponentially weighted moving average of the input series.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html
    .. [2] Eq. (4) Science & Practice of trend-following systems.
    .. [3] Eq. (7) Science & Practice of trend-following systems.

    """
    nu = _validate_nu_or_span(nu, span)
    return ewma(z=z, nu=nu) * ((1.0 + nu) / (1.0 - nu)) ** 0.5


def long_short_variance_preserving_ewma(
    z: pl.Series,
    nu1: float | None = None,
    span1: int | None = None,
    nu2: float | None = None,
    span2: int | None = None,
) -> pl.Series:
    r"""Calculate long-short variance preserving exponential moving average filter of a series.

    ... math::
        \text{LS-VP-EWMA}(z_t) = l_1 \text{EWMA}(z_t, \nu_1) - l_2 \text{EWMA}(z_t, \nu_2)
        \\text{where} \quad l_1 = \frac{q}{1 - \nu_1}, \quad l_2 = \frac{q}{1 - \nu_2}, \quad
        q = \left(\frac{1}{1 - \nu_1^2} + \frac{1}{1 - \nu_2^2} - \frac{2}{1 - \nu_1 \nu_2}\right)^{-0.5}

    Parameters
    ----------
    z : pl.Series
        The input time series.
    nu1 : float | None
        The smoothing factor for the long position (0 < nu1 < 1).
    span1 : int | None
        The span for the long position (alternative to nu1).
    nu2 : float | None
        The smoothing factor for the short position (0 < nu2 < 1).
    span2 : int | None
        The span for the short position (alternative to nu2).

    Returns
    -------
    pl.Series
        The long-short variance preserving exponentially weighted moving average of the input series.

    Raises
    ------
    ValueError
        If nu1 and nu2 are equal or invalid.

    References
    ----------
    .. [1] https://docs.pola.rs/api/python/stable/reference/series/api/polars.Series.ewm_mean.html
    .. [2] Eq. (8) Science & Practice of trend-following systems.
    .. [3] Eq. (9) Science & Practice of trend-following systems.
    In Eq. 9 of paper, there is a typo: q should be 1 / q.

    """
    nu1 = _validate_nu_or_span(nu1, span1)
    nu2 = _validate_nu_or_span(nu2, span2)
    if nu1 == nu2:
        msg = "nu1 and nu2 must be different. When they are equal, the long-short filter is ill-defined."
        raise ValueError(msg)
    q = (1 / (1 - nu1**2.) + 1 / (1 - nu2**2.) - 2 / (1 - nu1 * nu2)) ** -.5
    l1 = q / (1 - nu1)
    l2 = q / (1 - nu2)
    return l1 * ewma(z=z, nu=nu1) - l2 * ewma(z=z, nu=nu2)
