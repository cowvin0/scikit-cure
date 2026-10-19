"""Generalized gamma distribution family and numerical helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc, gammaincc, gammaln

from skcure.utils.numerics import _safe_exp

__all__ = [
    "GeneralizedGamma",
    "pdf",
    "logpdf",
    "cdf",
    "sf",
]


def _argument_from_parameters(t, a, p):
    """Compute transformed generalized-gamma quantities.

    Returns
    -------
    log_t : ndarray
        ``log(t)``.
    log_ta : ndarray
        ``log(t / a)``.
    log_u : ndarray
        ``p * log(t / a)``.
    u : ndarray
        ``(t / a)**p``.
    """
    t = np.asarray(
        t,
        dtype=float,
    )

    log_t = np.full_like(
        t,
        -np.inf,
        dtype=float,
    )

    positive_t = t > 0.0

    log_t[positive_t] = np.log(t[positive_t])
    log_t[t < 0.0] = np.nan

    log_ta = log_t - np.log(a)

    with np.errstate(
        over="ignore",
        invalid="ignore",
    ):
        log_u = p * log_ta

    u = _safe_exp(log_u)

    return (
        log_t,
        log_ta,
        log_u,
        u,
    )


def _gg_log_density(log_t, u, a, d, p, log_gamma_s):
    """Evaluate the GG log-density from precomputed transformed quantities."""
    return np.log(p) + (d - 1.0) * log_t - u - d * np.log(a) - log_gamma_s


def logpdf(t, a, d, p):
    """Log-density of the generalized gamma distribution."""
    log_t, _, _, u = _argument_from_parameters(
        t,
        a,
        p,
    )

    s = d / p

    return _gg_log_density(
        log_t,
        u,
        a,
        d,
        p,
        gammaln(s),
    )


def pdf(t, a, d, p):
    """Probability density function of the generalized gamma distribution."""
    return _safe_exp(
        logpdf(
            t,
            a,
            d,
            p,
        )
    )


def cdf(t, a, d, p):
    """Cumulative distribution function of the generalized gamma distribution."""
    _, _, _, u = _argument_from_parameters(
        t,
        a,
        p,
    )

    return gammainc(
        d / p,
        u,
    )


def sf(t, a, d, p):
    """Survival function of the generalized gamma distribution."""
    _, _, _, u = _argument_from_parameters(
        t,
        a,
        p,
    )

    return gammaincc(
        d / p,
        u,
    )


def _sf_from_cdf(t, a, d, p):
    """Compute survival as ``1 - CDF``.

    This is kept only for legacy numerical paths that historically used the
    CDF complement instead of ``gammaincc`` directly.
    """
    return np.clip(
        1.0 - cdf(t, a, d, p),
        0.0,
        1.0,
    )


def _validate_positive_finite_parameter(value, name):
    """Return a positive finite parameter as float."""
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")

    return value


@dataclass
class GeneralizedGamma:
    """Generalized gamma latency distribution.

    Parameters
    ----------
    a : float, default=1.0
        Positive scale parameter.
    d : float, default=1.0
        Positive shape parameter.
    p : float, default=1.0
        Positive shape parameter.

    Notes
    -----
    The module-level functions remain the canonical numerical implementation.
    This class is a lightweight parameterized interface around those functions,
    intended for composition inside cure-model estimators.
    """

    a: float = 1.0
    d: float = 1.0
    p: float = 1.0

    def __post_init__(self):
        self.a = _validate_positive_finite_parameter(
            self.a,
            "a",
        )
        self.d = _validate_positive_finite_parameter(
            self.d,
            "d",
        )
        self.p = _validate_positive_finite_parameter(
            self.p,
            "p",
        )

    def set_parameters(
        self,
        *,
        a=None,
        d=None,
        p=None,
    ):
        """Update distribution parameters in place and return ``self``."""
        new_a = self.a if a is None else _validate_positive_finite_parameter(a, "a")
        new_d = self.d if d is None else _validate_positive_finite_parameter(d, "d")
        new_p = self.p if p is None else _validate_positive_finite_parameter(p, "p")

        self.a = new_a
        self.d = new_d
        self.p = new_p

        return self

    def pdf(self, t):
        """Probability density function."""
        return pdf(
            t,
            self.a,
            self.d,
            self.p,
        )

    def logpdf(self, t):
        """Log-density."""
        return logpdf(
            t,
            self.a,
            self.d,
            self.p,
        )

    def cdf(self, t):
        """Cumulative distribution function."""
        return cdf(
            t,
            self.a,
            self.d,
            self.p,
        )

    def sf(self, t):
        """Survival function."""
        return sf(
            t,
            self.a,
            self.d,
            self.p,
        )

    def _sf_from_cdf(self, t):
        """Legacy survival evaluation through ``1 - CDF``."""
        return _sf_from_cdf(
            t,
            self.a,
            self.d,
            self.p,
        )

    def _argument(self, t):
        """Return transformed generalized-gamma quantities for ``t``."""
        return _argument_from_parameters(
            t,
            self.a,
            self.p,
        )
