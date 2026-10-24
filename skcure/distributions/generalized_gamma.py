from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, exp1, gammainc, gammaincc, gammaln

from skcure.utils.numerics import _safe_exp

__all__ = [
    "GeneralizedGamma",
    "pdf",
    "logpdf",
    "cdf",
    "sf",
    "logcdf",
    "logsf",
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

    with np.errstate(divide="ignore", invalid="ignore"):
        log_t = np.log(t)

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
    # At d == 1 the power of t is identically one, including at t == 0.
    with np.errstate(invalid="ignore", over="ignore"):
        power = np.zeros_like(log_t) if d == 1.0 else (d - 1.0) * log_t
        result = np.log(p) + power - u - d * np.log(a) - log_gamma_s
    return np.where(np.isposinf(log_t), -np.inf, result)


def _gamma_upper_tail_terms(s, u, log_u):
    """Return log(Q), (dP/ds)/Q, and u*gamma_pdf(u)/Q in the upper tail.

    A differentiated continued fraction evaluates the ratios directly. It
    remains useful after both the survival probability and dP/ds underflow.
    The caller uses this only for u >= s + 1, where the fraction converges.
    """
    u, log_u = np.broadcast_arrays(
        np.asarray(u, dtype=float), np.asarray(log_u, dtype=float)
    )
    b = u + (1.0 - s)
    c = np.full_like(u, 1.0 / np.finfo(float).tiny)
    dc = np.zeros_like(u)
    d = 1.0 / b
    dd = d * d
    h = d.copy()
    dh = dd.copy()
    tiny = np.finfo(float).tiny
    converged = np.zeros_like(u, dtype=bool)
    for i in range(1, 10001):
        an = i * (s - i)
        b = b + 2.0
        denominator = an * d + b
        d_denominator = i * d + an * dd - 1.0
        denominator = np.where(np.abs(denominator) < tiny, tiny, denominator)
        next_c = b + an / c
        next_dc = -1.0 + i / c - (an / c) * (dc / c)
        c = np.where(np.abs(next_c) < tiny, tiny, next_c)
        dc = next_dc
        d = 1.0 / denominator
        dd = -(d_denominator * d) * d
        factor = d * c
        d_factor = dd * c + d * dc
        next_dh = dh * factor + h * d_factor
        h = h * factor
        dh = next_dh
        # The derivative may converge a few iterations after the value.
        converged = (np.abs(factor - 1.0) <= 8.0 * np.finfo(float).eps) & (
            np.abs(d_factor) <= 8.0 * np.finfo(float).eps
        )
        if np.all(converged):
            break
    if not np.all(converged):
        raise FloatingPointError(
            "Generalized-gamma upper-tail fraction did not converge."
        )
    log_q = s * log_u - u - gammaln(s) + np.log(h)
    shape_ratio = -(log_u - digamma(s) + dh / h)
    return log_q, shape_ratio, 1.0 / h


def _gamma_lower_log_and_shape_derivative(s, u, log_u):
    """Evaluate log(P) and its shape derivative by a differentiated series."""
    term = np.ones_like(u)
    total = term.copy()
    derivative_term = np.zeros_like(u)
    derivative_total = derivative_term.copy()
    for k in range(1, 10001):
        factor = u / (s + k)
        term = term * factor
        derivative_term = derivative_term * factor - term / (s + k)
        total = total + term
        derivative_total = derivative_total + derivative_term
        if np.all(np.abs(term) <= np.abs(total) * (4.0 * np.finfo(float).eps)):
            break
    else:
        raise FloatingPointError(
            "Generalized-gamma lower-tail series did not converge."
        )
    # gammaln(1+s) must not round to zero before multiplication by tiny s.
    log_gamma = (
        -np.euler_gamma * s + (np.pi**2 / 12.0) * s * s
        if s < 1.0e-8
        else gammaln(s + 1.0)
    )
    log_probability = s * log_u - u - log_gamma + np.log(total)
    derivative = log_u - digamma(s + 1.0) + derivative_total / total
    return log_probability, derivative


def _gamma_lower_log_series(s, u, log_u):
    """Evaluate log(P(s,u)) without forming its possibly vanishing prefactor."""
    return _gamma_lower_log_and_shape_derivative(s, u, log_u)[0]


def _log_complement(log_probability):
    """Evaluate log(1-exp(log_probability)) on either side of log(1/2)."""
    result = np.empty_like(log_probability)
    small = log_probability < -np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        result[small] = np.log1p(-np.exp(log_probability[small]))
        result[~small] = np.log(-np.expm1(log_probability[~small]))
    return result


def _gamma_log_probabilities(s, u, log_u):
    """Evaluate regularized incomplete gamma probabilities on the log scale."""
    lower = np.asarray(gammainc(s, u), dtype=float)
    upper = np.asarray(gammaincc(s, u), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lower = np.asarray(np.log(lower), dtype=float)
        log_upper = np.asarray(np.log(upper), dtype=float)

    # log_u still carries a finite argument if exp(log_u) underflowed.
    lower_tail = np.isfinite(log_u) & (lower < 1.0e-280)
    if np.any(lower_tail):
        log_lower[lower_tail] = _gamma_lower_log_series(
            s, u[lower_tail], log_u[lower_tail]
        )
    upper_tail = np.isfinite(u) & (u >= s + 1.0) & (upper < 1.0e-280)
    if np.any(upper_tail):
        log_upper[upper_tail] = _gamma_upper_tail_terms(
            s, u[upper_tail], log_u[upper_tail]
        )[0]
    tiny_shape_tail = np.isfinite(log_u) & (u < s + 1.0) & (upper < 1.0e-280)
    if np.any(tiny_shape_tail):
        # Here s itself is tiny: Q(s,u) = s*E1(u) + O(s**2).
        argument = u[tiny_shape_tail]
        integral = np.where(
            argument == 0.0,
            -np.euler_gamma - log_u[tiny_shape_tail],
            exp1(argument),
        )
        log_upper[tiny_shape_tail] = np.log(s) + np.log(integral)

    # Complement a very small tail, rather than using a probability rounded
    # to one. Keep SciPy's direct evaluations in the ordinary central range.
    small_lower = log_lower < np.log(1.0e-8)
    log_upper[small_lower] = _log_complement(log_lower[small_lower])
    small_upper = log_upper < np.log(1.0e-8)
    log_lower[small_upper] = _log_complement(log_upper[small_upper])
    # When u underflows and shape is tiny, the recovered CDF need not be small.
    recovered = lower_tail & ~small_lower
    log_upper[recovered] = _log_complement(log_lower[recovered])
    return log_lower, log_upper


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
    _, _, log_u, u = _argument_from_parameters(
        t,
        a,
        p,
    )

    result = np.asarray(
        gammainc(
            d / p,
            u,
        ),
        dtype=float,
    )
    recover = np.isfinite(log_u) & (result < np.finfo(float).tiny)
    if np.any(recover):
        with np.errstate(under="ignore"):
            result[recover] = np.exp(
                _gamma_log_probabilities(d / p, u[recover], log_u[recover])[0]
            )
    return result


def sf(t, a, d, p):
    """Survival function of the generalized gamma distribution."""
    _, _, log_u, u = _argument_from_parameters(
        t,
        a,
        p,
    )

    result = np.asarray(
        gammaincc(
            d / p,
            u,
        ),
        dtype=float,
    )
    recover = np.isfinite(log_u) & ((result < np.finfo(float).tiny) | (u == 0.0))
    if np.any(recover):
        with np.errstate(under="ignore"):
            result[recover] = np.exp(
                _gamma_log_probabilities(d / p, u[recover], log_u[recover])[1]
            )
    return result


def logcdf(t, a, d, p):
    """Log-CDF, including probabilities below the floating-point range."""
    _, _, log_u, u = _argument_from_parameters(t, a, p)
    return _gamma_log_probabilities(d / p, u, log_u)[0]


def logsf(t, a, d, p):
    """Log-survival, computed without taking log of an underflowed tail."""
    _, _, log_u, u = _argument_from_parameters(t, a, p)
    return _gamma_log_probabilities(d / p, u, log_u)[1]


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

    def logcdf(self, t):
        """Log-CDF, preserving information in extremely small lower tails."""
        return logcdf(t, self.a, self.d, self.p)

    def logsf(self, t):
        """Log-survival, preserving information in extremely small upper tails."""
        return logsf(t, self.a, self.d, self.p)

    def _argument(self, t):
        """Return transformed generalized-gamma quantities for ``t``."""
        return _argument_from_parameters(
            t,
            self.a,
            self.p,
        )
