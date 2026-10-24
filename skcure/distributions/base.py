"""Interfaces for latency distributions used by cure models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@runtime_checkable
class LatencyDistribution(Protocol):
    """Protocol for susceptible-event-time distributions.

    A latency distribution describes the event-time distribution among
    susceptible individuals. Cure-model components should depend on this
    interface rather than on a specific parametric family.

    Implementations must provide probability density, log-density,
    cumulative distribution, and survival functions.

    Notes
    -----
    This protocol intentionally contains only distribution evaluation.
    Parameter estimation, EM updates, gradients, and optimization are kept
    separate because those procedures can differ substantially across
    latency families.
    """

    def pdf(self, t: ArrayLike) -> FloatArray:
        """Evaluate the probability density function."""
        ...

    def logpdf(self, t: ArrayLike) -> FloatArray:
        """Evaluate the log-density."""
        ...

    def cdf(self, t: ArrayLike) -> FloatArray:
        """Evaluate the cumulative distribution function."""
        ...

    def sf(self, t: ArrayLike) -> FloatArray:
        """Evaluate the survival function."""
        ...


@runtime_checkable
class _LogCDFDistribution(Protocol):
    def logcdf(self, t: ArrayLike) -> FloatArray: ...


@runtime_checkable
class _LogSurvivalDistribution(Protocol):
    def logsf(self, t: ArrayLike) -> FloatArray: ...


def latency_logcdf(distribution: LatencyDistribution, t: ArrayLike) -> FloatArray:
    """Use a distribution's log-CDF, retaining support for older interfaces."""
    if isinstance(distribution, _LogCDFDistribution):
        return np.asarray(distribution.logcdf(t), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(distribution.cdf(t), dtype=float))


def latency_logsf(distribution: LatencyDistribution, t: ArrayLike) -> FloatArray:
    """Use a distribution's log-survival without first rounding a tail to zero."""
    if isinstance(distribution, _LogSurvivalDistribution):
        return np.asarray(distribution.logsf(t), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(distribution.sf(t), dtype=float))


def validate_latency_distribution(distribution: object) -> LatencyDistribution:
    """Validate that an object implements the latency-distribution interface.

    Parameters
    ----------
    distribution : object
        Candidate latency-distribution object.

    Returns
    -------
    distribution : LatencyDistribution
        The validated object.

    Raises
    ------
    TypeError
        If the object does not implement the required latency-distribution
        methods.
    """
    if not isinstance(distribution, LatencyDistribution):
        raise TypeError(
            "latency distribution must implement "
            "pdf(t), logpdf(t), cdf(t), and sf(t)."
        )

    return distribution


__all__ = [
    "LatencyDistribution",
    "validate_latency_distribution",
    "latency_logcdf",
    "latency_logsf",
]
