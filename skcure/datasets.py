"""Reproducible synthetic cure data with observable and latent ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root_scalar
from scipy.special import expit, logit, ndtr
from sksurv.util import Surv


@dataclass(frozen=True)
class CureSimulation:
    """A simulated mixture-cure cohort and its known data-generating truth.

    ``cured`` is available only because these data are simulated. A censored
    observation in ``y`` does not imply cure. ``event_time`` is infinity for
    cured subjects. Covariates affect logistic cure probability; susceptible
    latency is independent of covariates and has unit scale.
    """

    X: np.ndarray
    y: np.ndarray
    cure_probability: np.ndarray
    cured: np.ndarray
    event_time: np.ndarray
    censoring_time: np.ndarray
    latency: str
    latency_shape: float

    def survival_probability(self, times: Any) -> np.ndarray:
        """Return the true conditional population survival on a time grid."""
        grid = np.asarray(times, dtype=float)
        if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
            raise ValueError("times must be a finite, non-negative one-dimensional array.")
        if self.latency == "weibull":
            susceptible_survival = np.exp(-(grid**self.latency_shape))
        else:
            with np.errstate(divide="ignore"):
                susceptible_survival = ndtr(-np.log(grid) / self.latency_shape)
        cure = self.cure_probability[:, None]
        return cure + (1.0 - cure) * susceptible_survival[None, :]


def simulate_cure_survival(
    n_samples: int = 500,
    *,
    n_features: int = 2,
    cure_fraction: float = 0.3,
    cure_coefficients: Any = None,
    latency: str = "weibull",
    latency_shape: float = 1.5,
    censoring_rate: float = 0.1,
    followup: float | None = 5.0,
    random_state: Any = None,
) -> CureSimulation:
    """Generate a logistic mixture-cure model with independent censoring.

    Features are independent standard normal variables. Logistic cure
    coefficients default to ``[0.8, -0.6, 0, ...]``. The intercept is calibrated
    by Gaussian quadrature so the population cure fraction is approximately
    ``cure_fraction``; a realized cohort's fraction is random.

    Susceptible latency is Weibull with unit scale and the given shape, or
    log-normal with log-mean zero and log-standard-deviation ``latency_shape``.
    Weibull provides a correctly specified generalized-gamma latency scenario;
    log-normal provides a latency misspecification scenario. Censoring is the
    minimum of an independent exponential time (rate ``censoring_rate``) and
    administrative ``followup``. A zero rate disables exponential censoring;
    ``followup=None`` disables administrative censoring. Neither parameter is
    a target observed censoring proportion: cured subjects are also censored.
    """
    for name, value in (("n_samples", n_samples), ("n_features", n_features)):
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(cure_fraction, Real) or not 0 < cure_fraction < 1:
        raise ValueError("cure_fraction must lie strictly between zero and one.")
    if latency not in ("weibull", "lognormal"):
        raise ValueError("latency must be 'weibull' or 'lognormal'.")
    if not isinstance(latency_shape, Real) or not np.isfinite(float(latency_shape)) or latency_shape <= 0:
        raise ValueError("latency_shape must be positive and finite.")
    if not isinstance(censoring_rate, Real) or not np.isfinite(float(censoring_rate)) or censoring_rate < 0:
        raise ValueError("censoring_rate must be non-negative and finite.")
    if followup is not None and (not isinstance(followup, Real) or not np.isfinite(float(followup)) or followup <= 0):
        raise ValueError("followup must be positive and finite, or None.")
    if followup is None and censoring_rate == 0:
        raise ValueError("At least one finite censoring mechanism is needed to observe cured subjects.")
    if cure_coefficients is None:
        coefficients = np.zeros(n_features)
        coefficients[: min(2, n_features)] = [0.8, -0.6][: min(2, n_features)]
    else:
        coefficients = np.asarray(cure_coefficients, dtype=float)
        if coefficients.shape != (n_features,) or not np.all(np.isfinite(coefficients)):
            raise ValueError("cure_coefficients must have one finite value per feature.")
    standard_deviation = float(np.linalg.norm(coefficients))
    if not np.isfinite(standard_deviation):
        raise ValueError("cure_coefficients have an unrepresentable norm.")
    nodes, weights = hermgauss(64)
    offsets = np.sqrt(2.0) * standard_deviation * nodes
    center = float(logit(cure_fraction))
    half_width = max(40.0, 12.0 * standard_deviation)
    solution = root_scalar(
        lambda value: float(weights @ expit(value + offsets) / np.sqrt(np.pi) - cure_fraction),
        bracket=(center - half_width, center + half_width),
    )
    if not solution.converged:
        raise RuntimeError("Could not calibrate the requested population cure fraction.")
    intercept = float(solution.root)
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    cure_probability = expit(intercept + X @ coefficients)
    cured = rng.uniform(size=n_samples) < cure_probability
    if latency == "weibull":
        susceptible_time = rng.weibull(latency_shape, size=n_samples)
    else:
        susceptible_time = rng.lognormal(sigma=latency_shape, size=n_samples)
    event_time = np.where(cured, np.inf, susceptible_time)
    censoring_time = (
        rng.exponential(scale=1.0 / censoring_rate, size=n_samples)
        if censoring_rate > 0
        else np.full(n_samples, np.inf)
    )
    if followup is not None:
        censoring_time = np.minimum(censoring_time, followup)
    observed_time = np.minimum(event_time, censoring_time)
    event = event_time <= censoring_time
    if np.any(~np.isfinite(observed_time)) or np.any(observed_time <= 0):
        raise ValueError("Simulation parameters produced non-finite or non-positive observed times.")
    y = Surv.from_arrays(event=event, time=observed_time)
    return CureSimulation(X, y, cure_probability, cured, event_time, censoring_time, latency, float(latency_shape))


__all__ = ["CureSimulation", "simulate_cure_survival"]
