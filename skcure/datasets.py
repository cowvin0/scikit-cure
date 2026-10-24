from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root_scalar
from scipy.special import expit, log_expit, log_ndtr, logit, ndtr, ndtri_exp
from sksurv.util import Surv

LATENT_DISTRIBUTIONS = ("binomial", "poisson", "negative_binomial")


def _negative_binomial_log_survival(log_cumulative_mean, phi):
    """Evaluate -log(1 + phi * exp(log_mean))/phi, including phi=0."""
    values = np.asarray(log_cumulative_mean, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if phi == 0.0:
            return -np.exp(values)
        argument = np.log(phi) + values
        result = np.asarray(-np.logaddexp(0.0, argument) / phi)
        # Dividing a rounded-to-zero phi*mean by phi would erase a finite mean.
        small = argument < -18.0
        relative = np.exp(argument[small])
        result[small] = -np.exp(values[small]) * (
            1.0 - relative / 2.0 + relative**2 / 3.0
        )
    return result


def _count_cure_probability(cure_predictor, family, trials, phi):
    """Evaluate P(N=0); positive cure coefficients decrease incidence."""
    values = np.asarray(cure_predictor, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        if family == "binomial":
            if trials == 1:
                return expit(values)
            return np.exp(trials * log_expit(values))
        if family == "poisson":
            return np.exp(-np.exp(-values))
        return np.exp(_negative_binomial_log_survival(-values, phi))


@dataclass(frozen=True)
class CureSimulation:
    """A simulated latent-count cure cohort and its known generating truth.

    ``cured`` is available only because these data are simulated. A censored
    observation in ``y`` does not imply cure. ``event_time`` is infinity for
    cured subjects. Each latent cause has a covariate-independent unit-scale
    latency; the observed event is the first cause's event. The additional
    count and incidence fields make family-specific validation possible.
    """

    X: np.ndarray
    y: np.ndarray
    cure_probability: np.ndarray
    cured: np.ndarray
    event_time: np.ndarray
    censoring_time: np.ndarray
    latency: str
    latency_shape: float
    latent_distribution: str = "binomial"
    K_bin: int = 1
    phi: float = 0.5
    latent_count: np.ndarray | None = None
    latent_mean: np.ndarray | None = None
    incidence_predictor: np.ndarray | None = None
    incidence_intercept: float | None = None
    incidence_coefficients: np.ndarray | None = None

    def survival_probability(self, times: Any) -> np.ndarray:
        """Return the true conditional population survival on a time grid."""
        grid = np.asarray(times, dtype=float)
        if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
            raise ValueError(
                "times must be a finite, non-negative one-dimensional array."
            )
        if self.latency == "weibull":
            with np.errstate(over="ignore", under="ignore"):
                hazard = grid**self.latency_shape
                susceptible_survival = np.exp(-hazard)
                log_latent_survival = -hazard
                latent_cdf = -np.expm1(-hazard)
        else:
            with np.errstate(divide="ignore"):
                susceptible_survival = ndtr(-np.log(grid) / self.latency_shape)
                log_latent_survival = log_ndtr(-np.log(grid) / self.latency_shape)
                latent_cdf = ndtr(np.log(grid) / self.latency_shape)
        if self.incidence_predictor is None:
            # Keep manually constructed instances from the original API valid.
            if self.latent_distribution != "binomial" or self.K_bin != 1:
                raise ValueError(
                    "incidence_predictor is required for this latent count family."
                )
            cure = self.cure_probability[:, None]
            return cure + (1.0 - cure) * susceptible_survival[None, :]
        eta = self.incidence_predictor[:, None]
        with np.errstate(divide="ignore", under="ignore"):
            if self.latent_distribution == "binomial":
                if self.K_bin == 1:
                    cure = self.cure_probability[:, None]
                    return cure + (1.0 - cure) * susceptible_survival[None, :]
                log_probability = np.logaddexp(
                    log_expit(-eta), log_expit(eta) + log_latent_survival
                )
                # For a small event probability, summing the two survival
                # components loses precision that a large K can amplify.
                # log1p is accurate here; retain the log-sum for the cure tail.
                cumulative = expit(eta) * latent_cdf
                small_event = cumulative <= 0.5
                log_probability[small_event] = np.log1p(-cumulative[small_event])
                # The log-sum can round a few ulps above or below zero when
                # the latent event probability vanishes. Preserve S(0)=1
                # exactly and prevent roundoff from producing S(t)>1.
                probability = np.exp(self.K_bin * np.minimum(log_probability, 0.0))
                probability[:, grid == 0.0] = 1.0
                return probability
            log_cumulative_mean = eta + np.log(latent_cdf)
            if self.latent_distribution == "poisson":
                return np.exp(-np.exp(log_cumulative_mean))
            return np.exp(
                _negative_binomial_log_survival(log_cumulative_mean, self.phi)
            )


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
    latent_distribution: str = "binomial",
    K_bin: int = 1,
    phi: float = 0.5,
) -> CureSimulation:
    """Generate a latent-count cure model with independent censoring.

    Features are independent standard normal variables. The
    ``cure_coefficients`` default to ``[0.8, -0.6, 0, ...]``; positive coefficients
    increase cure probability. The incidence predictor is linear under each
    count family's link, with the opposite sign to these cure coefficients.
    The intercept is calibrated
    by Gaussian quadrature so the population cure fraction is approximately
    ``cure_fraction``; a realized cohort's fraction is random.

    Latent-cause latency is Weibull with unit scale and the given shape, or
    log-normal with log-mean zero and log-standard-deviation ``latency_shape``.
    Weibull provides a correctly specified generalized-gamma latency scenario;
    log-normal provides a latency misspecification scenario. Censoring is the
    minimum of an independent exponential time (rate ``censoring_rate``) and
    administrative ``followup``. A zero rate disables exponential censoring;
    ``followup=None`` disables administrative censoring. Neither parameter is
    a target observed censoring proportion: cured subjects are also censored.

    ``latent_distribution`` selects binomial, Poisson, or negative-binomial
    causes. ``K_bin`` is the binomial trial count; the default one recovers the
    original logistic mixture-cure generator. Negative-binomial counts have
    variance ``mean + phi * mean**2``; ``phi=0`` gives the Poisson limit.
    Complete latent counts are retained, but event generation uses an inverse
    CDF without constructing one event time per latent cause.
    """
    for name, value in (("n_samples", n_samples), ("n_features", n_features)):
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    if not isinstance(cure_fraction, Real) or not 0 < cure_fraction < 1:
        raise ValueError("cure_fraction must lie strictly between zero and one.")
    if latency not in ("weibull", "lognormal"):
        raise ValueError("latency must be 'weibull' or 'lognormal'.")
    if (
        not isinstance(latency_shape, Real)
        or not np.isfinite(float(latency_shape))
        or latency_shape <= 0
    ):
        raise ValueError("latency_shape must be positive and finite.")
    if (
        not isinstance(censoring_rate, Real)
        or not np.isfinite(float(censoring_rate))
        or censoring_rate < 0
    ):
        raise ValueError("censoring_rate must be non-negative and finite.")
    if followup is not None and (
        not isinstance(followup, Real)
        or not np.isfinite(float(followup))
        or followup <= 0
    ):
        raise ValueError("followup must be positive and finite, or None.")
    if followup is None and censoring_rate == 0:
        raise ValueError(
            "At least one finite censoring mechanism is needed to observe cured subjects."
        )
    if latent_distribution not in LATENT_DISTRIBUTIONS:
        raise ValueError(f"latent_distribution must be one of {LATENT_DISTRIBUTIONS}.")
    if (
        not isinstance(K_bin, Integral)
        or isinstance(K_bin, (bool, np.bool_))
        or not 1 <= K_bin <= np.iinfo(np.int64).max
    ):
        raise ValueError("K_bin must be a positive representable integer.")
    if (
        not isinstance(phi, Real)
        or isinstance(phi, (bool, np.bool_))
        or not np.isfinite(float(phi))
        or phi < 0
    ):
        raise ValueError("phi must be non-negative and finite.")
    if (
        latent_distribution == "negative_binomial"
        and phi > 0
        and not np.isfinite(1.0 / phi)
    ):
        raise ValueError(
            "phi is too small for count sampling; use phi=0 for its Poisson limit."
        )
    if cure_coefficients is None:
        coefficients = np.zeros(n_features)
        coefficients[: min(2, n_features)] = [0.8, -0.6][: min(2, n_features)]
    else:
        coefficients = np.asarray(cure_coefficients, dtype=float)
        if coefficients.shape != (n_features,) or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "cure_coefficients must have one finite value per feature."
            )
    standard_deviation = float(np.linalg.norm(coefficients))
    if not np.isfinite(standard_deviation):
        raise ValueError("cure_coefficients have an unrepresentable norm.")
    nodes, weights = hermgauss(64)
    offsets = np.sqrt(2.0) * standard_deviation * nodes
    center = float(logit(cure_fraction))
    half_width = max(40.0, 12.0 * standard_deviation)

    def target(value):
        probability = _count_cure_probability(
            value + offsets, latent_distribution, K_bin, phi
        )
        return float(weights @ probability / np.sqrt(np.pi) - cure_fraction)

    for _ in range(48):
        if target(center - half_width) < 0 < target(center + half_width):
            break
        half_width *= 2.0
    else:
        raise ValueError(
            "The requested cure fraction needs unrepresentable count parameters."
        )
    solution = root_scalar(target, bracket=(center - half_width, center + half_width))
    if not solution.converged:
        raise RuntimeError(
            "Could not calibrate the requested population cure fraction."
        )
    intercept = float(solution.root)
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    cure_predictor = intercept + X @ coefficients
    eta = -cure_predictor
    cure_probability = _count_cure_probability(
        cure_predictor, latent_distribution, K_bin, phi
    )
    with np.errstate(over="ignore", under="ignore"):
        latent_mean = (
            K_bin * expit(eta) if latent_distribution == "binomial" else np.exp(eta)
        )
    if not np.all(np.isfinite(latent_mean)):
        raise ValueError(
            "Simulation parameters produced an unrepresentable latent mean."
        )
    if latent_distribution == "binomial" and K_bin == 1:
        # Preserve the original default generator, including its seeded stream.
        cured = rng.uniform(size=n_samples) < cure_probability
        latent_count = (~cured).astype(np.int64)
        if latency == "weibull":
            susceptible_time = rng.weibull(latency_shape, size=n_samples)
        else:
            susceptible_time = rng.lognormal(sigma=latency_shape, size=n_samples)
        event_time = np.where(cured, np.inf, susceptible_time)
    else:
        try:
            if latent_distribution == "binomial":
                latent_count = rng.binomial(K_bin, expit(eta))
            elif latent_distribution == "poisson" or phi == 0.0:
                latent_count = rng.poisson(latent_mean)
            else:
                # Gamma-Poisson mixing avoids success-probability rounding to
                # one when a small positive dispersion approaches Poisson.
                intensity = rng.gamma(1.0 / phi, phi * latent_mean)
                latent_count = rng.poisson(intensity)
        except ValueError as exc:
            raise ValueError(
                "Latent counts exceed NumPy's representable sampling range."
            ) from exc
        cured = latent_count == 0
        event_time = np.full(n_samples, np.inf)
        active = ~cured
        uniforms = np.clip(
            rng.uniform(size=int(active.sum())),
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
        # Conditional on N, P(T > t | N) = (1-F0(t))**N.
        log_survival = np.log(uniforms) / latent_count[active]
        if latency == "weibull":
            event_time[active] = (-log_survival) ** (1.0 / latency_shape)
        else:
            with np.errstate(divide="ignore", over="ignore"):
                log_cdf = np.log(-np.expm1(log_survival))
                normal_quantile = np.where(
                    log_cdf < -np.log(2.0),
                    ndtri_exp(log_cdf),
                    -ndtri_exp(log_survival),
                )
                event_time[active] = np.exp(latency_shape * normal_quantile)
    if np.any(~np.isfinite(event_time[~cured])) or np.any(event_time[~cured] <= 0):
        raise ValueError(
            "Simulation parameters produced unrepresentable latent event times."
        )
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
        raise ValueError(
            "Simulation parameters produced non-finite or non-positive observed times."
        )
    y = Surv.from_arrays(event=event, time=observed_time)
    return CureSimulation(
        X,
        y,
        cure_probability,
        cured,
        event_time,
        censoring_time,
        latency,
        float(latency_shape),
        latent_distribution=latent_distribution,
        K_bin=int(K_bin),
        phi=float(phi),
        latent_count=latent_count,
        latent_mean=latent_mean,
        incidence_predictor=eta,
        incidence_intercept=-intercept,
        incidence_coefficients=-coefficients.copy(),
    )


__all__ = ["CureSimulation", "simulate_cure_survival"]
