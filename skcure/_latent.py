from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from scipy.special import expit
from scipy.optimize import minimize

from skcure.distributions.base import latency_logcdf, latency_logsf
from skcure.utils.numerics import _maximum_log_float, _exp_log_quantity, _exp_minus_exp


def _nb_population_terms(log_A, phi):
    """Return log(1 + phi*A) and log(1 + phi*A)/phi from log(A).

    Factoring the second quantity as A * log1p(x)/x avoids both an
    overflowing reciprocal of phi and loss of the Poisson limit when x
    underflows. The ratio is evaluated by its series near zero.
    """
    log_A = np.asarray(log_A, dtype=float)
    log_x = np.log(phi) + log_A
    log_component = np.logaddexp(0.0, log_x)
    log_scaled_component = np.empty_like(log_x)
    small = log_x < -8.0
    x = np.exp(log_x[small])
    correction = x * (-0.5 + x * (1.0 / 3.0 + x * (-0.25 + x / 5.0)))
    log_scaled_component[small] = log_A[small] + np.log1p(correction)
    regular = ~small
    log_scaled_component[regular] = np.log(log_component[regular]) - np.log(phi)
    scaled_component = _exp_log_quantity(log_scaled_component)
    return log_component, scaled_component


def _nb_log_scaled_mean(log_A, phi):
    """Return log(A / (1 + phi*A)) without subtracting large predictors."""
    log_A = np.asarray(log_A, dtype=float)
    log_phi = np.log(phi)
    log_x = log_phi + log_A
    result = np.empty_like(log_A)
    large = log_x > 0.0
    result[large] = -log_phi - np.logaddexp(0.0, -log_x[large])
    result[~large] = log_A[~large] - np.logaddexp(0.0, log_x[~large])
    return result


def _nb_phi_derivative(log_A, phi, delta):
    """Derivative of the NB observed log-likelihood with respect to phi."""
    log_A = np.asarray(log_A, dtype=float)
    log_x = np.log(phi) + log_A
    log_first = np.empty_like(log_x)
    small = log_x < np.log(0.01)
    x = np.exp(log_x[small])
    # (log1p(x) - x/(1+x))/x**2 = 1/2 - 2*x/3 + 3*x**2/4 - ...
    coefficients = [(-1.0) ** k * (k + 1.0) / (k + 2.0) for k in range(9)]
    log_first[small] = 2.0 * log_A[small] + np.log(
        np.polynomial.polynomial.polyval(x, coefficients)
    )
    regular = ~small
    numerator = np.logaddexp(0.0, log_x[regular]) - expit(log_x[regular])
    log_first[regular] = np.log(numerator) - 2.0 * np.log(phi)
    first = _exp_log_quantity(log_first)
    second = _exp_log_quantity(_nb_log_scaled_mean(log_A, phi))
    return first - delta * second


def _normalize_latent_distribution(value):
    """Resolve the latent-family names shared by kernel and ensemble models."""
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "poisson": "poisson",
        "negative_binomial": "negative_binomial",
        "negativebinomial": "negative_binomial",
        "nb": "negative_binomial",
        "binomial": "binomial",
    }
    if key not in aliases:
        raise ValueError(
            "latent_distribution must be one of "
            "{'poisson', 'negative_binomial', 'nb', 'binomial'}."
        )
    return aliases[key]


class _LatentCountModel(ABC):
    """Abstract fixed-predictor interface for latent count distributions.

    Predictor implementations such as linear models, kernel machines, and
    gradient boosting provide ``eta``. Latent-count implementations own the
    incidence objective, observed likelihood, E-step, and population survival
    calculations for that fixed predictor.

    Fit lifecycle hooks have no-op defaults because only stateful count
    families, currently the negative-binomial model, need to override them.
    ``predict_theta_from_eta`` is intentionally optional because a common
    latent-mean interpretation is not available for every count family.
    """

    name: ClassVar[str]

    def log_nonnegative(self, x):
        x = np.asarray(x, dtype=float)

        result = np.full_like(
            x,
            -np.inf,
            dtype=float,
        )

        positive = x > 0.0
        result[positive] = np.log(x[positive])
        result[np.isnan(x)] = np.nan

        return result

    @abstractmethod
    def incidence_value_and_grad(
        self,
        eta: np.ndarray,
        Nhat: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return negative incidence objective and gradient with respect to eta."""

    @abstractmethod
    def observed_loglikelihood_from_eta(
        self,
        latency,
        eta: np.ndarray,
        t: np.ndarray,
        delta: np.ndarray,
    ) -> float:
        """Return the unpenalized observed log-likelihood for a fixed predictor."""

    @abstractmethod
    def e_step_from_eta(
        self,
        latency,
        eta: np.ndarray,
        t: np.ndarray,
        delta: np.ndarray,
    ) -> np.ndarray:
        """Return the conditional latent-count expectation for a fixed predictor."""

    def initial_beta0(self, approximate_cure: float) -> float:
        """Return a default intercept matching an approximate cure fraction."""
        theta0 = -np.log(approximate_cure)
        return float(np.log(theta0))

    def finalize_fit(self) -> None:
        """Publish latent-family fitted state after a successful fit."""
        return None

    def reset_for_fit(self) -> None:
        """Reset latent-family state before a fresh fit."""
        return None

    def predict_theta_from_eta(self, eta: np.ndarray) -> np.ndarray:
        """Return a family-specific latent mean when that operation is defined."""
        raise RuntimeError(
            "predict_theta is defined only for " "latent_distribution='poisson'."
        )

    @abstractmethod
    def predict_log_survival_from_eta(
        self,
        latency,
        eta: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        """Evaluate log population survival without probability underflow."""

    @abstractmethod
    def predict_survival_from_eta(
        self,
        latency,
        eta: np.ndarray,
        t_grid: np.ndarray,
    ) -> np.ndarray:
        """Evaluate population survival for a fixed incidence predictor."""

    @abstractmethod
    def predict_cure_probability_from_eta(
        self,
        latency,
        eta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate cure probability for a fixed incidence predictor."""


class _PoissonLatentModel(_LatentCountModel):
    name = "poisson"
    _e_step_predictor_error = "Non-finite Poisson predictor."
    _e_step_overflow_error = "Poisson E-step overflow in conditional latent count."
    _e_step_count_error = "Non-finite Poisson conditional count."

    @staticmethod
    def _cumulative_intensity_is_invalid(cumulative):
        return np.any(~np.isfinite(cumulative))

    @staticmethod
    def _poisson_predictor_is_invalid(f, max_log):
        return np.any(f > max_log)

    def _observed_loglikelihood(self, latency, f, t, delta, cumulative):
        event_mask = delta > 0.5
        event_term = 0.0
        if np.any(event_mask):
            event_term = np.sum(f[event_mask] + latency.logpdf(t[event_mask]))
        return event_term - np.sum(cumulative)

    def incidence_value_and_grad(self, eta, Nhat):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        max_log = _maximum_log_float()

        if self._poisson_predictor_is_invalid(eta, max_log):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        theta = np.exp(eta)
        sum_theta = np.sum(theta)

        if not np.isfinite(sum_theta):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        Q = -sum_theta + np.sum(Nhat * eta)
        score = Nhat - theta

        if not np.isfinite(Q) or np.any(~np.isfinite(score)):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        return (
            -Q,
            -score,
        )

    def observed_loglikelihood_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            return -np.inf

        log_latency_cdf = latency_logcdf(latency, t)
        if np.any(np.isnan(log_latency_cdf)) or np.any(log_latency_cdf > 0.0):
            return -np.inf

        theta_latency_cdf = _exp_log_quantity(eta + log_latency_cdf)

        if self._cumulative_intensity_is_invalid(theta_latency_cdf):
            return -np.inf

        ll = self._observed_loglikelihood(
            latency,
            eta,
            t,
            delta,
            theta_latency_cdf,
        )

        if not np.isfinite(ll):
            return -np.inf

        return float(ll)

    def e_step_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError(self._e_step_predictor_error)

        log_latency_sf = latency_logsf(latency, t)
        if np.any(np.isnan(log_latency_sf)) or np.any(log_latency_sf > 0.0):
            raise FloatingPointError("Invalid Poisson latency survival probability.")
        expected_remaining = _exp_log_quantity(eta + log_latency_sf)

        if np.any(~np.isfinite(expected_remaining)):
            raise FloatingPointError(self._e_step_overflow_error)

        Nhat = delta + expected_remaining

        if np.any(~np.isfinite(Nhat)):
            raise FloatingPointError(self._e_step_count_error)

        return Nhat

    def predict_theta_from_eta(self, eta):
        max_log = _maximum_log_float()

        if np.any(eta > max_log):
            raise FloatingPointError(
                "Predicted Poisson intensity " "exceeds double precision."
            )

        return np.exp(eta)

    def predict_log_survival_from_eta(self, latency, eta, t_grid):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite Poisson predictor.")

        log_latency_cdf = latency_logcdf(latency, t_grid)
        if np.any(np.isnan(log_latency_cdf)) or np.any(log_latency_cdf > 0.0):
            raise FloatingPointError("Invalid Poisson latency cumulative probability.")

        log_theta_F = eta[:, None] + log_latency_cdf[None, :]

        return -_exp_log_quantity(log_theta_F)

    def predict_survival_from_eta(self, latency, eta, t_grid):
        with np.errstate(under="ignore"):
            return np.exp(self.predict_log_survival_from_eta(latency, eta, t_grid))

    def predict_cure_probability_from_eta(self, latency, eta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite Poisson predictor.")

        return _exp_minus_exp(eta)


@dataclass
class _NegativeBinomialLatentModel(_LatentCountModel):
    phi: float
    estimate_phi: bool
    phi_: float | None = None

    name = "negative_binomial"

    def __init__(self, phi=None):
        if phi is not None and phi <= 0:
            raise ValueError("phi must be positive.")

        self.estimate_phi = phi is None
        self._initial_phi = 0.5 if phi is None else float(phi)
        self.phi = self._initial_phi
        self.phi_ = None

    def incidence_value_and_grad(self, eta, Nhat):
        eta = np.asarray(eta, dtype=float)
        if np.any(~np.isfinite(eta)):
            return np.inf, np.zeros_like(eta)

        log_component, scaled_component = _nb_population_terms(eta, self.phi)
        log_x = np.log(self.phi) + eta
        centered_eta = eta - log_component
        large = log_x > 0.0
        centered_eta[large] = -np.log(self.phi) - np.logaddexp(0.0, -log_x[large])
        Q = np.sum(Nhat * centered_eta - scaled_component)
        score = Nhat * expit(-log_x) - _exp_log_quantity(
            _nb_log_scaled_mean(eta, self.phi)
        )

        if not np.isfinite(Q) or np.any(~np.isfinite(score)):
            return np.inf, np.zeros_like(eta)

        return (
            -Q,
            -score,
        )

    def observed_loglikelihood_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            return -np.inf

        log_latency_cdf = latency_logcdf(latency, t)
        if np.any(np.isnan(log_latency_cdf)) or np.any(log_latency_cdf > 0.0):
            return -np.inf
        log_population_term, scaled_population_term = _nb_population_terms(
            eta + log_latency_cdf, self.phi
        )

        event_mask = delta > 0.5
        event_term = 0.0

        if np.any(event_mask):
            event_values = eta[event_mask] + latency.logpdf(t[event_mask])

            if np.any(~np.isfinite(event_values)):
                return -np.inf

            event_term = np.sum(event_values)

        ll = event_term - np.sum(scaled_population_term + delta * log_population_term)

        return float(ll) if np.isfinite(ll) else -np.inf

    def e_step_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite negative-binomial predictor.")

        log_latency_cdf = latency_logcdf(latency, t)
        log_latency_sf = latency_logsf(latency, t)
        if (
            np.any(np.isnan(log_latency_cdf))
            or np.any(log_latency_cdf > 0.0)
            or np.any(np.isnan(log_latency_sf))
            or np.any(log_latency_sf > 0.0)
        ):
            raise FloatingPointError("Invalid negative-binomial latency probability.")

        phi = self.phi

        log_phi_A = np.log(phi) + eta + log_latency_cdf
        log_denominator = np.logaddexp(0.0, log_phi_A)
        log_ratio = eta + log_latency_sf - log_denominator
        large = log_phi_A > 0.0
        log_ratio[large] = (
            -np.log(phi)
            - log_latency_cdf[large]
            + log_latency_sf[large]
            - np.logaddexp(0.0, -log_phi_A[large])
        )
        log1p_phi_delta = np.zeros_like(
            delta,
            dtype=float,
        )

        event_mask = delta > 0.5
        log1p_phi_delta[event_mask] = np.log1p(phi)

        log_extra = log1p_phi_delta + log_ratio

        max_log = np.log(np.finfo(float).max)

        if np.any(log_extra > max_log):
            raise FloatingPointError(
                "Negative-binomial E-step " "overflow in conditional count."
            )

        with np.errstate(under="ignore"):
            additional_count = np.exp(log_extra)

        Nhat = delta + additional_count

        if np.any(~np.isfinite(Nhat)):
            raise FloatingPointError(
                "Non-finite negative-binomial " "conditional count."
            )

        return Nhat

    @staticmethod
    def _phi_invalid_value_and_grad():
        return (
            np.inf,
            np.full(
                1,
                np.nan,
                dtype=float,
            ),
        )

    def phi_value_and_grad(
        self,
        phi_arr,
        const,
        log_A,
        delta,
    ):
        phi = float(phi_arr[0])

        if not np.isfinite(phi) or phi <= 0.0:
            return (
                np.inf,
                np.array(
                    [np.nan],
                    dtype=float,
                ),
            )

        log1p_phi_A, scaled_component = _nb_population_terms(log_A, phi)
        ll = np.sum(const - scaled_component - delta * log1p_phi_A)
        grad = np.sum(_nb_phi_derivative(log_A, phi, delta))

        if not np.isfinite(ll) or not np.isfinite(grad):
            return (
                np.inf,
                np.array(
                    [np.nan],
                    dtype=float,
                ),
            )

        return (
            -ll,
            np.array(
                [-grad],
                dtype=float,
            ),
        )

    def log_phi_value_and_grad(
        self,
        log_phi_arr,
        const,
        log_A,
        delta,
    ):
        log_phi_arr = np.asarray(
            log_phi_arr,
            dtype=float,
        )

        if log_phi_arr.shape != (1,) or not np.isfinite(log_phi_arr[0]):
            return self._phi_invalid_value_and_grad()

        log_phi = float(log_phi_arr[0])
        log_max = np.log(np.finfo(float).max)

        if log_phi > log_max:
            return self._phi_invalid_value_and_grad()

        phi = float(np.exp(log_phi))

        if not np.isfinite(phi) or phi <= 0.0:
            return self._phi_invalid_value_and_grad()

        value, grad_phi = self.phi_value_and_grad(
            np.array(
                [phi],
                dtype=float,
            ),
            const,
            log_A,
            delta,
        )

        grad_phi = np.asarray(
            grad_phi,
            dtype=float,
        )

        if (
            not np.isfinite(value)
            or grad_phi.shape != (1,)
            or np.any(~np.isfinite(grad_phi))
        ):
            return self._phi_invalid_value_and_grad()

        with np.errstate(
            over="ignore",
            invalid="ignore",
        ):
            grad_log_phi = grad_phi * phi

        if np.any(~np.isfinite(grad_log_phi)):
            return self._phi_invalid_value_and_grad()

        return (
            value,
            grad_log_phi,
        )

    def dispersion_objective_terms_from_eta(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        """Build the fixed-predictor terms used by the dispersion objective."""
        eta = np.asarray(
            eta,
            dtype=float,
        )

        log_latency_cdf = latency_logcdf(latency, t)
        log_A = eta + log_latency_cdf

        latency_logpdf = latency.logpdf(t)

        const = np.zeros_like(
            delta,
            dtype=float,
        )

        event_mask = delta > 0.5
        const[event_mask] = eta[event_mask] + latency_logpdf[event_mask]

        if np.any(~np.isfinite(const[event_mask])):
            raise FloatingPointError(
                "Non-finite event contribution "
                "during negative-binomial "
                "dispersion update."
            )

        return const, log_A

    def _update_dispersion_in_log_space(
        self,
        const,
        log_A,
        delta,
    ):
        """Update phi in log coordinates using the historical GGKM path."""
        log_phi_lower = np.log(1.0e-8)

        result_phi = minimize(
            fun=self.log_phi_value_and_grad,
            x0=np.array(
                [np.log(self.phi)],
                dtype=float,
            ),
            jac=True,
            args=(
                const,
                log_A,
                delta,
            ),
            method="L-BFGS-B",
            bounds=[
                (
                    log_phi_lower,
                    None,
                )
            ],
        )

        if (
            not result_phi.success
            or not np.isfinite(result_phi.fun)
            or np.shape(result_phi.x) != (1,)
            or not np.all(np.isfinite(result_phi.x))
        ):
            raise RuntimeError(
                "Negative-binomial "
                "log-dispersion "
                "optimization failed: "
                f"{result_phi.message}"
            )

        log_phi_new = float(result_phi.x[0])

        if log_phi_new > np.log(np.finfo(float).max):
            raise FloatingPointError(
                "Negative-binomial "
                "log-dispersion "
                "optimization returned a value "
                "outside the representable "
                "floating-point range."
            )

        phi_new = float(np.exp(log_phi_new))

        if not np.isfinite(phi_new) or phi_new <= 0.0:
            raise FloatingPointError(
                "Negative-binomial "
                "log-dispersion "
                "transformation returned "
                "an invalid phi."
            )

        value, gradient = self.phi_value_and_grad(
            np.array([phi_new]), const, log_A, delta
        )
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            raise FloatingPointError(
                "Dispersion optimization returned an invalid objective."
            )

        self.phi = phi_new

    def update_after_latency_from_eta(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        """Update dispersion for a fixed incidence predictor."""
        if not self.estimate_phi:
            return

        const, log_A = self.dispersion_objective_terms_from_eta(
            latency,
            eta,
            t,
            delta,
        )

        result_phi = minimize(
            fun=self.phi_value_and_grad,
            x0=np.array(
                [self.phi],
                dtype=float,
            ),
            jac=True,
            args=(
                const,
                log_A,
                delta,
            ),
            method="L-BFGS-B",
            bounds=[
                (
                    1.0e-8,
                    None,
                )
            ],
        )

        if (
            result_phi.success
            and np.isfinite(result_phi.fun)
            and np.shape(result_phi.x) == (1,)
            and np.all(np.isfinite(result_phi.x))
            and result_phi.x[0] > 0.0
        ):
            phi_new = float(result_phi.x[0])
        else:
            self._update_dispersion_in_log_space(
                const,
                log_A,
                delta,
            )
            phi_new = self.phi

        value, gradient = self.phi_value_and_grad(
            np.array(
                [phi_new],
                dtype=float,
            ),
            const,
            log_A,
            delta,
        )

        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            raise FloatingPointError(
                "Dispersion optimization returned an invalid objective."
            )

        self.phi = phi_new

    def update_after_latency_in_log_space_from_eta(
        self,
        latency,
        eta,
        t,
        delta,
    ):
        """Update dispersion in log space for a fixed incidence predictor."""
        if not self.estimate_phi:
            return

        const, log_A = self.dispersion_objective_terms_from_eta(
            latency,
            eta,
            t,
            delta,
        )

        return self._update_dispersion_in_log_space(
            const,
            log_A,
            delta,
        )

    def finalize_fit(self):
        self.phi_ = self.phi

    def reset_for_fit(self):
        self.phi = self._initial_phi
        self.phi_ = None

    def predict_log_survival_from_eta(self, latency, eta, t_grid):
        eta = np.asarray(eta, dtype=float)
        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite negative-binomial predictor.")

        log_latency_cdf = latency_logcdf(latency, t_grid)
        if np.any(np.isnan(log_latency_cdf)) or np.any(log_latency_cdf > 0.0):
            raise FloatingPointError(
                "Invalid negative-binomial latency cumulative probability."
            )
        _, scaled_component = _nb_population_terms(
            eta[:, None] + log_latency_cdf[None, :], self.phi
        )
        return -scaled_component

    def predict_survival_from_eta(self, latency, eta, t_grid):
        with np.errstate(under="ignore"):
            return np.exp(self.predict_log_survival_from_eta(latency, eta, t_grid))

    def predict_cure_probability_from_eta(self, latency, eta):
        eta = np.asarray(eta, dtype=float)
        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite negative-binomial predictor.")

        _, scaled_component = _nb_population_terms(eta, self.phi)
        with np.errstate(under="ignore"):
            return np.exp(-scaled_component)


@dataclass
class _BinomialLatentModel(_LatentCountModel):
    K_bin: int

    name = "binomial"
    _e_step_predictor_error = "Non-finite binomial predictor."

    @staticmethod
    def _likelihood_predictor_is_invalid(f):
        return False

    @staticmethod
    def _likelihood_survival_is_invalid(latency_sf):
        return np.any(~np.isfinite(latency_sf))

    @staticmethod
    def _likelihood_component_is_invalid(log_survival_component):
        return np.any(~np.isfinite(log_survival_component))

    @staticmethod
    def _validate_e_step_survival(latency_sf):
        if np.any(~np.isfinite(latency_sf)):
            raise FloatingPointError("Non-finite latency survival probability.")

    @staticmethod
    def _conditional_fraction(log_fraction):
        return np.exp(np.minimum(log_fraction, 0.0))

    @staticmethod
    def _validate_conditional_fraction(frac):
        if np.any(~np.isfinite(frac)):
            raise FloatingPointError(
                "Non-finite binomial E-step conditional probability."
            )

    @staticmethod
    def _finalize_conditional_count(Nhat):
        if np.any(~np.isfinite(Nhat)):
            raise FloatingPointError("Non-finite binomial conditional count.")
        return Nhat

    def __init__(self, K_bin=1):
        if (
            not isinstance(
                K_bin,
                (int, np.integer),
            )
            or K_bin < 1
        ):
            raise ValueError("K_bin must be a positive integer.")

        self.K_bin = int(K_bin)

    def incidence_value_and_grad(self, eta, Nhat):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        theta_star = expit(eta)
        softplus_eta = np.logaddexp(
            0.0,
            eta,
        )

        Q = np.sum(Nhat * eta - self.K_bin * softplus_eta)
        residual = Nhat - self.K_bin * theta_star

        if not np.isfinite(Q) or np.any(~np.isfinite(residual)):
            return (
                np.inf,
                np.zeros_like(eta),
            )

        return (
            -Q,
            -residual,
        )

    @staticmethod
    def _log_theta_star(f):
        return -np.logaddexp(
            0.0,
            -f,
        )

    @staticmethod
    def _log_one_minus_theta_star(f):
        return -np.logaddexp(
            0.0,
            f,
        )

    def _log_survival_component(
        self,
        f,
        log_latency_sf,
    ):
        f, log_latency_sf = np.broadcast_arrays(f, log_latency_sf)
        result = np.empty_like(f, dtype=float)
        near_origin = log_latency_sf >= -np.log(2.0)
        # log(1 - p*F) preserves the exact S(0)=1 boundary and avoids
        # cancellation of normalized log probabilities when F is small.
        result[near_origin] = np.log1p(
            expit(f[near_origin]) * np.expm1(log_latency_sf[near_origin])
        )
        tail = ~near_origin
        result[tail] = np.logaddexp(
            self._log_one_minus_theta_star(f[tail]),
            self._log_theta_star(f[tail]) + log_latency_sf[tail],
        )
        return result

    def _observed_loglikelihood(
        self,
        f,
        delta,
        log_theta,
        latency_logpdf,
        log_survival_component,
    ):
        event_mask = delta > 0.5
        event_term = 0.0

        if np.any(event_mask):
            event_values = (
                np.log(self.K_bin) + log_theta[event_mask] + latency_logpdf[event_mask]
            )

            if np.any(~np.isfinite(event_values)):
                return -np.inf

            event_term = np.sum(event_values)

        survival_weight = self.K_bin - delta
        survival_term = np.sum(survival_weight * log_survival_component)

        return event_term + survival_term

    def observed_loglikelihood_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)) or self._likelihood_predictor_is_invalid(eta):
            return -np.inf

        log_latency_sf = latency_logsf(latency, t)

        if np.any(np.isnan(log_latency_sf)) or np.any(log_latency_sf > 0.0):
            return -np.inf

        log_theta = self._log_theta_star(eta)
        log_survival_component = self._log_survival_component(
            eta,
            log_latency_sf,
        )

        if self._likelihood_component_is_invalid(log_survival_component):
            return -np.inf

        latency_logpdf = latency.logpdf(t)
        ll = self._observed_loglikelihood(
            eta,
            delta,
            log_theta,
            latency_logpdf,
            log_survival_component,
        )

        if not np.isfinite(ll):
            return -np.inf

        return float(ll)

    def e_step_from_eta(self, latency, eta, t, delta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError(self._e_step_predictor_error)

        log_latency_sf = latency_logsf(latency, t)
        if np.any(np.isnan(log_latency_sf)) or np.any(log_latency_sf > 0.0):
            raise FloatingPointError("Non-finite latency survival probability.")

        # Conditional odds are exp(eta) * S(t). This avoids normalizing
        # already-rounded probabilities and retains e.g. sigmoid(750-750).
        frac = expit(eta + log_latency_sf)
        self._validate_conditional_fraction(frac)

        Nhat = delta + (self.K_bin - delta) * frac

        return self._finalize_conditional_count(Nhat)

    def initial_beta0(
        self,
        approximate_cure,
    ):
        theta0 = 1.0 - approximate_cure ** (1.0 / self.K_bin)
        theta0 = np.clip(
            theta0,
            1.0e-8,
            1.0 - 1.0e-8,
        )

        return float(np.log(theta0) - np.log1p(-theta0))

    def predict_log_survival_from_eta(self, latency, eta, t_grid):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite binomial predictor.")

        log_latency_sf = latency_logsf(latency, t_grid)
        if np.any(np.isnan(log_latency_sf)) or np.any(log_latency_sf > 0.0):
            raise FloatingPointError("Non-finite latency survival probability.")

        return self.K_bin * self._log_survival_component(
            eta[:, None], log_latency_sf[None, :]
        )

    def predict_survival_from_eta(self, latency, eta, t_grid):
        with np.errstate(under="ignore"):
            return np.exp(self.predict_log_survival_from_eta(latency, eta, t_grid))

    def predict_cure_probability_from_eta(self, latency, eta):
        eta = np.asarray(eta, dtype=float)

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite binomial predictor.")

        return self._cure_from_eta(
            latency,
            eta,
        )

    def _cure_from_eta(self, latency, f):
        log_one_minus_theta = self._log_one_minus_theta_star(f)
        log_cure = self.K_bin * log_one_minus_theta

        with np.errstate(
            under="ignore",
        ):
            return np.exp(log_cure)
