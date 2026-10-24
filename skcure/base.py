"""Shared interface for scikit-survival compatible cure estimators."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,  # pyright: ignore[reportAttributeAccessIssue]
)
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.util import check_array_survival


class BaseCureEstimator(SurvivalAnalysisMixin, BaseEstimator, ABC):
    """Base class for estimators with a nonzero long-term survival probability.

    Subclasses implement fitting, risk prediction, continuous survival
    prediction, and cure probabilities. They set ``unique_times_`` to the sorted
    unique observed training times. The scikit-survival mixin supplies concordance
    scoring and estimator tags; this class supplies curve formatting and common
    validation.
    """

    unique_times_: np.ndarray
    max_em_iter: int

    @abstractmethod
    def fit(self, X, y) -> "BaseCureEstimator":
        """Fit to features and a structured boolean-event/observed-time target."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return one risk score per sample, with larger values meaning more risk."""

    @abstractmethod
    def predict_survival(self, X, times) -> np.ndarray:
        """Return continuous survival probabilities at the requested times."""

    @abstractmethod
    def predict_cure_probability(self, X) -> np.ndarray:
        """Return each sample's limiting survival probability."""

    def predict_latent_mean(self, X) -> np.ndarray:
        """Return the unconditional expected number of latent event causes.

        This is distinct from the risk score and from an E-step posterior
        expectation, which also conditions on the observed survival outcome.
        """
        raise NotImplementedError("This estimator does not expose latent counts.")

    def _validate_fit_data(self, X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(validate_data(self, X, dtype=np.float64, ensure_min_samples=2))
        event, time = check_array_survival(X, y, allow_time_zero=False)
        return X, event, time

    def _validate_em_parameters(self, *, nonnegative=(), booleans=()):
        """Validate common EM settings."""
        for name in ("a", "d", "p", "tol"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in nonnegative:
            value = getattr(self, name)
            if not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite.")
        for name in ("warm_start", *booleans):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean.")
        if (
            not isinstance(self.max_em_iter, Integral)
            or isinstance(self.max_em_iter, (bool, np.bool_))
            or self.max_em_iter < 1
        ):
            raise ValueError("max_em_iter must be a positive integer.")
        em_rtol = getattr(self, "em_rtol", 0.0)
        if (
            not isinstance(em_rtol, Real)
            or isinstance(em_rtol, (bool, np.bool_))
            or not isfinite(em_rtol)
            or em_rtol < 0.0
        ):
            raise ValueError("em_rtol must be non-negative and finite.")
        self._validate_solver_options()

    def _validate_solver_options(self):
        """Validate separate L-BFGS-B settings for the three EM M-steps."""
        settings = getattr(self, "solver_options", None)
        if settings is None:
            return
        if not isinstance(settings, Mapping):
            raise TypeError("solver_options must be a mapping or None.")
        stages = {"incidence", "latency", "dispersion"}
        if set(settings) - stages:
            raise ValueError(
                "solver_options keys must be incidence, latency, or dispersion."
            )
        integer_options = {"maxiter", "maxfun", "maxls", "maxcor"}
        real_options = {"ftol", "gtol"}
        for stage, options in settings.items():
            if not isinstance(options, Mapping):
                raise TypeError(f"solver_options[{stage!r}] must be a mapping.")
            if set(options) - integer_options - real_options:
                raise ValueError(
                    f"Unsupported {stage} solver option. Supported options are "
                    "ftol, gtol, maxiter, maxfun, maxls, and maxcor."
                )
            for name, value in options.items():
                if name in integer_options:
                    if (
                        not isinstance(value, Integral)
                        or isinstance(value, (bool, np.bool_))
                        or value < 1
                    ):
                        raise ValueError(f"{stage}.{name} must be a positive integer.")
                elif (
                    not isinstance(value, Real)
                    or isinstance(value, (bool, np.bool_))
                    or not isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(f"{stage}.{name} must be non-negative and finite.")

    def _validate_predict_data(self, X) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(validate_data(self, X, reset=False, dtype=np.float64))

    @staticmethod
    def _validate_times(times):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size == 0:
            raise ValueError("times must be a nonempty one-dimensional array.")
        if np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must contain only finite non-negative values.")
        return times

    def _format_prediction_functions(self, values, return_array):
        if not isinstance(return_array, (bool, np.bool_)):
            raise TypeError("return_array must be a boolean.")
        if return_array:
            return values
        functions = np.empty(values.shape[0], dtype=object)
        for i, row in enumerate(values):
            functions[i] = StepFunction(self.unique_times_, row)
        return functions

    def predict_log_survival(self, X, times) -> np.ndarray:
        """Return log survival probabilities at finite non-negative times.

        Estimators override this method to evaluate log survival directly,
        preserving finite cumulative hazards when survival underflows to zero.
        The default supports third-party estimators implementing only survival.
        """
        with np.errstate(divide="ignore"):
            return np.log(self.predict_survival(X, times))

    def predict_survival_function(self, X, return_array=False):
        """Predict survival functions on ``unique_times_``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        return_array : bool, default=False
            Return values on the training time grid instead of step functions.

        Returns
        -------
        prediction : ndarray
            An object array of ``sksurv.functions.StepFunction`` instances, or
            a float array of shape (n_samples, n_unique_times) when requested.

        Notes
        -----
        The step functions approximate the continuous model on the training
        time grid. Use ``predict_survival(X, times)`` for continuous predictions
        at arbitrary non-negative times, including beyond that grid.
        """
        check_is_fitted(self, "unique_times_")
        values = self.predict_survival(X, self.unique_times_)
        return self._format_prediction_functions(values, return_array)

    def predict_cumulative_hazard_function(self, X, return_array=False):
        """Predict cumulative hazards, ``-log(S(t | X))``, on ``unique_times_``.

        Parameters and return formats match ``predict_survival_function``.
        Values are evaluated from log survival so a finite cumulative hazard
        remains available when the survival probability underflows to zero.
        """
        check_is_fitted(self, "unique_times_")
        values = -self.predict_log_survival(X, self.unique_times_)
        return self._format_prediction_functions(values, return_array)


class _BaseParametricCureEstimator(BaseCureEstimator):
    """Common EM lifecycle for parametric cure estimators."""

    a: float
    d: float
    p: float
    warm_start: bool

    @abstractmethod
    def predict_eta(self, X) -> np.ndarray:
        """Return the fitted incidence predictor for each sample."""

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def _set_latency_parameters(self, a, d, p):
        """Store the current fitted latency parameters."""
        self.a_ = float(a)
        self.d_ = float(d)
        self.p_ = float(p)

    def predict(self, X) -> np.ndarray:
        """Return the linear predictor, with larger values indicating more risk."""
        return self.predict_eta(X)

    def _prepare_fit(self, X, y, *, nonnegative=(), booleans=()):
        """Validate inputs, initialize learned parameters, and reset fit history."""
        was_fitted = self.__sklearn_is_fitted__()
        self._is_fitted = False
        self.loglik_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        self.final_loglik_ = None
        self.termination_reason_ = "running"
        self.optimizer_results_ = []
        try:
            return self._prepare_fit_parameters(
                X, y, was_fitted=was_fitted, nonnegative=nonnegative, booleans=booleans
            )
        except Exception:
            self.termination_reason_ = "failed"
            raise

    def _prepare_fit_parameters(
        self, X, y, *, was_fitted, nonnegative, booleans
    ):
        """Initialize components after progress has been reset for this attempt."""
        self._validate_em_parameters(
            nonnegative=nonnegative,
            booleans=booleans,
        )
        has_phi = hasattr(self, "phi")
        phi = getattr(self, "phi", None)
        if (
            has_phi
            and phi is not None
            and (not isinstance(phi, Real) or not isfinite(phi) or phi <= 0)
        ):
            raise ValueError("phi must be positive and finite or None.")
        X, event, time = self._validate_fit_data(X, y)
        reuse = self.warm_start and was_fitted
        if not reuse:
            self._set_latency_parameters(
                self.a,
                self.d,
                self.p,
            )
        if has_phi:
            if not reuse or phi != self._phi_parameter:
                self.phi_ = 0.5 if phi is None else float(phi)
            self.estimate_phi_ = phi is None
            self._phi_parameter = phi
        return X, time, event.astype(float)

    def _finish_fit(self, time):
        """Publish the curve grid only after fitting has succeeded."""
        self.unique_times_ = np.unique(time)
        self._is_fitted = True
        return self
