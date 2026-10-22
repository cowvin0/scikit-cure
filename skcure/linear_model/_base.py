"""Shared linear-predictor cure-model infrastructure."""

from __future__ import annotations

import numpy as np

from skcure.base import _BaseParametricCureEstimator
from skcure.distributions import GeneralizedGamma
from skcure._em import fit_linear_cure_em
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure._latent import _LatentCountModel


class _BaseLinearCureModel(_BaseParametricCureEstimator):
    """Base class for generalized-gamma cure models with a linear incidence predictor.

    Subclasses supply only the latent-count distribution and public naming.
    Likelihood evaluation, E-step calculations, incidence optimization,
    generalized-gamma latency optimization, and population predictions are
    delegated to the same components used by GGKM and GradientBoostingCure.
    """

    _latency_optimizer = _GeneralizedGammaLatencyOptimizer()
    _coefficient_attribute: str
    _model_name: str
    _predictor_error: str
    _predictor_optimization_error: str
    _initialize_intercept_from_cure = False

    def _make_latent_model(self) -> _LatentCountModel:
        raise NotImplementedError

    @property
    def a_(self):
        return self._latency.a

    @property
    def d_(self):
        return self._latency.d

    @property
    def p_(self):
        return self._latency.p

    def _set_latency_parameters(self, a, d, p):
        """Initialize or update the shared generalized-gamma latency object."""
        if hasattr(self, "_latency"):
            self._latency.set_parameters(
                a=float(a),
                d=float(d),
                p=float(p),
            )
        else:
            self._latency = GeneralizedGamma(
                a=float(a),
                d=float(d),
                p=float(p),
            )

    def _reset_components_for_fit(self):
        # _prepare_fit() owns latency initialization and warm-start selection
        # through _set_latency_parameters(). Recreate only the stateless/count
        # component needed by this linear estimator.
        self._latent_model = self._make_latent_model()
        self._latent_model.reset_for_fit()

    def _design_matrix(self, X):
        if self.fit_intercept:
            return np.column_stack(
                (
                    np.ones(
                        X.shape[0],
                        dtype=float,
                    ),
                    X,
                )
            )

        return X

    def _initial_coefficients(self, delta):
        coefficients = np.zeros(
            self.W_train_.shape[1],
            dtype=float,
        )

        if self.fit_intercept and self._initialize_intercept_from_cure:
            approximate_cure = np.clip(
                1.0 - np.mean(delta),
                0.05,
                0.95,
            )
            coefficients[0] = self._latent_model.initial_beta0(
                approximate_cure,
            )

        return coefficients

    def _set_fitted_coefficients(self, coefficients):
        coefficients = np.asarray(
            coefficients,
            dtype=float,
        ).copy()

        setattr(
            self,
            self._coefficient_attribute,
            coefficients,
        )

        if self.fit_intercept:
            self.intercept_ = float(coefficients[0])
            self.coef_ = coefficients[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = coefficients.copy()

    def _fitted_coefficients(self):
        return getattr(
            self,
            self._coefficient_attribute,
        )

    def _predictor_from_coefficients(self, coefficients):
        return self.W_train_ @ coefficients

    def _loglikelihood(self, coefficients, t, delta):
        eta = self._predictor_from_coefficients(coefficients)

        return self._latent_model.observed_loglikelihood_from_eta(
            self._latency,
            eta,
            t,
            delta,
        )

    def _e_step(self, coefficients, t, delta):
        eta = self._predictor_from_coefficients(coefficients)

        return self._latent_model.e_step_from_eta(
            self._latency,
            eta,
            t,
            delta,
        )

    def _predictor_value_and_grad(self, coefficients, expected):
        eta = self._predictor_from_coefficients(coefficients)

        value, gradient_eta = self._latent_model.incidence_value_and_grad(
            eta,
            expected,
        )

        return (
            value,
            self.W_train_.T @ gradient_eta,
        )

    def _gg_log_value_and_grad(self, log_pars, t, delta, expected):
        return self._latency_optimizer.log_value_and_grad(
            log_pars,
            self._latent_model,
            t,
            delta,
            expected,
        )

    def _update_after_latency(
        self,
        coefficients,
        t,
        delta,
    ):
        """Update optional latent parameters after the latency M-step."""
        return None

    def _validate_predictor_result(self, result):
        if (
            not result.success
            or not np.isfinite(result.fun)
            or np.any(~np.isfinite(result.x))
        ):
            raise RuntimeError(f"{self._predictor_optimization_error}{result.message}")

    def _validate_initial_loglik(self, value):
        if not np.isfinite(value):
            raise FloatingPointError(
                f"Initial {self._model_name} log-likelihood is non-finite."
            )

    def fit(self, X, y):
        """Fit using a structured boolean-event/observed-time target."""
        X, t, delta = self._prepare_fit(
            X,
            y,
            booleans=("fit_intercept",),
        )

        self._reset_components_for_fit()
        self.X_train_ = X
        self.W_train_ = self._design_matrix(X)

        fit_linear_cure_em(
            self,
            t,
            delta,
        )

        self._finish_fit(t)
        return self

    def predict_eta(self, X):
        """Return the fitted linear incidence predictor."""
        X = self._validate_predict_data(X)

        eta = self._design_matrix(X) @ self._fitted_coefficients()

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError(self._predictor_error)

        return eta

    def predict_survival(self, X, times):
        """Evaluate population survival at finite non-negative times."""
        X = self._validate_predict_data(X)
        times = self._validate_times(times)

        eta = self._design_matrix(X) @ self._fitted_coefficients()

        return self._latent_model.predict_survival_from_eta(
            self._latency,
            eta,
            times,
        )

    def predict_cure_probability(self, X):
        """Return the limiting population survival probability."""
        X = self._validate_predict_data(X)

        eta = self._design_matrix(X) @ self._fitted_coefficients()

        return self._latent_model.predict_cure_probability_from_eta(
            self._latency,
            eta,
        )
