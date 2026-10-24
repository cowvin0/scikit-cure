"""Generalized gamma kernel cure estimator."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

import numpy as np

from sklearn.metrics.pairwise import check_pairwise_arrays, pairwise_kernels
from skcure.base import BaseCureEstimator
from skcure.distributions import (
    GeneralizedGamma,
    validate_latency_distribution,
)
from skcure._em import fit_ggkm_em
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure._latent import (
    _BinomialLatentModel,
    _LatentCountModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
    _normalize_latent_distribution,
)
from skcure.utils.kernels import (
    _validate_kernel_hyperparameters,
    _validate_kernel_matrix,
    _validate_positive_semidefinite,
    cauchy_kernel,
    exponential_kernel,
    wavelet_kernel,
)

__all__ = ["GGKM"]


class GGKM(BaseCureEstimator):
    """Generalized gamma kernel cure model with latent count distributions.

    Parameters
    ----------
    latent_distribution : {"poisson", "negative_binomial", "nb", "binomial"}, default="poisson"
        Distribution of the latent number of event causes.
    a, d, p : float, default=1.0
        Positive initial generalized gamma parameters. Learned values are
        available as ``a_``, ``d_``, and ``p_`` after fitting.
    phi : float or None, default=None
        Negative binomial dispersion. If None, estimate it starting at 0.5;
        otherwise hold it fixed. Ignored by other latent distributions.
    K_bin : int, default=1
        Positive number of binomial trials. One gives Bernoulli latent counts.
        Ignored by other latent distributions.
    kernel : str or callable, default="rbf"
        Any kernel name accepted by ``sklearn.metrics.pairwise.pairwise_kernels``,
        or "gaussian" (an alias for "rbf"), "exponential", "cauchy", or "wavelet".
        A callable accepts two feature matrices and returns their pairwise
        kernel matrix, preserving GGKM's vectorized callable interface.
        With "precomputed", fit accepts a square training kernel matrix and
        prediction accepts a matrix of kernels between new and training samples.
        Training kernels must be positive semidefinite. Callable, precomputed,
        sigmoid, and additive-chi-squared matrices are checked explicitly,
        allowing numerical roundoff in their eigenvalues.
    kernel_params : mapping or None, default=None
        Keyword arguments for a callable kernel.
    lambda_reg : float, default=1e-3
        Non-negative coefficient of the kernel regularization penalty.
    gamma : float, default=1.0
        Non-negative scale for built-in nonlinear kernels.
    degree : int, default=3
        Positive polynomial kernel degree.
    coef0 : float, default=0.0
        Offset for polynomial and sigmoid kernels. Must be non-negative for
        polynomial kernels to preserve a positive-semidefinite kernel.
    A : float, default=10
        Positive wavelet kernel scale.
    warm_start : bool, default=False
        Reuse generalized gamma and estimated dispersion parameters on refit.
        Predictor coefficients are initialized again on each fit.
    tol : float, default=1e-6
        Positive absolute log-likelihood tolerance for EM convergence.
    max_em_iter : int, default=1000
        Maximum number of EM iterations. Must be a positive integer.

    Attributes
    ----------
    a_, d_, p_ : float
        Learned generalized gamma parameters.
    phi_ : float
        Fitted dispersion, available for negative binomial models only.
    beta0_ : float
        Fitted kernel predictor intercept.
    alpha_ : ndarray of shape (n_samples,)
        Fitted kernel predictor coefficients.
    unique_times_ : ndarray
        Sorted unique observed training times, including censoring times.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names, when all training feature names are strings.
    converged_ : bool
        Whether the EM convergence criterion was satisfied.
    n_iter_ : int
        Number of completed EM iterations.
    loglik_history_ : list of float
        Penalized log-likelihood after each EM iteration.
    final_loglik_ : float
        Final penalized log-likelihood.
    """

    _POISSON = "poisson"
    _NB = "negative_binomial"
    _BINOMIAL = "binomial"

    def _gg_log_value_and_grad(
        self,
        log_pars,
        t,
        delta,
        Nhat,
    ):
        """Compatibility callback for the GG latency optimizer."""
        return self._latency_optimizer.log_value_and_grad(
            log_pars,
            self._latent_model,
            t,
            delta,
            Nhat,
        )

    def __init__(
        self,
        latent_distribution="poisson",
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
        K_bin=1,
        kernel="rbf",
        kernel_params=None,
        lambda_reg=1e-3,
        gamma=1.0,
        degree=3,
        coef0=0.0,
        A=10,
        warm_start=False,
        tol=1e-6,
        max_em_iter=1000,
    ):
        self.latent_distribution = latent_distribution
        self.a = a
        self.d = d
        self.p = p
        self.phi = phi
        self.K_bin = K_bin
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.A = A
        self.warm_start = warm_start
        self.tol = tol
        self.max_em_iter = max_em_iter

    @classmethod
    def _normalize_latent_distribution(
        cls,
        value,
    ):
        return _normalize_latent_distribution(value)

    @classmethod
    def _make_latent_model(
        cls,
        latent_distribution,
        phi,
        K_bin,
    ) -> _LatentCountModel:
        if latent_distribution == cls._POISSON:
            return _PoissonLatentModel()

        if latent_distribution == cls._NB:
            return _NegativeBinomialLatentModel(phi=phi)

        return _BinomialLatentModel(K_bin=K_bin)

    def _reset_parameters_for_fit(self):
        latency = GeneralizedGamma(
            a=self.a,
            d=self.d,
            p=self.p,
        )
        validate_latency_distribution(latency)
        self._latency = latency

        self._latency_optimizer = _GeneralizedGammaLatencyOptimizer()
        self._latent_model.reset_for_fit()

    @property
    def a_(self) -> float:
        return self._latency.a

    @property
    def d_(self) -> float:
        return self._latency.d

    @property
    def p_(self) -> float:
        return self._latency.p

    def _reset_fit_results(self):
        """Initialize fitted result attributes for a new fit."""
        self.beta0_: float | None = None
        self.alpha_: np.ndarray | None = None
        self.K_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.loglik_history_: list[float] = []
        self.converged_ = False
        self.n_iter_ = 0
        self.final_loglik_: float | None = None

    @property
    def estimate_phi(self) -> bool:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            raise AttributeError(
                "estimate_phi is defined only for "
                "latent_distribution='negative_binomial'."
            )

        return latent_model.estimate_phi

    @property
    def phi_(self) -> float | None:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            raise AttributeError(
                "phi_ is defined only for " "latent_distribution='negative_binomial'."
            )

        return latent_model.phi_

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _compute_kernel(self, X1, X2) -> np.ndarray:
        """Compute validated training or new-versus-training kernel matrices."""
        X1, X2 = check_pairwise_arrays(
            X1, X2, dtype="float64", precomputed=self.kernel == "precomputed"
        )
        if callable(self.kernel):
            K = self.kernel(X1, X2, **(self.kernel_params or {}))
        elif self.kernel == "exponential":
            K = exponential_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == "cauchy":
            K = cauchy_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == "wavelet":
            K = wavelet_kernel(X1, X2, A=self.A)
        else:
            metric = "rbf" if self.kernel == "gaussian" else self.kernel
            K = pairwise_kernels(
                X1,
                X2,
                metric=metric,
                filter_params=True,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
            )
        return _validate_kernel_matrix(K, (X1.shape[0], X2.shape[0]))

    def _validate_parameters_for_fit(self) -> _LatentCountModel:
        """Validate current parameters, including changes from set_params."""
        self._validate_em_parameters(nonnegative=("lambda_reg",))
        if not isinstance(self.kernel, str) and not callable(self.kernel):
            raise TypeError("kernel must be a string or a callable.")
        if self.kernel_params is not None and not isinstance(
            self.kernel_params, Mapping
        ):
            raise TypeError("kernel_params must be a mapping or None.")
        if isinstance(self.kernel, str) and self.kernel_params:
            raise ValueError(
                "kernel_params can only be supplied when kernel is callable."
            )
        _validate_kernel_hyperparameters(
            self.gamma, self.degree, self.coef0, self.A, kernel=self.kernel
        )
        distribution = self._normalize_latent_distribution(self.latent_distribution)
        if (
            distribution == self._NB
            and self.phi is not None
            and (
                not isinstance(self.phi, Real)
                or not isfinite(self.phi)
                or self.phi <= 0
            )
        ):
            raise ValueError("phi must be positive and finite or None.")
        if distribution == self._BINOMIAL and isinstance(self.K_bin, (bool, np.bool_)):
            raise ValueError("K_bin must be a positive integer.")
        latent_model = self._make_latent_model(
            distribution,
            phi=self.phi,
            K_bin=self.K_bin,
        )

        return latent_model

    def _predictor(
        self,
        beta0,
        alpha,
    ) -> np.ndarray:
        assert self.K_ is not None
        return beta0 + self.K_ @ alpha

    def _compute_prediction_kernel(self, X_new) -> np.ndarray:
        """Compute kernels between prediction samples and the training samples."""
        assert self.X_train_ is not None
        return self._compute_kernel(
            X_new,
            self.X_train_,
        )

    def _predictor_from_kernel(self, K_new) -> np.ndarray:
        """Evaluate the fitted kernel predictor from a prediction kernel matrix."""
        assert self.beta0_ is not None
        assert self.alpha_ is not None
        return self.beta0_ + K_new @ self.alpha_

    def _predict_eta_from_validated_X(self, X_new) -> np.ndarray:
        """Evaluate the fitted kernel predictor with legacy family validation."""
        if self._latent_model.name == self._POISSON:
            X_new = np.asarray(
                X_new,
                dtype=float,
            )

            K_new = self._compute_prediction_kernel(
                X_new,
            )

            eta = self._predictor_from_kernel(
                K_new,
            )

            if np.any(~np.isfinite(eta)):
                raise FloatingPointError("Non-finite Poisson predictor.")

            return eta

        if self._latent_model.name == self._NB:
            K_new = self._compute_prediction_kernel(
                X_new,
            )

            return self._predictor_from_kernel(
                K_new,
            )

        X_new = np.asarray(
            X_new,
            dtype=float,
        )

        if X_new.ndim != 2 or np.any(~np.isfinite(X_new)):
            raise ValueError("X_new must be a finite " "two-dimensional array.")

        K_new = self._compute_prediction_kernel(
            X_new,
        )

        if np.any(~np.isfinite(K_new)):
            raise FloatingPointError(
                "Prediction kernel matrix contains " "non-finite values."
            )

        eta = self._predictor_from_kernel(
            K_new,
        )

        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite binomial predictor.")

        return eta

    def _kernel_penalty(self, alpha):
        assert self.K_ is not None
        K_alpha = self.K_ @ alpha
        penalty = (self.lambda_reg / 2.0) * (alpha @ K_alpha)
        return K_alpha, penalty

    def _kernel_predictor_gradient(
        self,
        score,
        K_alpha,
    ):
        assert self.K_ is not None
        grad_beta0 = np.sum(score)
        grad_alpha = self.K_.T @ score - self.lambda_reg * K_alpha
        grad = np.concatenate(
            (
                np.array(
                    [grad_beta0],
                    dtype=float,
                ),
                grad_alpha,
            )
        )

        return grad

    def _penalized_loglikelihood(
        self,
        beta0,
        alpha,
        t,
        delta,
    ):
        eta = self._predictor(
            beta0,
            alpha,
        )

        ll = self._latent_model.observed_loglikelihood_from_eta(
            self._latency,
            eta,
            t,
            delta,
        )

        _, penalty = self._kernel_penalty(
            alpha,
        )

        value = ll - penalty

        if self._latent_model.name == self._NB:
            return value

        if not np.isfinite(value):
            return -np.inf

        return float(value)

    def _e_step(
        self,
        beta0,
        alpha,
        t,
        delta,
    ):
        eta = self._predictor(
            beta0,
            alpha,
        )

        return self._latent_model.e_step_from_eta(
            self._latency,
            eta,
            t,
            delta,
        )

    def _predictor_value_and_grad(
        self,
        pars,
        Nhat,
    ):
        beta0 = float(pars[0])
        alpha = pars[1:]

        eta = self._predictor(
            beta0,
            alpha,
        )

        incidence_value, incidence_grad = self._latent_model.incidence_value_and_grad(
            eta,
            Nhat,
        )

        if (
            not np.isfinite(incidence_value) or np.any(~np.isfinite(incidence_grad))
        ):
            return (
                np.inf,
                np.zeros_like(pars),
            )

        Q = -incidence_value
        score = -incidence_grad

        K_alpha, penalty = self._kernel_penalty(
            alpha,
        )

        Q -= penalty

        grad = self._kernel_predictor_gradient(
            score,
            K_alpha,
        )

        if (
            not np.isfinite(Q) or np.any(~np.isfinite(grad))
        ):
            return (
                np.inf,
                np.zeros_like(pars),
            )

        return (
            -Q,
            -grad,
        )

    def _validate_predictor_result(self, result):
        if self._latent_model.name == self._NB:
            if (
                not result.success
                or not np.isfinite(result.fun)
                or not np.all(np.isfinite(result.x))
            ):
                raise RuntimeError(
                    "Negative-binomial predictor "
                    "optimization failed: "
                    f"{result.message}"
                )
            return

        if (
            not result.success
            or not np.isfinite(result.fun)
            or not np.all(np.isfinite(result.x))
        ):
            if self._latent_model.name == self._POISSON:
                prefix = "Poisson predictor optimization failed: "
            else:
                prefix = "Binomial predictor optimization failed: "

            raise RuntimeError(prefix + f"{result.message}")

    def _validate_initial_loglik(self, ll_old):
        if not np.isfinite(ll_old):
            raise FloatingPointError("Initial penalized log-likelihood is non-finite.")

    def _update_phi(
        self,
        beta0,
        alpha,
        t,
        delta,
    ) -> None:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            return None

        eta = self._predictor(
            beta0,
            alpha,
        )

        return latent_model.update_after_latency_in_log_space_from_eta(
            self._latency,
            eta,
            t,
            delta,
        )

    def _initial_beta0(
        self,
        approximate_cure,
    ):
        return self._latent_model.initial_beta0(approximate_cure)

    def fit(self, X, y) -> GGKM:
        """Fit the cure model to right-censored survival observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric training features. For ``kernel="precomputed"``, a square
            kernel matrix of shape (n_samples, n_samples).
        y : structured ndarray of shape (n_samples,)
            Two fields: a boolean event indicator followed by a strictly
            positive observed event or censoring time. Field names are arbitrary.
            At least one event must be observed.

        Returns
        -------
        self : GGKM
            The fitted estimator.
        """
        was_fitted = self.__sklearn_is_fitted__()
        self._is_fitted = False
        latent_model = self._validate_parameters_for_fit()
        X, event, t = self._validate_fit_data(X, y)
        delta = event.astype(float)
        configuration = (
            latent_model.name,
            self.phi if latent_model.name == self._NB else None,
            self.K_bin if latent_model.name == self._BINOMIAL else None,
        )
        reuse = self.warm_start and was_fitted
        if not reuse or configuration != self._latent_configuration:
            self._latent_model = latent_model
        self._latent_configuration = configuration
        self._reset_fit_results()
        if not reuse:
            self._reset_parameters_for_fit()

        self.X_train_ = X.copy()
        self.K_ = self._compute_kernel(
            X,
            X,
        )

        if not np.allclose(
            self.K_,
            self.K_.T,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                "The kernel must produce a symmetric " "training kernel matrix."
            )
        if callable(self.kernel) or self.kernel in {"precomputed", "sigmoid", "additive_chi2"}:
            _validate_positive_semidefinite(self.K_)
        fit_ggkm_em(self, t, delta)
        self.unique_times_ = np.unique(t)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Return the linear predictor; larger scores indicate greater risk.

        With ``kernel="precomputed"``, X has shape (n_samples, n_train_samples)
        and contains kernels between the new and original training samples.
        """
        return self.predict_eta(X)

    def predict_eta(self, X) -> np.ndarray:
        """Return the fitted kernel predictor for each sample."""
        X = self._validate_predict_data(X)
        return self._predict_eta_from_validated_X(X)

    def predict_theta(self, X) -> np.ndarray:
        """Return Poisson latent intensities (Poisson models only)."""
        X = self._validate_predict_data(X)

        if self._latent_model.name != self._POISSON:
            raise RuntimeError(
                "predict_theta is defined only for latent_distribution='poisson'."
            )

        eta = self._predict_eta_from_validated_X(X)
        return self._latent_model.predict_theta_from_eta(eta)

    def predict_survival(self, X, times) -> np.ndarray:
        """Return continuous survival probabilities of shape (n_samples, n_times).

        Unlike the step function API, this evaluates the parametric model
        directly at any finite non-negative times, including after follow-up.
        """
        X = self._validate_predict_data(X)
        times = self._validate_times(times)

        eta = self._predict_eta_from_validated_X(X)

        return self._latent_model.predict_survival_from_eta(
            self._latency,
            eta,
            times,
        )

    def predict_log_survival(self, X, times) -> np.ndarray:
        """Evaluate log survival without first rounding survival probabilities."""
        eta = self.predict_eta(X)
        times = self._validate_times(times)
        return self._latent_model.predict_log_survival_from_eta(
            self._latency, eta, times
        )

    def predict_cure_probability(self, X) -> np.ndarray:
        """Return the limiting survival probability for each sample."""
        X = self._validate_predict_data(X)

        eta = self._predict_eta_from_validated_X(X)

        return self._latent_model.predict_cure_probability_from_eta(
            self._latency,
            eta,
        )
