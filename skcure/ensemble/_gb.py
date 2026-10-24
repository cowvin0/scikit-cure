"""Gradient-boosted generalized-gamma cure models."""

from math import isfinite
from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from skcure.base import BaseCureEstimator
from skcure.distributions import GeneralizedGamma
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure._latent import (
    _BinomialLatentModel,
    _LatentCountModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
    _normalize_latent_distribution,
)
from skcure.utils.numerics import _maximum_log_float, _safe_exp


class GradientBoostingCure(BaseCureEstimator):
    """Gradient-boosted generalized gamma cure model with latent counts.

    Parameters
    ----------
    latent_distribution : {"poisson", "negative_binomial", "nb", "binomial"}, default="poisson"
        Distribution of the latent number of event causes, as in ``GGKM``.
    a, d, p : float, default=1.0
        Positive initial generalized gamma parameters. Learned values are
        available as ``a_``, ``d_``, and ``p_`` after fitting.
    phi : float or None, default=None
        Negative binomial dispersion. None estimates it starting at 0.5;
        otherwise hold it fixed. Ignored by other latent distributions.
    K_bin : int, default=1
        Positive number of binomial trials. One gives Bernoulli latent counts.
        Ignored by other latent distributions.
    learning_rate : float, default=0.05
        Shrinkage applied to each tree's line-search weight, in (0, 1].
    n_estimators : int, default=100
        Maximum number of trees added per EM iteration. The complete ensemble
        can contain up to ``n_estimators * max_em_iter`` trees.
    max_depth : int or None, default=3
        Maximum depth of each regression tree.
    min_samples_leaf : int or float, default=20
        Minimum samples in a leaf, as in ``DecisionTreeRegressor``.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features considered when finding each tree split.
    random_state : int, RandomState instance or None, default=None
        Controls randomness in the regression trees.
    tol : float, default=1e-6
        Positive absolute log-likelihood tolerance for EM convergence.
    max_em_iter : int, default=1000
        Maximum number of EM iterations. Must be a positive integer.
    warm_start : bool, default=False
        Reuse fitted generalized gamma parameters on refit, and estimated
        dispersion when the latent configuration is unchanged. Every fit
        initializes a new tree ensemble.

    Attributes
    ----------
    a_, d_, p_ : float
        Learned generalized gamma parameters.
    phi_ : float
        Fitted dispersion, available for negative binomial models only.
    estimate_phi_ : bool
        Whether dispersion is estimated; negative binomial models only.
    models_ : list of DecisionTreeRegressor
        Fitted trees, in the order they were added across EM iterations.
    model_weights_ : list of float
        Learning-rate-scaled line-search weight for each fitted tree.
    n_estimators_ : int
        Total number of fitted trees across all EM iterations.
    eta_train_ : ndarray of shape (n_samples,)
        Fitted additive predictor for the training data.
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
        Observed log-likelihood after each EM iteration.
    final_loglik_ : float
        Final observed log-likelihood.

    Notes
    -----
    Trees model the incidence predictor; event times use a shared generalized
    gamma latency distribution. EM alternates tree boosting, latency fitting,
    and, for negative binomial counts with ``phi=None``, dispersion fitting.
    """

    _latency_optimizer = _GeneralizedGammaLatencyOptimizer()

    def __init__(
        self,
        latent_distribution="poisson",
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
        K_bin=1,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=20,
        max_features=None,
        random_state=None,
        tol=1e-6,
        max_em_iter=1000,
        warm_start=False,
    ):
        self.latent_distribution = latent_distribution
        self.a = a
        self.d = d
        self.p = p
        self.phi = phi
        self.K_bin = K_bin
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    @property
    def phi_(self) -> float | None:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            raise AttributeError(
                "phi_ is defined only for latent_distribution='negative_binomial'."
            )
        return latent_model.phi_

    @property
    def estimate_phi_(self) -> bool:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            raise AttributeError(
                "estimate_phi_ is defined only for "
                "latent_distribution='negative_binomial'."
            )
        return latent_model.estimate_phi

    def _gg_value_and_grad(
        self,
        pars,
        t,
        delta,
        Nhat,
    ):
        return self._latency_optimizer.value_and_grad(
            pars,
            self._latent_model,
            t,
            delta,
            Nhat,
        )

    def _gg_log_value_and_grad(
        self,
        log_pars,
        t,
        delta,
        Nhat,
    ):
        return self._latency_optimizer.log_value_and_grad(
            log_pars,
            self._latent_model,
            t,
            delta,
            Nhat,
        )

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

    def _make_latent_model(self) -> _LatentCountModel:
        distribution = _normalize_latent_distribution(self.latent_distribution)
        if distribution == "binomial":
            if isinstance(self.K_bin, (bool, np.bool_)):
                raise ValueError("K_bin must be a positive integer.")
            return _BinomialLatentModel(K_bin=self.K_bin)
        if distribution == "negative_binomial":
            if self.phi is not None and (
                not isinstance(self.phi, Real)
                or not isfinite(self.phi)
                or self.phi <= 0
            ):
                raise ValueError("phi must be positive and finite or None.")
            return _NegativeBinomialLatentModel(phi=self.phi)
        return _PoissonLatentModel()

    def _validate_boosting_parameters(self):
        if (
            not isinstance(self.learning_rate, Real)
            or not isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError("learning_rate must belong to (0, 1].")
        if (
            not isinstance(self.n_estimators, Integral)
            or isinstance(self.n_estimators, (bool, np.bool_))
            or self.n_estimators < 1
        ):
            raise ValueError("n_estimators must be a positive integer.")
        check_random_state(self.random_state)

    def _loglikelihood(self, t, delta):
        return self._latent_model.observed_loglikelihood_from_eta(
            self._latency,
            self.eta_train_,
            t,
            delta,
        )

    def _e_step(self, t, delta):
        return self._latent_model.e_step_from_eta(
            self._latency,
            self.eta_train_,
            t,
            delta,
        )

    def _incidence_value_and_grad(self, eta, Nhat):
        eta = np.asarray(eta, dtype=float)
        if np.any(~np.isfinite(eta)):
            return np.inf, np.zeros_like(eta)
        with np.errstate(over="ignore", invalid="ignore"):
            value, gradient = self._latent_model.incidence_value_and_grad(
                eta,
                Nhat,
            )
        if not np.isfinite(value) or np.any(~np.isfinite(gradient)):
            return np.inf, np.zeros_like(eta)
        return value, gradient

    def _functional_gradient(self, eta, Nhat):
        value, gradient = self._incidence_value_and_grad(eta, Nhat)
        if not np.isfinite(value):
            raise FloatingPointError(
                "Non-finite boosting incidence objective or gradient."
            )
        return -gradient

    def _line_search_step(self, eta, h, Nhat):
        initial_value, initial_gradient = self._incidence_value_and_grad(eta, Nhat)
        if not np.isfinite(initial_value) or np.any(~np.isfinite(h)):
            raise FloatingPointError("Non-finite boosting line-search inputs.")
        direction_scale = float(np.max(np.abs(h)))
        if direction_scale == 0.0:
            return 0.0
        direction = h / direction_scale
        initial_slope = float(initial_gradient @ direction)
        if not np.isfinite(initial_slope) or initial_slope >= 0.0:
            return 0.0

        # Optimize a predictor-sized displacement rather than a tree weight.
        # This keeps the scalar search well scaled even for very large trees.
        def negative_Q(displacement):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = eta + displacement[0] * direction
            value, gradient = self._incidence_value_and_grad(candidate, Nhat)
            return value, np.array([gradient @ direction], dtype=float)

        result = minimize(
            negative_Q,
            x0=np.array([min(1.0, direction_scale)]),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, min(direction_scale, np.finfo(float).max / 10.0) * 10.0)],
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 100},
        )
        displacement = float(result.x[0])
        if not result.success or not np.isfinite(displacement) or displacement <= 0:
            displacement = min(1.0, direction_scale)
        rho = min(10.0, displacement / direction_scale)

        # Check the actual shrunken update, including optimizer fallback paths.
        # A finite objective by itself does not establish a useful step.
        while rho > 0.0:
            candidate = eta + (self.learning_rate * rho) * h
            if np.array_equal(candidate, eta):
                return 0.0
            value, _ = self._incidence_value_and_grad(candidate, Nhat)
            if np.isfinite(value) and value <= initial_value:
                return rho
            rho *= 0.5
        return 0.0

    def _boosting_m_step(self, X, Nhat):
        eta = self.eta_train_.copy()
        for m in range(self.n_estimators):
            gradient = self._functional_gradient(eta=eta, Nhat=Nhat)
            seed = self.random_state
            if isinstance(seed, Integral):
                seed = (int(seed) + m) % (2**32)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(X, gradient)
            h = tree.predict(X)
            rho = self._line_search_step(eta=eta, h=h, Nhat=Nhat)
            if rho == 0.0:
                break
            step = self.learning_rate * rho
            eta_new = eta + step * h
            if np.any(~np.isfinite(eta_new)):
                raise FloatingPointError("Non-finite boosted predictor.")
            improvement = np.mean(np.abs(eta_new - eta))
            self.models_.append(tree)
            self.model_weights_.append(step)
            eta = eta_new
            if improvement < 1e-8:
                break
        self.eta_train_ = eta

    def _valid_gamma_result(self, result):
        return (
            result.success
            and np.isfinite(result.fun)
            and not self._latency_optimizer._log_objective_is_invalid(
                self._latent_model,
                result.fun,
            )
            and np.all(np.isfinite(result.x))
            and np.all(np.isfinite(getattr(result, "jac", np.zeros(3))))
        )

    def _gg_m_step(self, t, delta, Nhat):
        initial = np.array([self.a_, self.d_, self.p_])
        initial_value, initial_gradient = self._gg_value_and_grad(initial, t, delta, Nhat)
        if (
            self._latency_optimizer._log_objective_is_invalid(
                self._latent_model, initial_value
            )
            or not np.all(np.isfinite(initial_gradient))
        ):
            raise FloatingPointError("Initial generalized-gamma objective is invalid.")
        result = minimize(
            fun=self._gg_value_and_grad,
            x0=[self.a_, self.d_, self.p_],
            jac=True,
            args=(t, delta, Nhat),
            method="L-BFGS-B",
            bounds=[(1e-5, None)] * 3,
        )
        raw_valid = self._valid_gamma_result(result) and np.all(result.x > 0)
        if raw_valid:
            value, gradient = self._gg_value_and_grad(result.x, t, delta, Nhat)
            raw_valid = (
                not self._latency_optimizer._log_objective_is_invalid(
                    self._latent_model, value
                )
                and np.all(np.isfinite(gradient))
            )
        if not raw_valid:
            result = minimize(
                fun=self._gg_log_value_and_grad,
                x0=np.log([self.a_, self.d_, self.p_]),
                jac=True,
                args=(t, delta, Nhat),
                method="L-BFGS-B",
                bounds=[(np.log(1e-5), None)] * 3,
            )
            self._latency_optimizer._validate_result(
                self._latent_model,
                result,
            )
            if not self._valid_gamma_result(result):
                raise RuntimeError(
                    "Generalized-gamma optimization returned an invalid objective."
                )
            value, gradient = self._gg_log_value_and_grad(result.x, t, delta, Nhat)
            if (
                self._latency_optimizer._log_objective_is_invalid(
                    self._latent_model, value
                )
                or not np.all(np.isfinite(gradient))
            ):
                raise RuntimeError(
                    "Generalized-gamma optimization returned an invalid objective."
                )
            self._latency_optimizer._validate_log_parameters(
                self._latent_model,
                result.x,
            )
            parameters = _safe_exp(result.x)
        else:
            parameters = result.x
        if np.any(~np.isfinite(parameters)) or np.any(parameters <= 0):
            raise FloatingPointError(
                "Generalized-gamma optimization returned invalid parameters."
            )
        # Reevaluate in natural coordinates after the log transformation too;
        # cached optimizer success cannot justify a worsening EM update.
        final_value, final_gradient = self._gg_value_and_grad(parameters, t, delta, Nhat)
        if (
            self._latency_optimizer._log_objective_is_invalid(
                self._latent_model, final_value
            )
            or not np.all(np.isfinite(final_gradient))
        ):
            raise RuntimeError("Generalized-gamma optimization returned an invalid objective.")
        tolerance = 64.0 * np.finfo(float).eps * (1.0 + abs(initial_value))
        if final_value > initial_value + tolerance:
            return
        self._set_latency_parameters(*map(float, parameters))

    def _phi_m_step(self, t, delta) -> None:
        latent_model = self._latent_model
        if not isinstance(latent_model, _NegativeBinomialLatentModel):
            return

        latent_model.update_after_latency_from_eta(
            self._latency,
            self.eta_train_,
            t,
            delta,
        )

    def fit(self, X, y):
        """Fit the cure model to right-censored survival observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Numeric training features.
        y : structured ndarray of shape (n_samples,)
            A boolean event indicator followed by a strictly positive observed
            event or censoring time. At least one event must be observed.

        Returns
        -------
        self : GradientBoostingCure
            The fitted estimator.
        """
        was_fitted = self.__sklearn_is_fitted__()
        self._is_fitted = False
        self._validate_em_parameters()
        self._validate_boosting_parameters()
        latent_model = self._make_latent_model()
        X, event, t = self._validate_fit_data(X, y)
        delta = event.astype(float)
        configuration = (
            latent_model.name,
            self.phi if latent_model.name == "negative_binomial" else None,
            self.K_bin if latent_model.name == "binomial" else None,
        )
        reuse = self.warm_start and was_fitted
        if not reuse or configuration != self._latent_configuration:
            self._latent_model = latent_model
        self._latent_configuration = configuration
        if not reuse:
            self._set_latency_parameters(self.a, self.d, self.p)
        self.loglik_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        self.final_loglik_ = None
        self.X_train_ = X.copy()
        self.models_ = []
        self.model_weights_ = []
        self.initial_eta_ = np.zeros(len(t), dtype=float)
        self.eta_train_ = self.initial_eta_.copy()
        ll_old = self._loglikelihood(t, delta)
        if not np.isfinite(ll_old):
            raise FloatingPointError("Initial boosting log-likelihood is non-finite.")
        for iteration in range(self.max_em_iter):
            self.n_iter_ = iteration + 1
            Nhat = self._e_step(t, delta)
            self._boosting_m_step(X=X, Nhat=Nhat)
            self._gg_m_step(t=t, delta=delta, Nhat=Nhat)
            self._phi_m_step(t=t, delta=delta)
            ll_new = self._loglikelihood(t, delta)
            if not np.isfinite(ll_new):
                raise FloatingPointError("Non-finite boosting log-likelihood.")
            self.loglik_history_.append(float(ll_new))
            if abs(ll_new - ll_old) < self.tol:
                self.converged_ = True
                break
            ll_old = ll_new
        self.final_loglik_ = self.loglik_history_[-1]
        self.n_estimators_ = len(self.models_)
        self._latent_model.finalize_fit()
        self.unique_times_ = np.unique(t)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Return the additive predictor; larger scores indicate greater risk."""
        return self.predict_eta(X)

    def predict_eta(self, X):
        """Return the fitted additive tree predictor (larger means more risk)."""
        X = self._validate_predict_data(X)
        eta = np.full(len(X), self.initial_eta_.mean(), dtype=float)
        for tree, weight in zip(self.models_, self.model_weights_):
            eta += weight * tree.predict(X)
        if np.any(~np.isfinite(eta)):
            raise FloatingPointError("Non-finite boosted predictor.")
        return eta

    def predict_theta(self, X):
        """Return binomial cause probabilities or Poisson/NB latent means."""
        eta = self.predict_eta(X)

        if self._latent_model.name == "binomial":
            return expit(eta)

        if np.any(eta > _maximum_log_float()):
            raise FloatingPointError("Predicted latent mean exceeds double precision.")

        return _safe_exp(eta)

    def predict_survival(self, X, times):
        """Evaluate the continuous model at finite non-negative times."""
        X = self._validate_predict_data(X)
        times = self._validate_times(times)
        eta = self.predict_eta(X)

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

    def predict_cure_probability(self, X):
        """Return the limiting survival probability for each sample."""
        X = self._validate_predict_data(X)
        eta = self.predict_eta(X)

        return self._latent_model.predict_cure_probability_from_eta(
            self._latency,
            eta,
        )

