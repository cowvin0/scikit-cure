"""Gradient-boosted generalized-gamma cure models."""

from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

from skcure.utils.numerics import _maximum_log_float, _safe_exp
from skcure.base import _BaseParametricCureEstimator
from skcure.distributions import GeneralizedGamma
from skcure.models._latency import _GeneralizedGammaLatencyOptimizer
from skcure.models.latent import (
    _BinomialLatentModel,
    _LatentCountModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


class _BaseGradientBoostingCure(_BaseParametricCureEstimator):
    """One EM/boosting implementation, with distribution math shared with GGKM."""

    _default_max_em_iter = 1000
    _latency_optimizer = _GeneralizedGammaLatencyOptimizer()

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
        if self._distribution == "binomial":
            if isinstance(self.K_bin, (bool, np.bool_)):
                raise ValueError("K_bin must be a positive integer.")
            return _BinomialLatentModel(K_bin=self.K_bin)
        if self._distribution == "negative_binomial":
            latent = _NegativeBinomialLatentModel(phi=self.phi)
            latent.phi = self.phi_
            return latent
        return _PoissonLatentModel()

    def _validate_boosting_parameters(self):
        if (
            not isinstance(self.learning_rate, Real)
            or not np.isfinite(self.learning_rate)
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
        def negative_Q(rho_array):
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = eta + rho_array[0] * h
            return self._incidence_value_and_grad(candidate, Nhat)[0]

        result = minimize(
            negative_Q, x0=np.array([1.0]), method="L-BFGS-B", bounds=[(0.0, 10.0)]
        )
        rho = float(result.x[0])
        if not result.success or not np.isfinite(rho) or rho <= 1e-12:
            rho = 1.0
        while rho > 1e-12 and not np.isfinite(negative_Q(np.array([rho]))):
            rho *= 0.5
        if rho <= 1e-12:
            raise FloatingPointError("No finite boosting line-search step was found.")
        return rho

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
        )

    def _gg_m_step(self, t, delta, Nhat):
        result = minimize(
            fun=self._gg_value_and_grad,
            x0=[self.a_, self.d_, self.p_],
            jac=True,
            args=(t, delta, Nhat),
            method="L-BFGS-B",
            bounds=[(1e-5, None)] * 3,
        )
        if not self._valid_gamma_result(result) or np.any(result.x <= 0):
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
        self._set_latency_parameters(*map(float, parameters))

    def _phi_m_step(self, t, delta):
        if self._distribution != "negative_binomial" or self.phi is not None:
            return

        self._latent_model.update_after_latency_from_eta(
            self._latency,
            self.eta_train_,
            t,
            delta,
        )
        self.phi_ = self._latent_model.phi

    def fit(self, X, y):
        """Fit to numeric features and a structured boolean-event/time target."""
        X, t, delta = self._prepare_fit(
            X, y, default_max_em_iter=self._default_max_em_iter
        )
        self._validate_boosting_parameters()
        self._latent_model = self._make_latent_model()
        self.X_train_ = X.copy()
        self.models_ = []
        self.model_weights_ = []
        self.initial_eta_ = np.zeros(len(t), dtype=float)
        self.eta_train_ = self.initial_eta_.copy()
        ll_old = self._loglikelihood(t, delta)
        if not np.isfinite(ll_old):
            raise FloatingPointError("Initial boosting log-likelihood is non-finite.")
        for iteration in range(self._max_em_iter):
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
        self.final_loglik_ = float(ll_new)
        self.n_estimators_ = len(self.models_)
        self._latent_model.finalize_fit()
        return self._finish_fit(t)

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

        if self._distribution == "binomial":
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

    def predict_cure_probability(self, X):
        """Return the limiting survival probability for each sample."""
        X = self._validate_predict_data(X)
        eta = self.predict_eta(X)

        return self._latent_model.predict_cure_probability_from_eta(
            self._latency,
            eta,
        )


class GGBinomialGB(_BaseGradientBoostingCure):
    """Gradient-boosted binomial generalized-gamma cure model.

    ``K_bin=1`` gives Bernoulli latent counts. ``n_estimators`` is the maximum
    number of trees added *per EM iteration*. Trees, line-search bounds,
    learning-rate shrinkage and the zero initial predictor follow the original
    boosting algorithm. Gamma parameters use GGKM's shared stable objectives.

    Fitting accepts ``fit(X, y)`` with a structured survival target. Solver
    settings belong in the constructor; learned parameters are ``a_``, ``d_``,
    and ``p_``. ``models_`` and ``model_weights_`` store the fitted ensemble.
    ``warm_start`` reuses gamma parameters and initializes a new ensemble.
    """

    _distribution = "binomial"

    def __init__(
        self,
        K_bin=1,
        a=1.0,
        d=1.0,
        p=1.0,
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
        self.K_bin = K_bin
        self.a = a
        self.d = d
        self.p = p
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

    @staticmethod
    def _theta_star(eta):
        return expit(eta)

    def _predict_theta_star_train(self):
        return self._theta_star(self.eta_train_)


class GGNBGB(_BaseGradientBoostingCure):
    """Gradient-boosted negative-binomial generalized-gamma cure model.

    ``phi=None`` estimates dispersion starting at 0.5; positive ``phi`` fixes
    it. The learned dispersion is ``phi_``. Predictor calculations use the
    true additive predictor without clipping at +/-50. Other controls and
    fitted attributes follow ``GGBinomialGB``. ``warm_start`` reuses gamma
    parameters and estimated dispersion while initializing a new ensemble.
    """

    _distribution = "negative_binomial"

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        phi=None,
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
        self.a = a
        self.d = d
        self.p = p
        self.phi = phi
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start


class GGPoissonGB(_BaseGradientBoostingCure):
    """Gradient-boosted Poisson generalized-gamma cure model.

    Parameters and fitted ensemble attributes follow ``GGBinomialGB`` without
    ``K_bin``. The original default EM limit is 10000. Survival and cure
    probabilities are computed in the log domain, without forming an
    overflowing latent intensity. ``warm_start`` initializes a new ensemble
    while reusing the fitted gamma parameters.
    """

    _distribution = "poisson"
    _default_max_em_iter = 10000

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=20,
        max_features=None,
        random_state=None,
        tol=1e-6,
        max_em_iter=10000,
        warm_start=False,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start
