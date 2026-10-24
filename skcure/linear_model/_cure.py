"""Generalized gamma cure estimators with linear incidence predictors."""

from __future__ import annotations

from scipy.special import expit

from skcure._latent import _BinomialLatentModel, _PoissonLatentModel

from ._base import _BaseLinearCureModel


class MixtureCure(_BaseLinearCureModel):
    """Generalized-gamma mixture cure model with logistic susceptibility.

    One Bernoulli susceptibility indicator is represented by the shared
    binomial latent-count model with ``K_bin=1``. Generalized-gamma latency and
    its optimizer are the same components used by GGKM and GradientBoostingCure.
    Incidence optimization scales its coordinates internally; fitted
    coefficients and predictions retain the original feature units.
    """

    _latent_model = _BinomialLatentModel(K_bin=1)
    _coefficient_attribute = "beta_"
    _model_name = "mixture-cure"
    _predictor_error = "Non-finite mixture-cure predictor."
    _predictor_optimization_error = "Mixture-cure incidence optimization failed: "

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        fit_intercept=True,
        tol=1e-6,
        max_em_iter=1000,
        warm_start=False,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

    def _make_latent_model(self):
        return _BinomialLatentModel(K_bin=1)

    def predict_susceptible_probability(self, X):
        """Return the fitted susceptibility probability."""
        return expit(self.predict_eta(X))


class PromotionTimeCure(_BaseLinearCureModel):
    """Generalized-gamma promotion-time cure model with linear log-intensity.

    This is the linear-predictor counterpart of a Poisson GGKM or
    GradientBoostingCure model. Poisson latent counts, generalized-gamma latency,
    and the shared latency optimizer are reused directly.
    Incidence optimization scales its coordinates internally; fitted
    coefficients and predictions retain the original feature units.
    """

    _latent_model = _PoissonLatentModel()
    _coefficient_attribute = "alpha_"
    _model_name = "promotion-time"
    _predictor_error = "Non-finite promotion-time predictor."
    _predictor_optimization_error = "Promotion-time predictor optimization failed: "
    _initialize_intercept_from_cure = True

    def __init__(
        self,
        a=1.0,
        d=1.0,
        p=1.0,
        fit_intercept=True,
        tol=1e-6,
        max_em_iter=1000,
        warm_start=False,
    ):
        self.a = a
        self.d = d
        self.p = p
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_em_iter = max_em_iter
        self.warm_start = warm_start

    def _make_latent_model(self):
        return _PoissonLatentModel()

    def predict_theta(self, X):
        """Return the fitted Poisson latent intensity."""
        return self._latent_model.predict_theta_from_eta(self.predict_eta(X))
