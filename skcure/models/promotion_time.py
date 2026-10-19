"""Generalized-gamma promotion-time cure model."""

from __future__ import annotations

from skcure.models._linear_cure import _BaseLinearCureModel
from skcure.models.latent import _PoissonLatentModel


class PromotionTimeModel(_BaseLinearCureModel):
    """Generalized-gamma promotion-time cure model with linear log-intensity.

    This is the linear-predictor counterpart of a Poisson GGKM/GDBT model:
    Poisson latent counts, generalized-gamma latency, and the shared latency
    optimizer are reused directly. Only the incidence predictor is linear.
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
