"""Generalized-gamma mixture cure estimator with logistic susceptibility."""

from __future__ import annotations

from scipy.special import expit

from skcure.models._linear_cure import _BaseLinearCureModel
from skcure.models.latent import _BinomialLatentModel


class MixtureCureModel(_BaseLinearCureModel):
    """Generalized-gamma mixture cure model with logistic susceptibility.

    One Bernoulli susceptibility indicator is represented by the shared
    binomial latent-count model with ``K_bin=1``. Generalized-gamma latency and
    its optimizer are the same components used by GGKM and GDBT.
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
