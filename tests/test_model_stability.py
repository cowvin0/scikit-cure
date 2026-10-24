"""Numerical regression cases for predictor fitting and public predictions."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_kernels
from sksurv.util import Surv

from skcure import GGKM
from skcure.ensemble import GradientBoostingCure
from skcure.linear_model import MixtureCure, PromotionTimeCure
import skcure.ensemble._gb as boosting_module


def _data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, len(X)), 2),
    )
    return X, y


@pytest.mark.parametrize("optimizer_fails", [False, True])
def test_boosting_step_improves_incidence_even_after_optimizer_failure(
    optimizer_fails, monkeypatch,
):
    model = GradientBoostingCure()
    model._latent_model = model._make_latent_model()
    # This is the Poisson posterior for a censored sample with S0(t)=1/2.
    eta = np.array([20.])
    expected = 0.5 * np.exp(eta)
    direction = model._functional_gradient(eta, expected)
    if optimizer_fails:
        monkeypatch.setattr(
            boosting_module, "minimize",
            lambda *args, **kwargs: SimpleNamespace(success=False, x=np.array([1.])),
        )
    rho = model._line_search_step(eta, direction, expected)
    updated = eta + model.learning_rate * rho * direction
    before = model._incidence_value_and_grad(eta, expected)[0]
    after = model._incidence_value_and_grad(updated, expected)[0]
    assert 0.0 < rho <= 10.0
    assert np.isfinite(after)
    assert after < before


def test_boosting_does_not_append_a_tree_for_a_zero_gradient():
    model = GradientBoostingCure(latent_distribution="binomial", min_samples_leaf=1)
    model._latent_model = model._make_latent_model()
    model.eta_train_ = np.zeros(4)
    model.models_ = []
    model.model_weights_ = []
    model._boosting_m_step(np.ones((4, 1)), np.full(4, 0.5))
    assert model.models_ == []
    assert model.model_weights_ == []
    np.testing.assert_array_equal(model.eta_train_, np.zeros(4))


def test_boosting_rejects_an_ascent_direction():
    model = GradientBoostingCure()
    model._latent_model = model._make_latent_model()
    eta = np.zeros(2)
    expected = np.full(2, 2.)
    _, gradient = model._incidence_value_and_grad(eta, expected)
    assert model._line_search_step(eta, gradient, expected) == 0.0


@pytest.mark.parametrize("distribution", ["poisson", "negative_binomial", "binomial"])
def test_scalar_boosting_search_matches_analytic_incidence_optimum(distribution):
    model = GradientBoostingCure(latent_distribution=distribution, K_bin=4, phi=0.7)
    model._latent_model = model._make_latent_model()
    eta = np.array([0.])
    expected = np.array([1. if distribution == "binomial" else 2.])
    direction = model._functional_gradient(eta, expected)
    rho = model._line_search_step(eta, direction, expected)
    optimum = np.log(1. / 3.) if distribution == "binomial" else np.log(2.)
    np.testing.assert_allclose(eta + rho * direction, optimum, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("estimator", [GGKM, GradientBoostingCure, PromotionTimeCure])
def test_cumulative_hazard_remains_finite_when_survival_underflows(estimator):
    X, y = _data()
    params = {"max_em_iter": 1}
    if estimator is GradientBoostingCure:
        params.update(n_estimators=1, min_samples_leaf=3, random_state=73)
    model = estimator(**params).fit(X, y)
    model._latency.set_parameters(a=1., d=1., p=1.)
    if isinstance(model, GGKM):
        model.beta0_ = 8.
        model.alpha_ = np.zeros(len(X))
    elif isinstance(model, GradientBoostingCure):
        model.initial_eta_ = np.full(len(X), 8.)
        model.models_ = []
        model.model_weights_ = []
    else:
        model._set_fitted_coefficients(np.r_[8., np.zeros(X.shape[1])])
    model.unique_times_ = np.array([1.])
    expected = np.exp(8.) * (-np.expm1(-1.))
    np.testing.assert_array_equal(model.predict_survival(X[:2], [1.]), 0.)
    np.testing.assert_allclose(model.predict_log_survival(X[:2], [1.]), -expected)
    hazards = model.predict_cumulative_hazard_function(X[:2], return_array=True)
    np.testing.assert_allclose(hazards, expected)
    for curve in model.predict_cumulative_hazard_function(X[:2]):
        assert curve(1.) == pytest.approx(expected)


@pytest.mark.parametrize("estimator", [MixtureCure, PromotionTimeCure])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_linear_fit_is_invariant_to_feature_units(estimator, fit_intercept):
    X, y = _data()
    factors = np.array([1e-6, 1e6])
    model = estimator(max_em_iter=2, fit_intercept=fit_intercept).fit(X, y)
    scaled = estimator(max_em_iter=2, fit_intercept=fit_intercept).fit(X * factors, y)
    np.testing.assert_allclose(scaled.predict(X * factors), model.predict(X), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(scaled.coef_ * factors, model.coef_, rtol=2e-5, atol=2e-6)
    assert scaled.intercept_ == pytest.approx(model.intercept_, rel=2e-5, abs=2e-6)


@pytest.mark.parametrize("kernel", ["precomputed", "callable"])
def test_indefinite_training_kernels_are_rejected(kernel):
    _, y = _data()
    gram = np.eye(len(y))
    gram[0, 0] = -1.
    if kernel == "callable":
        model = GGKM(kernel=lambda X, Y: gram, max_em_iter=1)
        X = np.zeros((len(y), 1))
    else:
        model = GGKM(kernel="precomputed", max_em_iter=1)
        X = gram
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.fit(X, y)


def test_semidefinite_kernel_allows_roundoff_negative_eigenvalues():
    X, y = _data()
    gram = pairwise_kernels(X, metric="linear")
    # Rank-deficient linear Gram matrices naturally have tiny negative computed
    # eigenvalues. They remain valid kernels and should not require a nugget.
    model = GGKM(kernel="precomputed", max_em_iter=1).fit(gram, y)
    assert np.all(np.isfinite(model.predict(gram)))


@pytest.mark.parametrize("params", [
    {"kernel": "rbf", "gamma": -1.},
    {"kernel": "exponential", "gamma": -1.},
    {"kernel": "cauchy", "gamma": -1.},
    {"kernel": "poly", "coef0": -1.},
])
def test_invalid_kernel_parameter_domains_are_rejected(params):
    with pytest.raises(ValueError, match="non-negative"):
        GGKM(**params).fit(*_data())


def test_indefinite_additive_chi_squared_kernel_is_rejected():
    X = np.array([[1.], [2.]])
    y = Surv.from_arrays([True, False], [1., 2.])
    with pytest.raises(ValueError, match="positive semidefinite"):
        GGKM(kernel="additive_chi2").fit(X, y)


def test_boosting_keeps_latency_when_optimizer_candidate_worsens_objective(monkeypatch):
    model = GradientBoostingCure()
    model._latent_model = model._make_latent_model()
    model._set_latency_parameters(1., 1., 1.)
    candidate = np.array([0.1, 1., 1.])
    monkeypatch.setattr(
        boosting_module, "minimize",
        lambda **kwargs: SimpleNamespace(
            success=True, fun=0., x=candidate, jac=np.zeros(3)
        ),
    )
    model._gg_m_step(np.array([1., 2.]), np.zeros(2), np.ones(2))
    assert (model.a_, model.d_, model.p_) == (1., 1., 1.)


def test_boosting_rechecks_a_cached_finite_gamma_objective(monkeypatch):
    model = GradientBoostingCure()
    model._latent_model = model._make_latent_model()
    model._set_latency_parameters(1., 1., 1.)
    monkeypatch.setattr(
        boosting_module, "minimize",
        lambda **kwargs: SimpleNamespace(
            success=True, fun=0., x=np.zeros(3), jac=np.zeros(3), message="stale result"
        ),
    )
    # The raw zero parameters trigger fallback; the log objective is then
    # independently reevaluated instead of trusting the cached finite value.
    monkeypatch.setattr(
        model, "_gg_log_value_and_grad",
        lambda *args: (np.inf, np.full(3, np.nan)),
    )
    with pytest.raises(RuntimeError, match="invalid objective"):
        model._gg_m_step(np.array([1., 2.]), np.zeros(2), np.ones(2))
    assert (model.a_, model.d_, model.p_) == (1., 1., 1.)
