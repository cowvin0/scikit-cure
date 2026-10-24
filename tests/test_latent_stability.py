"""Numerical limits of the shared fixed-predictor count families."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skcure._latent import (
    _BinomialLatentModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


class _ExponentialLatency:
    def pdf(self, t):
        return np.exp(self.logpdf(t))

    def logpdf(self, t):
        return -np.asarray(t, dtype=float)

    def cdf(self, t):
        return -np.expm1(-np.asarray(t, dtype=float))

    def logcdf(self, t):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(t))

    def sf(self, t):
        return np.exp(self.logsf(t))

    def logsf(self, t):
        return -np.asarray(t, dtype=float)


def test_poisson_likelihood_does_not_evaluate_density_for_censored_rows():
    class EventOnlyLatency(_ExponentialLatency):
        def logpdf(self, t):
            assert np.all(t < 700.0)
            return super().logpdf(t)

    latent = _PoissonLatentModel()
    t = np.array([1.0, 1000.0])
    actual = latent.observed_loglikelihood_from_eta(
        EventOnlyLatency(), np.zeros(2), t, np.array([1.0, 0.0])
    )
    assert_allclose(actual, -1.0 - np.sum(-np.expm1(-t)))


def test_poisson_likelihood_skips_density_for_an_entirely_censored_batch():
    class NoDensityLatency(_ExponentialLatency):
        def logpdf(self, t):
            raise AssertionError("The density should not be evaluated.")

    actual = _PoissonLatentModel().observed_loglikelihood_from_eta(
        NoDensityLatency(), np.zeros(2), np.array([1.0, 1000.0]), np.zeros(2)
    )
    assert_allclose(actual, -2.0 + np.exp(-1.0))


def test_poisson_log_nonnegative_preserves_nan():
    assert np.isnan(_PoissonLatentModel().log_nonnegative(np.array([np.nan]))[0])


@pytest.mark.parametrize("factory", [_PoissonLatentModel, _NegativeBinomialLatentModel])
def test_censored_invalid_latency_is_not_silently_replaced_by_zero(factory):
    class InvalidLatency(_ExponentialLatency):
        def logcdf(self, t):
            return np.full_like(t, np.nan)

    assert (
        factory().observed_loglikelihood_from_eta(
            InvalidLatency(), np.array([0.0]), np.array([1.0]), np.array([0.0])
        )
        == -np.inf
    )


@pytest.mark.parametrize("factory", [_PoissonLatentModel, _BinomialLatentModel])
def test_conditional_count_retains_cancelling_extreme_log_quantities(factory):
    latent = factory()
    count = latent.e_step_from_eta(
        _ExponentialLatency(), np.array([750.0]), np.array([750.0]), np.array([0.0])
    )
    expected = 1.0 if isinstance(latent, _PoissonLatentModel) else 0.5
    assert_allclose(count, [expected], rtol=0.0, atol=0.0)


def test_poisson_log_survival_remains_finite_after_survival_underflows():
    latent = _PoissonLatentModel()
    log_survival = latent.predict_log_survival_from_eta(
        _ExponentialLatency(), np.array([8.0]), np.array([1.0])
    )
    assert_allclose(log_survival, [[-np.exp(8.0) * -np.expm1(-1.0)]])
    assert np.isfinite(log_survival[0, 0])
    assert (
        latent.predict_survival_from_eta(
            _ExponentialLatency(), np.array([8.0]), np.array([1.0])
        )[0, 0]
        == 0.0
    )


@pytest.mark.parametrize("phi", [1e-12, 1e-100, np.nextafter(0.0, 1.0)])
def test_negative_binomial_incidence_has_finite_poisson_limit(phi):
    eta = np.array([-3.0, 0.0, 1.0])
    expected_count = np.array([0.2, 1.5, 2.0])
    expected = _PoissonLatentModel().incidence_value_and_grad(eta, expected_count)
    actual = _NegativeBinomialLatentModel(phi).incidence_value_and_grad(
        eta, expected_count
    )
    assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-12)
    assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("phi", [1e-12, 1e-100, np.nextafter(0.0, 1.0)])
def test_negative_binomial_likelihood_and_prediction_have_poisson_limit(phi):
    eta = np.array([-1.0, 0.0, 1.0])
    t = np.array([0.1, 1.0, 10.0])
    delta = np.array([1.0, 0.0, 1.0])
    poisson = _PoissonLatentModel()
    nb = _NegativeBinomialLatentModel(phi)
    latency = _ExponentialLatency()
    assert_allclose(
        nb.observed_loglikelihood_from_eta(latency, eta, t, delta),
        poisson.observed_loglikelihood_from_eta(latency, eta, t, delta),
        rtol=1e-11,
    )
    assert_allclose(
        nb.predict_log_survival_from_eta(latency, eta, t),
        poisson.predict_log_survival_from_eta(latency, eta, t),
        rtol=1e-11,
    )
    assert_allclose(
        nb.predict_cure_probability_from_eta(latency, eta),
        poisson.predict_cure_probability_from_eta(latency, eta),
        rtol=1e-11,
    )


@pytest.mark.parametrize("phi", [1e-8, 1e-16, 1e-100, np.nextafter(0.0, 1.0)])
def test_negative_binomial_phi_derivative_uses_analytic_poisson_limit(phi):
    A = np.array([0.5, 1.0, 3.0])
    delta = np.array([0.0, 1.0, 1.0])
    value, grad = _NegativeBinomialLatentModel(phi).phi_value_and_grad(
        np.array([phi]), np.zeros(3), np.log(A), delta
    )
    assert np.isfinite(value)
    expected = np.sum(delta * A - 0.5 * A**2)
    assert_allclose(grad, [expected], rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("phi", [1e-3, 0.5, 30.0])
def test_negative_binomial_phi_derivative_matches_finite_differences(phi):
    latent = _NegativeBinomialLatentModel(phi)
    arguments = (
        np.array([0.0, -0.7, 0.8]),
        np.log([0.1, 1.0, 20.0]),
        np.array([0.0, 1.0, 1.0]),
    )
    step = phi * 1e-4
    value, grad = latent.phi_value_and_grad(np.array([phi]), *arguments)
    numerical = (
        latent.phi_value_and_grad(np.array([phi + step]), *arguments)[0]
        - latent.phi_value_and_grad(np.array([phi - step]), *arguments)[0]
    ) / (2.0 * step)
    assert np.isfinite(value)
    assert_allclose(grad, [numerical], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("phi", [1e-3, 0.5, 30.0])
def test_negative_binomial_incidence_gradient_matches_finite_differences(phi):
    latent = _NegativeBinomialLatentModel(phi)
    eta = np.array([-3.0, 0.1, 1.0, 3.0])
    counts = np.array([0.2, 1.0, 3.0, 8.0])
    _, gradient = latent.incidence_value_and_grad(eta, counts)
    numerical = np.empty_like(eta)
    for j in range(len(eta)):
        step = np.zeros_like(eta)
        step[j] = 1e-5
        numerical[j] = (
            latent.incidence_value_and_grad(eta + step, counts)[0]
            - latent.incidence_value_and_grad(eta - step, counts)[0]
        ) / 2e-5
    assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("phi", [1e-3, 0.5, 30.0])
def test_negative_binomial_likelihood_and_e_step_match_direct_regular_formula(phi):
    latent = _NegativeBinomialLatentModel(phi)
    latency = _ExponentialLatency()
    eta = np.array([-3.0, 0.1, 1.0, 3.0])
    t = np.array([0.01, 0.3, 2.0, 5.0])
    delta = np.array([1.0, 0.0, 1.0, 0.0])
    theta = np.exp(eta)
    expected_ll = np.sum(
        delta * (eta - t) - (1.0 / phi + delta) * np.log1p(phi * theta * latency.cdf(t))
    )
    expected_count = delta + (1.0 + phi * delta) * theta * latency.sf(t) / (
        1.0 + phi * theta * latency.cdf(t)
    )
    assert_allclose(
        latent.observed_loglikelihood_from_eta(latency, eta, t, delta),
        expected_ll,
        rtol=1e-14,
    )
    assert_allclose(
        latent.e_step_from_eta(latency, eta, t, delta), expected_count, rtol=1e-14
    )


def test_negative_binomial_retains_finite_huge_predictor_log_survival():
    latent = _NegativeBinomialLatentModel(phi=1.0)
    actual = latent.predict_log_survival_from_eta(
        _ExponentialLatency(), np.array([1e300]), np.array([1.0])
    )
    assert_allclose(actual, [[-1e300]], rtol=1e-13)


@pytest.mark.parametrize("invalid_value", [np.inf, np.nan])
def test_negative_binomial_rejects_invalid_log_dispersion_optimizer_result(
    monkeypatch, invalid_value
):
    latent = _NegativeBinomialLatentModel()
    original_phi = latent.phi

    def controlled_minimize(**kwargs):
        return SimpleNamespace(
            success=True,
            fun=invalid_value,
            x=np.array([np.log(1.2)]),
            message="invalid optimizer objective",
        )

    monkeypatch.setattr("skcure._latent.minimize", controlled_minimize)
    with pytest.raises(RuntimeError, match="log-dispersion"):
        latent._update_dispersion_in_log_space(np.zeros(1), np.zeros(1), np.zeros(1))
    assert latent.phi == original_phi


def test_negative_binomial_keeps_old_phi_when_candidate_reevaluation_is_invalid(
    monkeypatch,
):
    latent = _NegativeBinomialLatentModel()
    original_phi = latent.phi

    def controlled_minimize(**kwargs):
        return SimpleNamespace(
            success=True, fun=1.0, x=np.array([1.2]), message="reported success"
        )

    monkeypatch.setattr("skcure._latent.minimize", controlled_minimize)
    monkeypatch.setattr(
        latent, "phi_value_and_grad", lambda *args: (np.inf, np.array([np.nan]))
    )
    with pytest.raises(FloatingPointError, match="invalid objective"):
        latent.update_after_latency_from_eta(
            _ExponentialLatency(), np.zeros(1), np.ones(1), np.zeros(1)
        )
    assert latent.phi == original_phi


@pytest.mark.parametrize(
    "factory", [_PoissonLatentModel, _NegativeBinomialLatentModel, _BinomialLatentModel]
)
def test_log_survival_matches_survival_at_regular_values(factory):
    latent = factory()
    arguments = (
        _ExponentialLatency(),
        np.array([-2.0, 0.1, 1.0]),
        np.array([0.0, 0.2, 2.0]),
    )
    log_survival = latent.predict_log_survival_from_eta(*arguments)
    assert_allclose(np.exp(log_survival), latent.predict_survival_from_eta(*arguments))
    assert_allclose(log_survival[:, 0], 0.0, atol=1e-16)
