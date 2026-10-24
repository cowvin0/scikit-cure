from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import gammaln

import skcure._latency as latency_module
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure._latent import (
    _BinomialLatentModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)
from skcure.distributions import GeneralizedGamma
from skcure.distributions.generalized_gamma import _gamma_upper_tail_terms


@pytest.fixture(params=["poisson", "negative_binomial", "binomial"])
def latent_model(request):
    if request.param == "poisson":
        return _PoissonLatentModel()
    if request.param == "negative_binomial":
        return _NegativeBinomialLatentModel(phi=0.5)
    return _BinomialLatentModel(K_bin=2)


def test_exponential_log_survival_survives_probability_underflow():
    latency = GeneralizedGamma()
    time = np.array([720.0, 1000.0, 1.0e6])
    np.testing.assert_allclose(latency.logsf(time), -time, rtol=2e-14)
    assert latency.sf(time)[0] > 0.0
    np.testing.assert_allclose(latency.sf(time)[0], np.exp(-720.0), rtol=1e-10, atol=0)


def test_erlang_log_probabilities_match_closed_form_in_both_tails():
    latency = GeneralizedGamma(d=2.0)
    time = np.array([720.0, 1000.0, 1.0e6])
    np.testing.assert_allclose(latency.logsf(time), np.log1p(time) - time, rtol=2e-14)
    small_time = np.array([1.0e-200, 1.0e-300])
    np.testing.assert_allclose(
        latency.logcdf(small_time), 2.0 * np.log(small_time) - np.log(2.0), rtol=2e-14
    )


def test_log_cdf_retains_argument_that_underflows_during_transformation():
    latency = GeneralizedGamma(d=2.0, p=2.0)
    time = np.array([1.0e-200, 1.0e-300])
    np.testing.assert_allclose(latency.logcdf(time), 2.0 * np.log(time), rtol=2e-14)
    # With a small gamma shape, a tiny transformed argument can still have
    # substantial CDF and survival probabilities.
    latency = GeneralizedGamma(d=1.0e-4, p=1000.0)
    expected_logcdf = latency.d * np.log(time) - gammaln(1.0 + latency.d / latency.p)
    np.testing.assert_allclose(latency.logcdf(time), expected_logcdf, rtol=1e-12)
    np.testing.assert_allclose(latency.cdf(time), np.exp(expected_logcdf), rtol=1e-12)
    np.testing.assert_allclose(latency.sf(time), -np.expm1(expected_logcdf), rtol=1e-12)


@pytest.mark.parametrize("d,expected", [(0.5, np.inf), (1.0, 1.0), (2.0, 0.0)])
def test_generalized_gamma_density_has_analytic_zero_boundary(d, expected):
    latency = GeneralizedGamma(d=d)
    assert latency.pdf(0.0) == expected
    assert latency.pdf(np.inf) == 0.0
    assert latency.logpdf(np.inf) == -np.inf
    assert GeneralizedGamma(a=2.0, d=1.0, p=2.0).logpdf(0.0) == pytest.approx(
        -gammaln(0.5)
    )


@pytest.mark.parametrize("method", ["pdf", "logpdf", "cdf", "sf", "logcdf", "logsf"])
def test_generalized_gamma_preserves_nan_and_invalid_negative_times(method):
    result = getattr(GeneralizedGamma(), method)([np.nan, -1.0])
    assert np.isnan(result).all()


@pytest.mark.parametrize("pars", [[1.0, 1.0, 1.0], [1.1, 2.3, 0.9], [0.7, 0.4, 1.2]])
def test_latency_tail_gradient_matches_numerical_derivative(latent_model, pars):
    optimizer = _GeneralizedGammaLatencyOptimizer()
    pars = np.asarray(pars)
    time = np.array([1000.0, 2000.0])
    delta = np.zeros(2)
    expected = np.array([0.5, 1.2])
    objective = lambda candidate: optimizer.value_and_grad(
        candidate, latent_model, time, delta, expected
    )
    value, gradient = objective(pars)
    numerical = approx_derivative(
        lambda candidate: objective(candidate)[0], pars
    ).ravel()
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-7, atol=2e-7)


def test_upper_tail_density_ratio_avoids_cancellation_of_large_logs():
    u = np.array([1.0e30, 1.0e100])
    _, _, ratio = _gamma_upper_tail_terms(1.0, u, np.log(u))
    np.testing.assert_allclose(ratio, u, rtol=1e-14)


def test_latency_gradient_retains_tiny_argument_with_substantial_cdf(latent_model):
    optimizer = _GeneralizedGammaLatencyOptimizer()
    log_parameters = np.log([1.0, 1.0e-4, 1000.0])
    time = np.array([1.0e-200, 1.0e-100])
    objective = lambda candidate: optimizer.log_value_and_grad(
        candidate, latent_model, time, np.zeros(2), np.ones(2)
    )
    value, gradient = objective(log_parameters)
    numerical = approx_derivative(
        lambda candidate: objective(candidate)[0], log_parameters
    ).ravel()
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-9)


def test_invalid_latency_trial_is_not_a_finite_stationary_point(latent_model):
    value, gradient = _GeneralizedGammaLatencyOptimizer().value_and_grad(
        [0.0, 1.0, 1.0], latent_model, np.ones(2), np.zeros(2), np.ones(2)
    )
    assert value == np.inf
    assert np.isnan(gradient).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_successful_latency_optimizer_cannot_accept_invalid_objective(
    latent_model, invalid
):
    result = SimpleNamespace(
        success=True, fun=invalid, x=np.zeros(3), jac=np.zeros(3), message="controlled"
    )
    with pytest.raises(RuntimeError, match="optimization failed"):
        _GeneralizedGammaLatencyOptimizer._validate_result(latent_model, result)


@pytest.mark.parametrize("finite_value", [1.0e100, 1.0e200, 1.0e300])
def test_large_finite_latency_objectives_are_not_invalid_sentinels(
    latent_model, finite_value
):
    result = SimpleNamespace(
        success=True, fun=finite_value, x=np.zeros(3), jac=np.ones(3), message="valid"
    )
    optimizer = _GeneralizedGammaLatencyOptimizer()
    optimizer._validate_result(latent_model, result)
    assert not optimizer._log_objective_is_invalid(latent_model, finite_value)


def test_successful_latency_optimizer_cannot_accept_invalid_gradient(latent_model):
    result = SimpleNamespace(
        success=True,
        fun=0.0,
        x=np.zeros(3),
        jac=np.full(3, np.nan),
        message="controlled",
    )
    with pytest.raises(RuntimeError, match="optimization failed"):
        _GeneralizedGammaLatencyOptimizer._validate_result(latent_model, result)


@pytest.mark.parametrize("invalid_part", ["value", "gradient"])
def test_latency_optimizer_reevaluates_candidate_before_mutating_distribution(
    latent_model, invalid_part, monkeypatch
):
    latency = GeneralizedGamma()
    result = SimpleNamespace(
        success=True, fun=0.0, x=np.ones(3), jac=np.zeros(3), message="controlled"
    )
    monkeypatch.setattr(latency_module, "minimize", lambda **kwargs: result)

    def objective(parameters, *args):
        if np.any(parameters):
            return (
                (np.inf, np.zeros(3))
                if invalid_part == "value"
                else (0.0, np.full(3, np.nan))
            )
        return 0.0, np.zeros(3)

    with pytest.raises(RuntimeError, match="final objective or gradient is invalid"):
        _GeneralizedGammaLatencyOptimizer().maximize(
            latency, latent_model, objective, np.ones(2), np.zeros(2), np.ones(2)
        )
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


def test_latency_optimizer_keeps_previous_parameters_if_finite_candidate_is_worse(
    latent_model, monkeypatch
):
    latency = GeneralizedGamma()
    result = SimpleNamespace(
        success=True, fun=1.0, x=np.ones(3), jac=np.zeros(3), message="controlled"
    )
    monkeypatch.setattr(latency_module, "minimize", lambda **kwargs: result)
    objective = lambda parameters, *args: (
        float(parameters @ parameters),
        2.0 * parameters,
    )
    _GeneralizedGammaLatencyOptimizer().maximize(
        latency, latent_model, objective, np.ones(2), np.zeros(2), np.ones(2)
    )
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)
