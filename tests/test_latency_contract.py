"""Contract tests for susceptible-event-time latency distributions."""

import numpy as np
import pytest

from skcure.distributions import GeneralizedGamma
from skcure.distributions.base import (
    LatencyDistribution,
    validate_latency_distribution,
)
from skcure._latent import (
    _BinomialLatentModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


class _ExponentialLatency:
    """Minimal structural implementation of the latency interface."""

    def pdf(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-t)

    def logpdf(self, t):
        t = np.asarray(t, dtype=float)
        return -t

    def cdf(self, t):
        t = np.asarray(t, dtype=float)
        return -np.expm1(-t)

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-t)


def test_generalized_gamma_satisfies_latency_distribution_contract():
    latency = GeneralizedGamma(
        a=1.2,
        d=0.9,
        p=1.1,
    )

    assert isinstance(latency, LatencyDistribution)
    assert validate_latency_distribution(latency) is latency


def test_latency_distribution_contract_is_structural():
    latency = _ExponentialLatency()

    assert isinstance(latency, LatencyDistribution)
    assert validate_latency_distribution(latency) is latency


def test_latency_distribution_validator_rejects_incomplete_object():
    class _MissingSurvivalFunction:
        def pdf(self, t):
            return np.asarray(t, dtype=float)

        def logpdf(self, t):
            return np.asarray(t, dtype=float)

        def cdf(self, t):
            return np.asarray(t, dtype=float)

    with pytest.raises(
        TypeError,
        match="latency distribution must implement",
    ):
        validate_latency_distribution(
            _MissingSurvivalFunction(),
        )


@pytest.mark.parametrize(
    "latent_model",
    [
        _PoissonLatentModel(),
        _NegativeBinomialLatentModel(phi=0.5),
        _BinomialLatentModel(K_bin=2),
    ],
)
def test_latent_count_models_depend_only_on_latency_distribution_contract(
    latent_model,
):
    latency = _ExponentialLatency()

    eta = np.array(
        [-0.25, 0.4],
        dtype=float,
    )
    t = np.array(
        [0.5, 1.5],
        dtype=float,
    )
    delta = np.array(
        [1.0, 0.0],
        dtype=float,
    )
    t_grid = np.array(
        [0.0, 0.5, 2.0],
        dtype=float,
    )

    loglik = latent_model.observed_loglikelihood_from_eta(
        latency,
        eta,
        t,
        delta,
    )

    expected = latent_model.e_step_from_eta(
        latency,
        eta,
        t,
        delta,
    )

    survival = latent_model.predict_survival_from_eta(
        latency,
        eta,
        t_grid,
    )

    cure = latent_model.predict_cure_probability_from_eta(
        latency,
        eta,
    )

    assert np.isfinite(loglik)

    assert expected.shape == eta.shape
    assert np.all(np.isfinite(expected))

    assert survival.shape == (
        len(eta),
        len(t_grid),
    )
    assert np.all(np.isfinite(survival))

    assert cure.shape == eta.shape
    assert np.all(np.isfinite(cure))
