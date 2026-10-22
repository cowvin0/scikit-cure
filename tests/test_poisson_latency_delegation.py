import numpy as np
from numpy.testing import assert_allclose
from scipy.special import gammainc, gammaincc

from skcure.distributions import GeneralizedGamma
from skcure._latent import (
    _BinomialLatentModel,
    _LatentCountModel,
    _NegativeBinomialLatentModel,
    _PoissonLatentModel,
)


def test_generalized_gamma_sf_uses_stable_upper_tail():
    """
    The GG survival function should preserve probabilities in the extreme
    upper tail instead of computing 1 - CDF, which suffers cancellation.
    """
    latency = GeneralizedGamma(
        a=1.0,
        d=2.0,
        p=1.0,
    )

    t = np.array([50.0])

    # For these parameters:
    # s = d / p = 2
    # u = (t / a)**p = 50
    s = 2.0
    u = np.array([50.0])

    direct_sf = gammaincc(s, u)
    subtraction_sf = 1.0 - gammainc(s, u)

    assert direct_sf[0] > 0.0
    assert subtraction_sf[0] == 0.0

    assert_allclose(
        latency.sf(t),
        direct_sf,
        rtol=1e-13,
        atol=0.0,
    )


def test_one_minus_cdf_and_sf_agree_away_from_extreme_tail():
    """
    Away from the extreme tail, 1 - CDF and the direct survival function
    should be numerically equivalent.
    """
    latency = GeneralizedGamma(
        a=2.0,
        d=3.0,
        p=1.5,
    )

    t = np.array(
        [
            0.25,
            0.5,
            1.0,
            2.0,
            4.0,
        ]
    )

    direct_sf = latency.sf(t)
    subtraction_sf = 1.0 - latency.cdf(t)

    assert_allclose(
        direct_sf,
        subtraction_sf,
        rtol=1e-12,
        atol=1e-15,
    )


class _SentinelLatency:
    """
    Fake latency distribution used to verify that latent-count models
    delegate latency evaluation rather than implementing GG themselves.
    """

    def __init__(self):
        self.calls = []

    def pdf(self, t):
        self.calls.append("pdf")
        t = np.asarray(t, dtype=float)
        return np.full_like(t, 0.11)

    def logpdf(self, t):
        self.calls.append("logpdf")
        t = np.asarray(t, dtype=float)
        return np.full_like(t, -2.22)

    def cdf(self, t):
        self.calls.append("cdf")
        t = np.asarray(t, dtype=float)
        return np.full_like(t, 0.33)

    def sf(self, t):
        self.calls.append("sf")
        t = np.asarray(t, dtype=float)
        return np.full_like(t, 0.67)


def test_gg_named_latency_aliases_are_removed():
    """
    GG-specific names belong only to genuinely GG-specific code paths.
    """
    for latent_class in (
        _LatentCountModel,
        _PoissonLatentModel,
        _NegativeBinomialLatentModel,
        _BinomialLatentModel,
    ):
        for name in (
            "fGG",
            "log_fGG",
            "FGG",
            "SGG",
        ):
            assert not hasattr(latent_class, name)


def test_poisson_uses_supplied_latency_distribution():
    latency = _SentinelLatency()
    latent = _PoissonLatentModel()

    t = np.array([1.0, 2.0, 3.0])
    delta = np.array([1.0, 0.0, 1.0])
    eta = np.array([-0.2, 0.1, 0.4])

    ll = latent.observed_loglikelihood_from_eta(latency, eta, t, delta)
    nhat = latent.e_step_from_eta(latency, eta, t, delta)
    survival = latent.predict_survival_from_eta(
        latency,
        eta,
        np.array([0.5, 1.5]),
    )

    assert np.isfinite(ll)
    assert np.all(np.isfinite(nhat))
    assert np.all(np.isfinite(survival))
    assert {"cdf", "logpdf", "sf"} <= set(latency.calls)


def test_negative_binomial_uses_supplied_latency_distribution():
    latency = _SentinelLatency()
    latent = _NegativeBinomialLatentModel(phi=0.5)

    t = np.array([1.0, 2.0, 3.0])
    delta = np.array([1.0, 0.0, 1.0])
    eta = np.array([-0.2, 0.1, 0.4])

    ll = latent.observed_loglikelihood_from_eta(latency, eta, t, delta)
    nhat = latent.e_step_from_eta(latency, eta, t, delta)
    survival = latent.predict_survival_from_eta(
        latency,
        eta,
        np.array([0.5, 1.5]),
    )
    const, log_A = latent.dispersion_objective_terms_from_eta(
        latency,
        eta,
        t,
        delta,
    )

    assert np.isfinite(ll)
    assert np.all(np.isfinite(nhat))
    assert np.all(np.isfinite(survival))
    assert np.all(np.isfinite(const))
    assert np.all(np.isfinite(log_A))
    assert {"cdf", "logpdf", "sf"} <= set(latency.calls)


def test_binomial_uses_supplied_latency_distribution():
    latency = _SentinelLatency()
    latent = _BinomialLatentModel(K_bin=3)

    t = np.array([1.0, 2.0, 3.0])
    delta = np.array([1.0, 0.0, 1.0])
    eta = np.array([-0.2, 0.1, 0.4])

    ll = latent.observed_loglikelihood_from_eta(
        latency,
        eta,
        t,
        delta,
    )
    nhat = latent.e_step_from_eta(
        latency,
        eta,
        t,
        delta,
    )
    survival = latent.predict_survival_from_eta(
        latency,
        eta,
        np.array([0.5, 1.5]),
    )

    assert np.isfinite(ll)
    assert np.all(np.isfinite(nhat))
    assert np.all(np.isfinite(survival))
    assert {"logpdf", "sf"} <= set(latency.calls)


def test_generalized_gamma_sf_preserves_extreme_upper_tail():
    """
    The GG survival function should preserve probabilities in the extreme
    upper tail instead of relying on 1 - CDF, which can lose them through
    cancellation.
    """
    latency = GeneralizedGamma(
        a=1.0,
        d=2.0,
        p=1.0,
    )

    t = np.array([50.0])

    s = 2.0
    u = np.array([50.0])

    direct_sf = gammaincc(s, u)
    subtraction_sf = 1.0 - gammainc(s, u)

    assert direct_sf[0] > 0.0
    assert subtraction_sf[0] == 0.0

    actual_sf = latency.sf(t)

    assert actual_sf[0] > 0.0

    assert_allclose(
        actual_sf,
        direct_sf,
        rtol=1e-13,
        atol=0.0,
    )
