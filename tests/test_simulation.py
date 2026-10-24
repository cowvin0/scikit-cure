from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from skcure.datasets import simulate_cure_survival
from skcure.linear_model import MixtureCure, PromotionTimeCure


def test_simulator_reproducibility_and_observation_mechanism():
    left = simulate_cure_survival(1000, random_state=9)
    right = simulate_cure_survival(1000, random_state=9)
    assert_array_equal(left.X, right.X)
    assert_array_equal(left.y, right.y)
    assert_array_equal(left.y["event"], left.event_time <= left.censoring_time)
    assert_array_equal(left.y["time"], np.minimum(left.event_time, left.censoring_time))
    assert_array_equal(np.isinf(left.event_time), left.cured)
    assert not left.y["event"][left.cured].any()


@pytest.mark.parametrize("latency", ["weibull", "lognormal"])
def test_ground_truth_matches_empirical_survival(latency):
    sample = simulate_cure_survival(30000, latency=latency, random_state=16)
    times = np.array([0, 0.5, 1, 2, 5])
    empirical = np.mean(sample.event_time[:, None] > times, axis=0)
    assert_allclose(
        empirical, sample.survival_probability(times).mean(axis=0), atol=0.01
    )
    assert_allclose(sample.cure_probability.mean(), 0.3, atol=0.005)
    assert_allclose(sample.cured.mean(), 0.3, atol=0.01)
    assert_allclose(sample.survival_probability([0]), 1)


def test_short_followup_censoring_is_not_a_cure_label():
    sample = simulate_cure_survival(
        2000, followup=0.1, censoring_rate=0, random_state=1
    )
    assert (~sample.y["event"]).mean() > sample.cured.mean() + 0.5


def test_correctly_specified_fit_recovers_cure_and_survival_probabilities():
    sample = simulate_cure_survival(
        700, random_state=7, censoring_rate=0.05, followup=5
    )
    model = MixtureCure(max_em_iter=80, tol=1e-6).fit(sample.X, sample.y)
    assert model.converged_
    # Fixed-seed, broad tolerances test statistical recovery rather than exact coefficients.
    assert (
        np.mean(
            np.abs(model.predict_cure_probability(sample.X) - sample.cure_probability)
        )
        < 0.07
    )
    times = [0.5, 1, 2, 4]
    assert (
        np.mean(
            np.abs(
                model.predict_survival(sample.X, times)
                - sample.survival_probability(times)
            )
        )
        < 0.04
    )


def test_promotion_time_fit_recovers_true_poisson_cure_and_survival():
    sample = simulate_cure_survival(
        700,
        latent_distribution="poisson",
        random_state=7,
        censoring_rate=0.05,
        followup=5,
    )
    model = PromotionTimeCure(max_em_iter=150, tol=1e-6).fit(sample.X, sample.y)
    assert model.converged_
    assert (
        np.mean(
            np.abs(model.predict_cure_probability(sample.X) - sample.cure_probability)
        )
        < 0.07
    )
    times = [0.5, 1, 2, 4]
    assert (
        np.mean(
            np.abs(
                model.predict_survival(sample.X, times)
                - sample.survival_probability(times)
            )
        )
        < 0.04
    )


@pytest.mark.parametrize("family", ["binomial", "poisson", "negative_binomial"])
@pytest.mark.parametrize("latency", ["weibull", "lognormal"])
def test_latent_count_truth_matches_empirical_cure_and_survival(family, latency):
    sample = simulate_cure_survival(
        40000,
        latent_distribution=family,
        latency=latency,
        K_bin=3,
        phi=0.4,
        random_state=160,
    )
    assert_array_equal(sample.cured, sample.latent_count == 0)
    assert_array_equal(np.isinf(sample.event_time), sample.cured)
    assert_array_equal(
        sample.y["time"], np.minimum(sample.event_time, sample.censoring_time)
    )
    assert_allclose(sample.cure_probability.mean(), 0.3, atol=0.006)
    assert_allclose(sample.cured.mean(), 0.3, atol=0.01)
    grid = np.array([0, 0.2, 0.5, 1, 2, 5])
    empirical = np.mean(sample.event_time[:, None] > grid, axis=0)
    assert_allclose(
        empirical, sample.survival_probability(grid).mean(axis=0), atol=0.01
    )
    assert_allclose(sample.survival_probability([0]), 1.0)
    assert_allclose(
        sample.survival_probability([1e10]).ravel(), sample.cure_probability, atol=1e-13
    )
    # Conditional first-event survival given N=2 is S0(t)**2, independently
    # of the count family's marginal PGF and its covariate effects.
    two_causes = sample.latent_count == 2
    assert two_causes.sum() > 1000
    susceptible = (
        np.exp(-(0.5**1.5)) if latency == "weibull" else ndtr(-np.log(0.5) / 1.5)
    )
    assert np.mean(sample.event_time[two_causes] > 0.5) == pytest.approx(
        susceptible**2, abs=0.025
    )


@pytest.mark.parametrize("family", ["binomial", "poisson", "negative_binomial"])
def test_latent_count_means_and_zero_probabilities_match_family_formulas(family):
    sample = simulate_cure_survival(
        40000,
        latent_distribution=family,
        K_bin=4,
        phi=0.4,
        random_state=91,
    )
    eta = sample.incidence_predictor
    assert_allclose(
        eta, sample.incidence_intercept + sample.X @ sample.incidence_coefficients
    )
    if family == "binomial":
        probability = 1 / (1 + np.exp(-eta))
        mean = 4 * probability
        variance = 4 * probability * (1 - probability)
        zero_probability = (1 - probability) ** 4
    else:
        mean = np.exp(eta)
        variance = mean if family == "poisson" else mean + 0.4 * mean**2
        zero_probability = (
            np.exp(-mean) if family == "poisson" else (1 + 0.4 * mean) ** (-1 / 0.4)
        )
    assert_allclose(sample.latent_mean, mean, rtol=1e-13)
    assert_allclose(sample.cure_probability, zero_probability, rtol=1e-12, atol=1e-15)
    assert abs(np.mean(sample.latent_count - mean)) < 5 * np.sqrt(
        np.mean(variance) / len(mean)
    )
    cure_standard_error = np.sqrt(
        np.mean(zero_probability * (1 - zero_probability)) / len(mean)
    )
    assert abs(np.mean(sample.cured - zero_probability)) < 5 * cure_standard_error


def test_negative_binomial_poisson_limit_has_identical_seeded_data():
    poisson = simulate_cure_survival(
        1000, latent_distribution="poisson", random_state=82
    )
    limit = simulate_cure_survival(
        1000, latent_distribution="negative_binomial", phi=0, random_state=82
    )
    for field in ("X", "y", "latent_count", "event_time", "cure_probability"):
        assert_array_equal(getattr(limit, field), getattr(poisson, field))
    assert_allclose(
        limit.survival_probability([0, 0.2, 1, 10]),
        poisson.survival_probability([0, 0.2, 1, 10]),
    )


def test_tiny_negative_binomial_dispersion_preserves_nonzero_mean():
    sample = simulate_cure_survival(
        1000, latent_distribution="negative_binomial", phi=1e-12, random_state=84
    )
    assert sample.latent_count.sum() > 0
    assert_allclose(
        sample.cure_probability, np.exp(-sample.latent_mean), rtol=2e-9, atol=1e-14
    )


def test_unrepresentable_event_times_are_not_mislabeled_as_cure():
    with pytest.raises(ValueError, match="unrepresentable latent event times"):
        simulate_cure_survival(
            1,
            latency="lognormal",
            latency_shape=2000,
            cure_coefficients=[0, 0],
            random_state=4,
        )


def test_legacy_simulation_without_incidence_still_has_correct_mixture_truth():
    sample = simulate_cure_survival(20, random_state=73)
    legacy = replace(sample, incidence_predictor=None)
    assert_allclose(
        legacy.survival_probability([0, 1, 3]), sample.survival_probability([0, 1, 3])
    )


def test_multi_trial_binomial_truth_preserves_zero_boundary_and_probability_bounds():
    sample = simulate_cure_survival(
        32, latent_distribution="binomial", K_bin=3, random_state=73
    )
    truth = sample.survival_probability([0.0, np.nextafter(0.0, 1.0), 1e-200, 1.0, 5.0])
    assert_array_equal(truth[:, 0], np.ones(32))
    assert np.all(truth <= 1.0)
    assert np.all(truth >= sample.cure_probability[:, None])
    assert np.all(np.diff(truth, axis=1) <= 0.0)


def test_large_binomial_count_preserves_tiny_event_probabilities():
    sample = simulate_cure_survival(
        1000,
        K_bin=10**16,
        cure_coefficients=[20, 0],
        random_state=73,
    )
    # Even K*F0 is far below float precision at this positive time. A rounded
    # log-sum error amplified by K must not create an artificial event risk.
    truth = sample.survival_probability([0.0, 1e-200])
    assert_array_equal(truth, np.ones((1000, 2)))


@pytest.mark.parametrize(
    "family,trials",
    [
        ("poisson", 1),
        ("negative_binomial", 1),
        ("binomial", 3),
    ],
)
def test_non_mixture_truth_requires_its_incidence_parameters(family, trials):
    sample = simulate_cure_survival(
        20, latent_distribution=family, K_bin=trials, random_state=73
    )
    incomplete = replace(sample, incidence_predictor=None)
    with pytest.raises(ValueError, match="incidence_predictor is required"):
        incomplete.survival_probability([0, 1, 3])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"n_features": 0},
        {"cure_fraction": 0},
        {"latency": "other"},
        {"latency_shape": 0},
        {"censoring_rate": -1},
        {"followup": 0},
        {"followup": None, "censoring_rate": 0},
        {"cure_coefficients": [1]},
        {"latent_distribution": "other"},
        {"K_bin": 0},
        {"K_bin": True},
        {"phi": -1},
        {"phi": np.nan},
        {"phi": True},
    ],
)
def test_invalid_simulation_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate_cure_survival(**kwargs)
