"""Scientific checks against known latent cure and survival probabilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skcure.datasets import simulate_cure_survival
from skcure.linear_model import MixtureCure


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
    times = np.array([0, .5, 1, 2, 5])
    empirical = np.mean(sample.event_time[:, None] > times, axis=0)
    assert_allclose(empirical, sample.survival_probability(times).mean(axis=0), atol=.01)
    assert_allclose(sample.cure_probability.mean(), .3, atol=.005)
    assert_allclose(sample.cured.mean(), .3, atol=.01)
    assert_allclose(sample.survival_probability([0]), 1)


def test_short_followup_censoring_is_not_a_cure_label():
    sample = simulate_cure_survival(2000, followup=.1, censoring_rate=0, random_state=1)
    assert (~sample.y["event"]).mean() > sample.cured.mean() + .5


def test_correctly_specified_fit_recovers_cure_and_survival_probabilities():
    sample = simulate_cure_survival(700, random_state=7, censoring_rate=.05, followup=5)
    model = MixtureCure(max_em_iter=80, tol=1e-6).fit(sample.X, sample.y)
    assert model.converged_
    # Fixed-seed, broad tolerances test statistical recovery rather than exact coefficients.
    assert np.mean(np.abs(model.predict_cure_probability(sample.X) - sample.cure_probability)) < .07
    times = [.5, 1, 2, 4]
    assert np.mean(np.abs(model.predict_survival(sample.X, times) - sample.survival_probability(times))) < .04


@pytest.mark.parametrize("kwargs", [
    {"n_samples": 0}, {"n_features": 0}, {"cure_fraction": 0},
    {"latency": "other"}, {"latency_shape": 0}, {"censoring_rate": -1},
    {"followup": 0}, {"followup": None, "censoring_rate": 0},
    {"cure_coefficients": [1]},
])
def test_invalid_simulation_parameters(kwargs):
    with pytest.raises(ValueError):
        simulate_cure_survival(**kwargs)
