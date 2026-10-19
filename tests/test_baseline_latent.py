"""The linear baselines share the count-family strategies used by GGKM."""

import pickle

import numpy as np
import pytest
from sksurv.util import Surv

from skcure.models import GGKM, MixtureCureModel, PromotionTimeModel
from skcure.models.latent import _BinomialLatentModel, _PoissonLatentModel

FAMILIES = [
    (PromotionTimeModel, _PoissonLatentModel, "poisson", "_predictor_value_and_grad"),
    (MixtureCureModel, _BinomialLatentModel, "binomial", "_predictor_value_and_grad"),
]


def _training_data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    time = np.round(rng.uniform(0.4, 3.2, size=24), 2)
    event = np.tile([True, False, True, True], 6)
    return X, Surv.from_arrays(event, time)


def _fixed_predictors(estimator_class, family, fit_intercept):
    X = np.array([[-0.7, 0.2], [0.3, -0.8], [0.9, 0.4], [-0.2, 0.7]])
    time = np.array([0.2, 0.8, 1.7, 2.4])
    event = np.array([1.0, 0.0, 1.0, 0.0])
    dual = np.array([0.2, -0.1, 0.3, -0.2])
    intercept = 0.15 if fit_intercept else 0.0
    coefficients = X.T @ dual
    if fit_intercept:
        coefficients = np.r_[intercept, coefficients]

    baseline = estimator_class(a=1.2, d=0.9, p=1.1, fit_intercept=fit_intercept)
    baseline._set_latency_parameters(1.2, 0.9, 1.1)
    baseline._reset_components_for_fit()
    baseline.W_train_ = baseline._design_matrix(X)

    kernel = GGKM(
        a=1.2,
        d=0.9,
        p=1.1,
        latent_distribution=family,
        kernel="linear",
        lambda_reg=0.0,
    )
    kernel._latent_model = kernel._make_latent_model(family, phi=None, K_bin=1)
    kernel._reset_parameters_for_fit()
    kernel._reset_fit_results()
    kernel.K_ = X @ X.T

    return baseline, coefficients, kernel, intercept, dual, time, event


@pytest.mark.parametrize("estimator_class,strategy_class,family,objective", FAMILIES)
def test_baseline_uses_its_count_family(
    estimator_class, strategy_class, family, objective
):
    model = estimator_class()
    assert isinstance(model._latent_model, strategy_class)
    if family == "binomial":
        assert model._latent_model.K_bin == 1


@pytest.mark.parametrize("estimator_class,strategy_class,family,objective", FAMILIES)
def test_fit_and_predictions_call_shared_family_entrypoints(
    estimator_class,
    strategy_class,
    family,
    objective,
    monkeypatch,
):
    methods = {
        "penalized_loglikelihood": "observed_loglikelihood_from_eta",
        "e_step": "e_step_from_eta",
        "predictor_value_and_grad": "incidence_value_and_grad",
        "predict_survival": "predict_survival_from_eta",
        "predict_cure_probability": "predict_cure_probability_from_eta",
    }
    calls = dict.fromkeys(methods, 0)
    model = estimator_class(max_em_iter=1)

    def record_calls(label, shared_method):
        original = getattr(strategy_class, shared_method)

        if shared_method == "incidence_value_and_grad":

            def delegated(strategy, *args, **kwargs):
                calls[label] += 1
                return original(strategy, *args, **kwargs)

        else:

            def delegated(strategy, latency, *args, **kwargs):
                assert latency is model._latency
                calls[label] += 1
                return original(strategy, latency, *args, **kwargs)

        monkeypatch.setattr(strategy_class, shared_method, delegated)

    for label, shared_method in methods.items():
        record_calls(label, shared_method)

    X, y = _training_data()
    model.fit(X, y)
    model.predict_survival(X[:2], [0.0, 0.5, 2.0])
    model.predict_cure_probability(X[:2])
    assert all(count > 0 for count in calls.values()), calls


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("estimator_class,strategy_class,family,objective", FAMILIES)
def test_linear_baseline_matches_unpenalized_linear_ggkm(
    estimator_class,
    strategy_class,
    family,
    objective,
    fit_intercept,
):
    baseline, coefficients, kernel, intercept, dual, time, event = _fixed_predictors(
        estimator_class,
        family,
        fit_intercept,
    )
    np.testing.assert_allclose(
        baseline.W_train_ @ coefficients,
        kernel._predictor(intercept, dual),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        baseline._e_step(coefficients, time, event),
        kernel._e_step(intercept, dual, time, event),
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        baseline._loglikelihood(coefficients, time, event),
        kernel._penalized_loglikelihood(intercept, dual, time, event),
        rtol=1e-13,
        atol=1e-15,
    )


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("estimator_class,strategy_class,family,objective", FAMILIES)
def test_linear_predictor_gradient_matches_finite_differences(
    estimator_class,
    strategy_class,
    family,
    objective,
    fit_intercept,
):
    model, coefficients, _, _, _, time, event = _fixed_predictors(
        estimator_class,
        family,
        fit_intercept,
    )
    expected = model._e_step(coefficients, time, event)
    value_and_grad = getattr(model, objective)
    value, gradient = value_and_grad(coefficients, expected)
    assert np.isfinite(value)

    numerical = np.empty_like(coefficients)
    for j in range(len(coefficients)):
        step = np.zeros_like(coefficients)
        step[j] = 1e-6
        numerical[j] = (
            value_and_grad(coefficients + step, expected)[0]
            - value_and_grad(coefficients - step, expected)[0]
        ) / 2e-6

    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("estimator_class,strategy_class,family,objective", FAMILIES)
def test_fitted_pickle_preserves_composed_strategy_state(
    estimator_class,
    strategy_class,
    family,
    objective,
):
    X, y = _training_data()
    model = estimator_class(max_em_iter=1).fit(X, y)

    assert isinstance(model._latent_model, strategy_class)
    assert model.a_ == model._latency.a
    assert model.d_ == model._latency.d
    assert model.p_ == model._latency.p

    restored = pickle.loads(pickle.dumps(model))

    assert isinstance(restored._latent_model, strategy_class)
    assert restored.a_ == restored._latency.a
    assert restored.d_ == restored._latency.d
    assert restored.p_ == restored._latency.p

    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    np.testing.assert_array_equal(
        restored.predict_survival(X, [0.0, 0.5, 2.0]),
        model.predict_survival(X, [0.0, 0.5, 2.0]),
    )
    np.testing.assert_array_equal(
        restored.predict_cure_probability(X),
        model.predict_cure_probability(X),
    )
