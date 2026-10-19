"""Regression tests for linear cure models built from shared cure components."""

import importlib
import pickle

import numpy as np
import pytest
from sksurv.util import Surv

from skcure.distributions import GeneralizedGamma
from skcure.models import MixtureCureModel, PromotionTimeModel
from skcure.models._latency import _GeneralizedGammaLatencyOptimizer
from skcure.models.latent import _BinomialLatentModel, _PoissonLatentModel

MODEL_INFO = {
    "mixture": (MixtureCureModel, _BinomialLatentModel),
    "promotion": (PromotionTimeModel, _PoissonLatentModel),
}


def _data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    time = np.round(rng.uniform(0.4, 3.2, size=24), 2)
    event = np.tile([True, False, True, True], 6)
    return X, Surv.from_arrays(event, time)


def _cases():
    return {
        "default": {},
        "one_feature": {},
        "no_intercept": {"fit_intercept": False},
        "nondefault_initial_values": {"a": 1.2, "d": 0.9, "p": 1.1},
        "early_convergence": {"tol": 1e6},
        "iteration_cap": {"max_em_iter": 1},
        "default_iteration_limit": {"max_em_iter": None, "tol": 1e6},
        "cold_refit": {},
        "warm_refit": {"warm_start": True},
        "warm_no_intercept": {
            "warm_start": True,
            "fit_intercept": False,
            "max_em_iter": 1,
        },
    }


SUCCESS_CASES = [(family, case) for family in MODEL_INFO for case in _cases()]


def _assert_valid_fit(model, X, y, X_test):
    assert model.__sklearn_is_fitted__()
    assert isinstance(model._latency, GeneralizedGamma)
    assert isinstance(model._latency_optimizer, _GeneralizedGammaLatencyOptimizer)

    assert model.a_ == model._latency.a
    assert model.d_ == model._latency.d
    assert model.p_ == model._latency.p
    assert model.a_ > 0.0
    assert model.d_ > 0.0
    assert model.p_ > 0.0

    assert model.n_iter_ >= 1
    assert model.n_iter_ <= model._max_em_iter
    assert len(model.loglik_history_) == model.n_iter_
    assert np.all(np.isfinite(model.loglik_history_))
    assert np.isfinite(model.final_loglik_)

    eta = model.predict_eta(X_test)
    risk = model.predict(X_test)
    cure = model.predict_cure_probability(X_test)
    times = np.array([0.0, 0.5, 2.0, 5.0])
    survival = model.predict_survival(X_test, times)

    assert eta.shape == (len(X_test),)
    assert risk.shape == (len(X_test),)
    assert cure.shape == (len(X_test),)
    assert survival.shape == (len(X_test), len(times))

    assert np.all(np.isfinite(eta))
    assert np.all(np.isfinite(risk))
    assert np.all((0.0 <= cure) & (cure <= 1.0))
    assert np.all((0.0 <= survival) & (survival <= 1.0))
    np.testing.assert_allclose(survival[:, 0], 1.0, rtol=0.0, atol=np.finfo(float).eps)

    survival_array = model.predict_survival_function(X_test, return_array=True)
    hazard_array = model.predict_cumulative_hazard_function(X_test, return_array=True)

    assert survival_array.shape == (len(X_test), len(model.unique_times_))
    assert hazard_array.shape == survival_array.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        expected_hazard = -np.log(survival_array)
    np.testing.assert_allclose(
        hazard_array,
        expected_hazard,
        rtol=1e-13,
        atol=1e-15,
        equal_nan=True,
    )

    survival_steps = model.predict_survival_function(X_test)
    hazard_steps = model.predict_cumulative_hazard_function(X_test)

    for row, step in zip(survival_array, survival_steps):
        np.testing.assert_array_equal(step(model.unique_times_), row)

    for row, step in zip(hazard_array, hazard_steps):
        np.testing.assert_array_equal(step(model.unique_times_), row)

    assert np.isfinite(model.score(X, y))

    if isinstance(model, PromotionTimeModel):
        theta = model.predict_theta(X_test)
        assert theta.shape == (len(X_test),)
        assert np.all(np.isfinite(theta))
        assert np.all(theta > 0.0)
    else:
        susceptible = model.predict_susceptible_probability(X_test)
        assert susceptible.shape == (len(X_test),)
        assert np.all((0.0 <= susceptible) & (susceptible <= 1.0))


@pytest.mark.parametrize("family", MODEL_INFO)
def test_linear_baseline_uses_shared_components(family):
    model_class, latent_class = MODEL_INFO[family]
    model = model_class(a=1.2, d=0.9, p=1.1)

    assert isinstance(model._latent_model, latent_class)
    assert isinstance(model._latency_optimizer, _GeneralizedGammaLatencyOptimizer)

    if family == "mixture":
        assert model._latent_model.K_bin == 1

    model._set_latency_parameters(1.2, 0.9, 1.1)
    assert isinstance(model._latency, GeneralizedGamma)
    assert model.a_ == 1.2
    assert model.d_ == 0.9
    assert model.p_ == 1.1


@pytest.mark.parametrize("family,case", SUCCESS_CASES)
def test_linear_baseline_fit_and_refit_invariants(family, case):
    model_class, latent_class = MODEL_INFO[family]
    X, y = _data()
    X_test = np.array([[0.2, -0.4], [1.1, 0.7], [-0.3, 0.2]])

    if case == "one_feature":
        X = X[:, :1]
        X_test = X_test[:, :1]

    parameters = {"max_em_iter": 3}
    parameters.update(_cases()[case])
    model = model_class(**parameters)

    returned = model.fit(X, y)
    assert returned is model
    assert isinstance(model._latent_model, latent_class)
    _assert_valid_fit(model, X, y, X_test)

    if case == "early_convergence":
        assert model.converged_

    if case == "iteration_cap":
        assert model.n_iter_ == 1

    if case == "default_iteration_limit":
        assert model._max_em_iter == 1000

    if model.fit_intercept:
        assert model.coef_.shape == (X.shape[1],)
        assert np.isfinite(model.intercept_)
    else:
        assert model.coef_.shape == (X.shape[1],)
        assert model.intercept_ == 0.0

    if "refit" in case or case == "warm_no_intercept":
        returned = model.fit(X, y)
        assert returned is model
        assert isinstance(model._latent_model, latent_class)
        _assert_valid_fit(model, X, y, X_test)


@pytest.mark.parametrize("family", MODEL_INFO)
def test_latency_evaluation_is_the_shared_generalized_gamma(family):
    model_class, _ = MODEL_INFO[family]
    model = model_class(a=1.2, d=0.9, p=1.1)
    model._set_latency_parameters(1.2, 0.9, 1.1)
    reference = GeneralizedGamma(a=1.2, d=0.9, p=1.1)

    times = np.array([0.0, 1e-12, 0.1, 1.0, 100.0, np.inf, np.nan])

    with np.errstate(all="ignore"):
        for method in ("pdf", "logpdf", "cdf", "sf"):
            observed = getattr(model._latency, method)(times)
            expected = getattr(reference, method)(times)
            np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("family", MODEL_INFO)
def test_gg_log_objective_delegates_to_shared_optimizer(family):
    model_class, _ = MODEL_INFO[family]
    model = model_class(a=1.2, d=0.9, p=1.1)
    model._set_latency_parameters(1.2, 0.9, 1.1)
    model._reset_components_for_fit()

    t = np.array([0.2, 0.8, 1.7, 2.4])
    delta = np.array([1.0, 0.0, 1.0, 0.0])
    eta = np.array([0.2, -0.1, 0.3, -0.2])
    expected_count = model._latent_model.e_step_from_eta(
        model._latency,
        eta,
        t,
        delta,
    )
    log_parameters = np.log(np.array([1.2, 0.9, 1.1]))

    observed = model._gg_log_value_and_grad(
        log_parameters,
        t,
        delta,
        expected_count,
    )
    expected = model._latency_optimizer.log_value_and_grad(
        log_parameters,
        model._latent_model,
        t,
        delta,
        expected_count,
    )

    np.testing.assert_array_equal(observed[0], expected[0])
    np.testing.assert_array_equal(observed[1], expected[1])


@pytest.mark.parametrize("family", MODEL_INFO)
def test_fit_routes_latency_update_through_shared_optimizer(family, monkeypatch):
    model_class, latent_class = MODEL_INFO[family]
    model = model_class(max_em_iter=1)
    optimizer = model._latency_optimizer
    original = optimizer.maximize
    calls = []

    def wrapped(latency, latent_model, objective, t, delta, expected):
        calls.append((latency, latent_model, objective.__name__))
        return original(
            latency,
            latent_model,
            objective,
            t,
            delta,
            expected,
        )

    monkeypatch.setattr(optimizer, "maximize", wrapped)

    X, y = _data()
    model.fit(X, y)

    assert len(calls) == 1
    latency, latent_model, objective_name = calls[0]
    assert latency is model._latency
    assert latent_model is model._latent_model
    assert isinstance(latent_model, latent_class)
    assert objective_name == "_gg_log_value_and_grad"


@pytest.mark.parametrize("family", MODEL_INFO)
@pytest.mark.parametrize(
    "stage,method",
    [
        ("initial_likelihood", "_loglikelihood"),
        ("e_step", "_e_step"),
        ("predictor", "_predictor_value_and_grad"),
        ("generalized_gamma", "_gg_log_value_and_grad"),
        ("finalize", "_finish_fit"),
    ],
)
def test_callback_failures_propagate(family, stage, method, monkeypatch):
    model_class, _ = MODEL_INFO[family]
    model = model_class(max_em_iter=1)
    X, y = _data()

    def failed(*args, **kwargs):
        raise ArithmeticError("injected " + stage + " failure")

    monkeypatch.setattr(model, method, failed)

    with pytest.raises(ArithmeticError, match="injected " + stage + " failure"):
        model.fit(X, y)


@pytest.mark.parametrize("family", MODEL_INFO)
def test_nonfinite_post_m_likelihood_is_rejected(family, monkeypatch):
    model_class, _ = MODEL_INFO[family]
    model = model_class(max_em_iter=1)
    X, y = _data()

    original = model._loglikelihood
    calls = 0

    def likelihood(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return np.nan
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_loglikelihood", likelihood)

    with pytest.raises(FloatingPointError):
        model.fit(X, y)


@pytest.mark.parametrize(
    "failure",
    ["nan_fun", "unsuccessful", "nan_x"],
)
@pytest.mark.parametrize("family", MODEL_INFO)
def test_predictor_optimizer_rejects_invalid_results(family, failure, monkeypatch):
    model_class, _ = MODEL_INFO[family]
    model = model_class(max_em_iter=1)
    X, y = _data()

    em = importlib.import_module("skcure.models._em")
    original = em.minimize

    def invalid_minimize(*args, **kwargs):
        result = original(*args, **kwargs)
        if getattr(kwargs.get("fun"), "__name__", None) == "_predictor_value_and_grad":
            if failure == "nan_fun":
                result.fun = np.nan
            elif failure == "unsuccessful":
                result.success = False
                result.message = "injected unsuccessful result"
            else:
                result.x = result.x.copy()
                result.x[0] = np.nan
        return result

    monkeypatch.setattr(em, "minimize", invalid_minimize)

    with pytest.raises(RuntimeError):
        model.fit(X, y)


@pytest.mark.parametrize(
    "family,module_name",
    [("promotion", "promotion_time"), ("mixture", "mixture_cure")],
)
def test_baseline_import_compatibility(family, module_name):
    model_class, _ = MODEL_INFO[family]
    new_module = importlib.import_module("skcure.models." + module_name)
    public = importlib.import_module("skcure.models")
    legacy = importlib.import_module("skcure.models.baseline_model")

    assert getattr(new_module, model_class.__name__) is model_class
    assert getattr(public, model_class.__name__) is model_class
    assert getattr(legacy, model_class.__name__) is model_class


@pytest.mark.parametrize("family", MODEL_INFO)
def test_baseline_legacy_pickle_compatibility(family):
    """Old module-qualified pickles still resolve after moving a class."""
    model_class, _ = MODEL_INFO[family]
    original_module = model_class.__module__

    try:
        model_class.__module__ = "skcure.models.baseline_model"
        model = model_class(max_em_iter=1)
        payload = pickle.dumps(model)
    finally:
        model_class.__module__ = original_module

    restored = pickle.loads(payload)
    assert type(restored) is model_class
    assert restored.get_params() == model.get_params()


@pytest.mark.parametrize("family", MODEL_INFO)
def test_fitted_pickle_preserves_composed_components(family):
    model_class, latent_class = MODEL_INFO[family]
    X, y = _data()
    X_test = X[:3]
    model = model_class(max_em_iter=1).fit(X, y)

    restored = pickle.loads(pickle.dumps(model))

    assert isinstance(restored._latency, GeneralizedGamma)
    assert isinstance(restored._latent_model, latent_class)
    assert isinstance(restored._latency_optimizer, _GeneralizedGammaLatencyOptimizer)

    np.testing.assert_array_equal(
        restored.predict_eta(X_test), model.predict_eta(X_test)
    )
    np.testing.assert_array_equal(
        restored.predict_cure_probability(X_test),
        model.predict_cure_probability(X_test),
    )
    np.testing.assert_array_equal(
        restored.predict_survival(X_test, [0.0, 0.5, 2.0]),
        model.predict_survival(X_test, [0.0, 0.5, 2.0]),
    )
