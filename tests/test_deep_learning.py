"""Estimator and backend orchestration contracts for the optional deep model.

The small backend double isolates scikit-cure's validation, model selection,
refit lifecycle, and probability calculations from expensive neural training.
The integration tests exercise the real upstream implementation separately.
"""

import builtins
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import skcure
import skcure.deep_learning
from skcure.base import BaseCureEstimator
from skcure.deep_learning import DeepPromotionTimeCure


@pytest.fixture
def survival_data():
    X = np.column_stack((np.linspace(-1.0, 1.0, 24), np.arange(24) % 3))
    event = np.tile([True, False, True, True], 6)
    time = np.linspace(0.2, 4.8, 24)
    return X, Surv.from_arrays(event, time)


@pytest.fixture
def backend(monkeypatch):
    """Record the estimator's data/model boundaries without optional packages."""
    records = SimpleNamespace(models=[], fits=[])

    def build_model(self, breakpoints, seed):
        size = 2 if self.t_func == "weibull" else len(breakpoints) - 1
        parameters = np.ones(size)
        model = SimpleNamespace(
            endpoint=SimpleNamespace(get_weights=lambda: [parameters.copy()]),
            parameters=parameters,
            breakpoints=None if breakpoints is None else breakpoints.copy(),
            seed=seed,
        )
        records.models.append(model)
        return model

    def fit_model(
        self, model, X, t, event, *, validation_data=None, epochs,
        early_stopping=False,
    ):
        records.fits.append({
            "model": model,
            "X": X.copy(),
            "time": t.copy(),
            "event": event.copy(),
            "validation_data": validation_data,
            "epochs": epochs,
            "early_stopping": early_stopping,
        })
        if early_stopping:
            # The third raw loss is smaller, but min_delta may make epoch two
            # the restored checkpoint. Fit must honor the backend selection.
            history = {"loss": [3.0, 2.0, 1.0], "val_loss": [2.0, 1.0, 0.99]}
            return history, 2
        return {"loss": [1.0] * epochs}, epochs

    monkeypatch.setattr(
        DeepPromotionTimeCure, "_imports", staticmethod(lambda: (None, None))
    )
    monkeypatch.setattr(DeepPromotionTimeCure, "_build_model", build_model)
    monkeypatch.setattr(DeepPromotionTimeCure, "_fit_model", fit_model)
    monkeypatch.setattr(
        DeepPromotionTimeCure, "_predict_eta_from_validated_X",
        lambda self, X: X[:, 0].copy(),
    )
    return records


def test_public_estimator_contract():
    hidden_layers = [np.int64(5), np.int64(3)]
    learning_rate = np.float64(0.003)
    model = DeepPromotionTimeCure(
        hidden_layers=hidden_layers, learning_rate=learning_rate,
    )
    assert isinstance(model, BaseCureEstimator)
    assert model.get_params()["hidden_layers"] is hidden_layers
    assert model.get_params()["learning_rate"] is learning_rate
    assert not any(name.endswith("_") for name in vars(model))
    assert list(inspect.signature(model.fit).parameters) == ["X", "y"]
    assert clone(model).get_params() == model.get_params()
    assert model.set_params(epochs=3) is model
    for namespace in (skcure, skcure.deep_learning):
        assert namespace.DeepPromotionTimeCure is DeepPromotionTimeCure
        assert "DeepPromotionTimeCure" in namespace.__all__
    assert DeepPromotionTimeCure.__module__.startswith("skcure.deep_learning.")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_public_imports_do_not_load_optional_backends():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import skcure; import skcure.deep_learning; "
            "assert not any(name == 'tensorflow' or name.startswith('tensorflow.') "
            "or name == 'deepcure' or name.startswith('deepcure.') "
            "for name in sys.modules)"
        )],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unfitted_pickle_roundtrip_keeps_backend_imports_lazy():
    result = subprocess.run(
        [sys.executable, "-c", (
            "import pickle, sys; "
            "from skcure.deep_learning import DeepPromotionTimeCure; "
            "model = DeepPromotionTimeCure(hidden_layers=[4, 2], epochs=3); "
            "restored = pickle.loads(pickle.dumps(model)); "
            "assert restored.get_params() == model.get_params(); "
            "assert not restored.__sklearn_is_fitted__(); "
            "assert not any(name == 'tensorflow' or name.startswith('tensorflow.') "
            "or name == 'deepcure' or name.startswith('deepcure.') "
            "for name in sys.modules)"
        )],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_tensorflow_fails_only_when_fitting(monkeypatch, survival_data):
    original_import = builtins.__import__

    def without_tensorflow(name, *args, **kwargs):
        if name == "tensorflow" or name.startswith("tensorflow."):
            raise ImportError("TensorFlow deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tensorflow)
    model = DeepPromotionTimeCure(epochs=1)
    with pytest.raises(ImportError, match="(?i)tensorflow"):
        model.fit(*survival_data)
    with pytest.raises(NotFittedError):
        model.predict(survival_data[0])


def test_missing_deepcure_has_an_actionable_error(monkeypatch, survival_data):
    original_import = builtins.__import__

    def without_deepcure(name, *args, **kwargs):
        if name == "tensorflow":
            return SimpleNamespace()
        if name == "deepcure" or name.startswith("deepcure."):
            raise ImportError("deepcure deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_deepcure)
    with pytest.raises(ImportError, match="(?i)deepcure"):
        DeepPromotionTimeCure(epochs=1).fit(*survival_data)


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
@pytest.mark.parametrize("parametrization", [1, 2])
def test_probability_and_survival_contract(
    t_func, parametrization, backend, survival_data,
):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        t_func=t_func, parametrization=parametrization,
        epochs=4, validation_fraction=0,
    )
    assert model.fit(X, y) is model
    check_is_fitted(model)
    np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"]))
    eta = X[:4, 0]
    theta = np.exp(eta) if parametrization == 1 else np.logaddexp(0, eta)
    cure = np.exp(-theta)
    times = np.array([1.5, 0.0, 0.2, 1.5, 1e6])
    # Unit PE rates or unit Weibull shape and scale both give F0(t)=1-exp(-t).
    expected = np.exp(-theta[:, None] * (-np.expm1(-times)))
    np.testing.assert_allclose(model.predict(X[:4]), eta)
    np.testing.assert_allclose(model.predict_eta(X[:4]), eta)
    np.testing.assert_allclose(model.predict_theta(X[:4]), theta)
    np.testing.assert_allclose(model.predict_cure_probability(X[:4]), cure)
    observed = model.predict_survival(X[:4], times)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(observed[:, 1], np.ones(4))
    np.testing.assert_allclose(observed[:, -1], cure)
    assert np.all(observed >= cure[:, None] - 1e-15)
    assert np.all(np.diff(model.predict_survival(X[:4], np.sort(times)), axis=1) <= 0)
    expected_score = concordance_index_censored(y["event"], y["time"], X[:, 0])[0]
    assert model.score(X, y) == expected_score


@pytest.mark.parametrize("parametrization", [1, 2])
@pytest.mark.parametrize("t_func", ["pe", "weibull"])
def test_extreme_predictors_keep_survival_and_cure_finite(
    parametrization, t_func, backend, survival_data,
):
    model = DeepPromotionTimeCure(
        parametrization=parametrization, t_func=t_func,
        epochs=1, validation_fraction=0,
    ).fit(*survival_data)
    X = np.column_stack(([-1000.0, 0.0, 1000.0], np.zeros(3)))
    times = np.array([0.0, 1e-300, 0.2, 2.0, 1e6])
    survival = model.predict_survival(X, times)
    cure = model.predict_cure_probability(X)
    assert np.all(np.isfinite(survival))
    assert np.all(np.isfinite(cure))
    assert np.all((0 <= survival) & (survival <= 1))
    assert np.all((0 <= cure) & (cure <= 1))
    np.testing.assert_array_equal(survival[:, 0], np.ones(3))
    np.testing.assert_array_equal(survival[0], np.ones(len(times)))
    np.testing.assert_allclose(survival[:, -1], cure)
    np.testing.assert_array_equal(cure[[0, 2]], [1.0, 0.0])
    assert np.all(survival >= cure[:, None] - 1e-15)
    assert np.all(np.diff(survival, axis=1) <= 1e-15)
    assert np.all(np.diff(cure) <= 0)
    np.testing.assert_array_equal(model.predict(X), X[:, 0])
    if parametrization == 1:
        with pytest.raises(FloatingPointError, match="(?i)(mean|precision|exceed|overflow)"):
            model.predict_theta(X)
        np.testing.assert_array_equal(model.predict_theta(X[:2]), [0.0, 1.0])
    else:
        np.testing.assert_allclose(
            model.predict_theta(X), [0.0, np.log(2.0), 1000.0],
        )


def test_curve_and_cumulative_hazard_formats(backend, survival_data):
    X, y = survival_data
    model = DeepPromotionTimeCure(epochs=2, validation_fraction=0).fit(X, y)
    expected = model.predict_survival(X[:3], model.unique_times_)
    curves = model.predict_survival_function(X[:3])
    hazards = model.predict_cumulative_hazard_function(X[:3])
    np.testing.assert_allclose(
        model.predict_survival_function(X[:3], return_array=True), expected,
    )
    np.testing.assert_allclose(
        model.predict_cumulative_hazard_function(X[:3], return_array=True),
        -np.log(expected),
    )
    for i, (curve, hazard) in enumerate(zip(curves, hazards)):
        assert isinstance(curve, StepFunction)
        assert isinstance(hazard, StepFunction)
        np.testing.assert_allclose(curve(model.unique_times_), expected[i])
        np.testing.assert_allclose(hazard(model.unique_times_), -np.log(expected[i]))


@pytest.mark.parametrize("t_func", ["pe", "weibull"])
def test_log_survival_preserves_finite_cumulative_hazard_after_underflow(
    t_func, backend, survival_data,
):
    model = DeepPromotionTimeCure(
        t_func=t_func, epochs=1, validation_fraction=0,
    ).fit(*survival_data)
    X = np.array([[8.0, 0.0]])
    expected = -np.exp(8.0) * (-np.expm1(-1.0))
    assert model.predict_survival(X, [1.0])[0, 0] == 0.0
    np.testing.assert_allclose(model.predict_log_survival(X, [1.0]), [[expected]])
    cumulative_hazard = model.predict_cumulative_hazard_function(X, return_array=True)
    assert np.all(np.isfinite(cumulative_hazard))


def test_small_weibull_baseline_cdf_is_combined_before_exponentiating(
    backend, survival_data,
):
    model = DeepPromotionTimeCure(
        t_func="weibull", epochs=1, validation_fraction=0,
    ).fit(*survival_data)
    model.baseline_parameters_ = np.array([2.0, 1.0])
    X = np.array([[900.0, 0.0]])
    time = 1e-200
    expected_hazard = np.exp(900.0 + 2 * np.log(time))
    np.testing.assert_allclose(
        model.predict_log_survival(X, [time]), [[-expected_hazard]], rtol=1e-13,
    )
    assert model.predict_survival(X, [time])[0, 0] < 1.0


def test_softplus_log_link_preserves_large_negative_predictors(backend, survival_data):
    model = DeepPromotionTimeCure(
        parametrization=2, epochs=1, validation_fraction=0,
    ).fit(*survival_data)
    np.testing.assert_array_equal(model._log_theta(np.array([-1000.0])), [-1000.0])


def test_piecewise_rates_and_final_interval_extrapolation(backend, survival_data):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        n_intervals=3, epochs=1, validation_fraction=0,
    ).fit(X, y)
    rates = np.linspace(0.2, 0.8, len(model.breakpoints_) - 1)
    model.baseline_parameters_ = rates
    breaks = model.breakpoints_
    times = np.concatenate(([0.0], breaks[1:], [breaks[-1] + 2.0]))
    cumulative_hazard = np.zeros(len(times))
    for j, rate in enumerate(rates):
        exposure = np.maximum(times - breaks[j], 0)
        if j < len(rates) - 1:
            exposure = np.minimum(exposure, breaks[j + 1] - breaks[j])
        cumulative_hazard += rate * exposure
    cdf = -np.expm1(-cumulative_hazard)
    theta = np.exp(X[:3, 0])
    expected = np.exp(-theta[:, None] * cdf)
    np.testing.assert_allclose(model.predict_survival(X[:3], times), expected)
    # The finite final breakpoint marks training support; the final rate
    # continues beyond it so the latency distribution remains proper.
    assert np.all(expected[:, -1] < expected[:, -2])


def test_weibull_baseline_parameter_order(backend, survival_data):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        t_func="weibull", epochs=1, validation_fraction=0,
    ).fit(X, y)
    model.baseline_parameters_ = np.array([1.7, 2.3])
    times = np.array([0.0, 0.5, 1.3, 4.0])
    cdf = -np.expm1(-(times / 2.3) ** 1.7)
    expected = np.exp(-np.exp(X[:3, 0, None]) * cdf)
    np.testing.assert_allclose(model.predict_survival(X[:3], times), expected)


def test_selection_refits_fresh_model_for_selected_checkpoint(backend, survival_data):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        epochs=9, validation_fraction=0.25, min_delta=0.1,
        refit_full_data=True, random_state=73,
    ).fit(X, y)
    assert len(backend.models) == len(backend.fits) == 2
    selection, refit = backend.fits
    assert selection["early_stopping"] is True
    assert selection["epochs"] == 9
    assert selection["validation_data"] is not None
    assert len(selection["X"]) == 18
    assert len(selection["validation_data"][0]) == 6
    assert np.any(selection["event"])
    assert refit["early_stopping"] is False
    assert refit["validation_data"] is None
    assert refit["epochs"] == model.best_epoch_ == 2
    np.testing.assert_array_equal(refit["X"], X)
    np.testing.assert_array_equal(refit["time"], y["time"])
    assert model.model_ is refit["model"]
    assert model.model_ is not selection["model"]
    assert model.history_ == {"loss": [3.0, 2.0, 1.0], "val_loss": [2.0, 1.0, 0.99]}
    assert model.refit_history_ == {"loss": [1.0, 1.0]}
    assert model.n_iter_ == 2
    assert backend.models[0].seed == backend.models[1].seed
    assert backend.models[0].breakpoints[-1] > np.max(y["time"])


def test_selection_can_keep_the_selected_training_model(backend, survival_data):
    model = DeepPromotionTimeCure(
        epochs=9, refit_full_data=False,
    ).fit(*survival_data)
    assert len(backend.models) == len(backend.fits) == 1
    assert model.model_ is backend.models[0]
    assert model.best_epoch_ == 2
    assert model.refit_history_ is None
    np.testing.assert_array_equal(model.breakpoints_, backend.models[0].breakpoints)


@pytest.mark.parametrize("refit_full_data", [True, False])
def test_zero_validation_fraction_trains_once_on_all_data(
    refit_full_data, backend, survival_data,
):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        epochs=4, validation_fraction=0, refit_full_data=refit_full_data,
    ).fit(X, y)
    assert len(backend.models) == len(backend.fits) == 1
    call = backend.fits[0]
    np.testing.assert_array_equal(call["X"], X)
    assert call["validation_data"] is None
    assert call["early_stopping"] is False
    assert call["epochs"] == model.best_epoch_ == model.n_iter_ == 4
    assert model.refit_history_ is None


@pytest.mark.parametrize("n_samples", [2, 4, 7])
@pytest.mark.parametrize("event_count", [1, 2])
def test_small_split_keeps_an_event_in_training(
    n_samples, event_count, backend,
):
    X = np.arange(n_samples * 2, dtype=float).reshape(n_samples, 2)
    event = np.arange(n_samples) < event_count
    y = Surv.from_arrays(event, np.arange(1.0, n_samples + 1.0))
    DeepPromotionTimeCure(
        epochs=3, validation_fraction=0.15, refit_full_data=False,
    ).fit(X, y)
    call = backend.fits[0]
    assert np.any(call["event"])
    validation_X = call["validation_data"][0]
    assert len(call["X"]) >= 1
    assert len(validation_X) >= 1
    assert len(call["X"]) + len(validation_X) == n_samples
    assert not set(call["X"][:, 0]).intersection(validation_X[:, 0])


def test_tied_event_times_produce_valid_piecewise_breakpoints(backend):
    X = np.arange(16, dtype=float).reshape(8, 2)
    event = np.tile([True, False], 4)
    times = np.where(event, 1.0, np.arange(2.0, 10.0))
    model = DeepPromotionTimeCure(
        n_intervals=8, validation_fraction=0, epochs=1,
    ).fit(X, Surv.from_arrays(event, times))
    assert model.breakpoints_[0] == 0
    assert model.breakpoints_[-1] > times.max()
    assert np.all(np.diff(model.breakpoints_) > 0)
    assert len(model.baseline_parameters_) == len(model.breakpoints_) - 1


def test_training_pipeline_and_dataframe_feature_contract(backend, survival_data):
    X, y = survival_data
    frame = pd.DataFrame(X, columns=["age", "marker"])
    custom_y = Surv.from_arrays(
        y["event"], y["time"], name_event="observed", name_time="duration",
    )
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0).fit(frame, custom_y)
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    assert model.predict(frame).shape == (len(frame),)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(frame[["marker", "age"]])
    pipeline = make_pipeline(
        StandardScaler(), DeepPromotionTimeCure(epochs=1, validation_fraction=0),
    ).fit(frame, custom_y)
    assert pipeline.predict(frame[:3]).shape == (3,)
    assert np.isfinite(pipeline.score(frame, custom_y))


def test_prediction_uses_fitted_configuration_after_set_params(backend, survival_data):
    X, y = survival_data
    model = DeepPromotionTimeCure(
        t_func="pe", parametrization=1, epochs=1, validation_fraction=0,
    ).fit(X, y)
    times = np.array([0.0, 0.5, 3.0, 100.0])
    theta = model.predict_theta(X)
    cure = model.predict_cure_probability(X)
    survival = model.predict_survival(X, times)
    model.set_params(t_func="weibull", parametrization=2, n_intervals=1)
    assert model.t_func_ == "pe"
    assert model.parametrization_ == 1
    np.testing.assert_array_equal(model.predict_theta(X), theta)
    np.testing.assert_array_equal(model.predict_cure_probability(X), cure)
    np.testing.assert_array_equal(model.predict_survival(X, times), survival)


def test_repeated_fit_uses_the_same_seed_and_split(backend, survival_data):
    model = DeepPromotionTimeCure(
        epochs=3, refit_full_data=False, random_state=73,
    )
    model.fit(*survival_data)
    first_split = backend.fits[0]["X"].copy()
    first_breakpoints = model.breakpoints_.copy()
    first_model = model.model_
    model.fit(*survival_data)
    assert model.model_ is not first_model
    assert backend.models[0].seed == backend.models[1].seed
    np.testing.assert_array_equal(backend.fits[1]["X"], first_split)
    np.testing.assert_array_equal(model.breakpoints_, first_breakpoints)


@pytest.mark.parametrize("method,args", [
    ("predict", ()), ("predict_eta", ()), ("predict_theta", ()),
    ("predict_cure_probability", ()), ("predict_survival", ([0.0, 1.0],)),
    ("predict_survival_function", ()), ("predict_cumulative_hazard_function", ()),
])
def test_unfitted_predictions_raise_estimator_error(method, args):
    with pytest.raises(NotFittedError):
        getattr(DeepPromotionTimeCure(), method)(np.ones((2, 2)), *args)


@pytest.mark.parametrize("params", [
    {"hidden_layers": (3, 0)}, {"dropout": 1.0}, {"dropout": np.nan},
    {"t_func": "gamma"}, {"n_intervals": 0}, {"parametrization": 3},
    {"optimizer": "invalid"}, {"learning_rate": 0.0},
    {"decay_steps": 0}, {"decay_rate": np.nan}, {"epochs": 0},
    {"batch_size": 0}, {"validation_fraction": -0.1},
    {"validation_fraction": 1.0}, {"patience": -1}, {"min_delta": -0.1},
    {"deterministic": "yes"}, {"refit_full_data": "yes"},
])
def test_fit_validates_hyperparameters(params, backend, survival_data):
    model = DeepPromotionTimeCure(**params)
    with pytest.raises((ValueError, TypeError)):
        model.fit(*survival_data)
    assert backend.models == []
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("invalid_target", [
    "unstructured", "integer_event", "length", "zero_time", "all_censored",
])
def test_fit_validates_survival_targets(invalid_target, backend, survival_data):
    X, y = survival_data
    if invalid_target == "unstructured":
        y = np.column_stack((y["event"], y["time"]))
    elif invalid_target == "integer_event":
        y = np.array(list(zip(y["event"], y["time"])), dtype=[("event", int), ("time", float)])
    elif invalid_target == "length":
        y = y[:-1]
    elif invalid_target == "zero_time":
        y = Surv.from_arrays(y["event"], np.zeros(len(y)))
    else:
        y = Surv.from_arrays(np.zeros(len(y), dtype=bool), y["time"])
    with pytest.raises(ValueError):
        DeepPromotionTimeCure(epochs=1).fit(X, y)
    assert backend.models == []


@pytest.mark.parametrize("times", [[], [[1.0]], [-1.0], [np.nan], [np.inf]])
def test_prediction_validates_times(times, backend, survival_data):
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0).fit(*survival_data)
    with pytest.raises(ValueError, match="times"):
        model.predict_survival(survival_data[0][:2], times)


def test_prediction_validates_features(backend, survival_data):
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0).fit(*survival_data)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((3, 3)))
    with pytest.raises(ValueError):
        model.predict(np.full((3, 2), np.nan))


@pytest.mark.parametrize("failure", ["validation", "parameters", "training"])
def test_failed_refit_invalidates_predictions(failure, backend, survival_data, monkeypatch):
    X, y = survival_data
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0).fit(X, y)
    if failure == "validation":
        with pytest.raises(ValueError):
            model.fit(X, y[:-1])
    elif failure == "parameters":
        model.set_params(epochs=0)
        with pytest.raises(ValueError):
            model.fit(X, y)
    else:
        def fail_training(*args, **kwargs):
            raise FloatingPointError("Synthetic training failure")

        monkeypatch.setattr(model, "_fit_model", fail_training)
        with pytest.raises(FloatingPointError, match="Synthetic training failure"):
            model.fit(X, y)
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.predict_survival_function(X)


@pytest.mark.parametrize("history", [{"loss": []}, {"loss": [np.nan]}])
def test_invalid_backend_loss_prevents_fitted_state(
    history, backend, survival_data, monkeypatch,
):
    monkeypatch.setattr(
        DeepPromotionTimeCure, "_fit_model",
        lambda *args, **kwargs: (history, 1),
    )
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0)
    with pytest.raises(FloatingPointError, match="loss"):
        model.fit(*survival_data)
    with pytest.raises(NotFittedError):
        model.predict(survival_data[0])


@pytest.mark.parametrize("parameter", [0.0, np.nan])
def test_invalid_backend_baseline_prevents_fitted_state(
    parameter, backend, survival_data, monkeypatch,
):
    original_build = DeepPromotionTimeCure._build_model

    def invalid_baseline(self, breakpoints, seed):
        model = original_build(self, breakpoints, seed)
        model.parameters[:] = parameter
        return model

    monkeypatch.setattr(DeepPromotionTimeCure, "_build_model", invalid_baseline)
    model = DeepPromotionTimeCure(epochs=1, validation_fraction=0)
    with pytest.raises(FloatingPointError, match="baseline"):
        model.fit(*survival_data)
    with pytest.raises(NotFittedError):
        model.predict(survival_data[0])
