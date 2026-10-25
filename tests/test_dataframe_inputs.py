"""Eager DataFrames share the numerical and metadata contracts of array inputs."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
import warnings

import narwhals as nw
import narwhals.stable.v2 as nw2
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv

from skcure.deep_learning import DeepPromotionTimeCure
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


@contextmanager
def _fit_warning(model):
    """Allow only the convergence warning justified by a capped EM fit."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "always", category=ConvergenceWarning,
            message=r"EM did not converge within \d+ iterations\.",
        )
        yield
    if model.converged_:
        assert not captured
    else:
        assert model.termination_reason_ == "max_em_iter"
        assert len(captured) == 1
        assert issubclass(captured[0].category, ConvergenceWarning)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    frame = pl.DataFrame(X, schema=["age", "marker"])
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, size=len(X)), 3),
    )
    return X, frame, y


@pytest.fixture(scope="module", params=[
    pytest.param(GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1), id="ggkm"),
    pytest.param(MixtureCure(max_em_iter=1), id="mixture"),
    pytest.param(PromotionTimeCure(max_em_iter=1), id="promotion-time"),
    pytest.param(GradientBoostingCure(
        n_estimators=1, max_em_iter=1, min_samples_leaf=3, random_state=73,
    ), id="boosting"),
])
def fitted_pair(request, data):
    X, frame, y = data
    array_model: Any = clone(request.param)
    frame_model: Any = clone(request.param)
    with _fit_warning(array_model):
        array_model.fit(X, y)
    with _fit_warning(frame_model):
        frame_model.fit(frame, y)
    return array_model, frame_model


@pytest.mark.parametrize("wrap", [lambda x: x, nw.from_native, nw2.from_native],
                         ids=["native", "narwhals", "stable-v2"])
def test_frame_predictions_match_array_predictions(fitted_pair, data, wrap):
    array_model, frame_model = fitted_pair
    X, frame, y = data
    actual_X = wrap(frame)
    np.testing.assert_array_equal(frame_model.feature_names_in_, frame.columns)
    methods = [
        ("predict", ()),
        ("predict_eta", ()),
        ("predict_cure_probability", ()),
        ("predict_latent_mean", ()),
        ("predict_survival", ([0.0, 1.0, 3.0],)),
        ("predict_log_survival", ([0.0, 1.0, 3.0],)),
        ("predict_survival_function", (True,)),
        ("predict_cumulative_hazard_function", (True,)),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method, args in methods:
            actual = getattr(frame_model, method)(actual_X, *args)
            expected = getattr(array_model, method)(X, *args)
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
        assert frame_model.score(actual_X, y) == array_model.score(X, y)


@pytest.mark.parametrize("wrap", [nw.from_native, nw2.from_native],
                         ids=["narwhals", "stable-v2"])
def test_wrapped_frames_can_be_fitted(fitted_pair, data, wrap):
    array_model, _ = fitted_pair
    X, frame, y = data
    model: Any = clone(array_model)
    with _fit_warning(model):
        model.fit(wrap(frame), y)
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    np.testing.assert_allclose(model.predict(frame), array_model.predict(X),
                               rtol=1e-9, atol=1e-11)


def test_feature_names_are_checked_across_backends(fitted_pair, data):
    _, model = fitted_pair
    X, frame, y = data
    pandas_frame = pd.DataFrame(X, columns=frame.columns)
    np.testing.assert_allclose(model.predict(pandas_frame), model.predict(frame),
                               rtol=1e-9, atol=1e-11)
    pandas_model: Any = clone(model)
    with _fit_warning(pandas_model):
        pandas_model.fit(nw.from_native(pandas_frame), y)
    np.testing.assert_allclose(pandas_model.predict(frame), model.predict(frame),
                               rtol=1e-9, atol=1e-11)
    for invalid in [
        frame.select("marker", "age"),
        frame.select("age"),
        frame.with_columns(pl.lit(1.0).alias("extra")),
        frame.rename({"marker": "other"}),
    ]:
        with pytest.raises(ValueError, match="feature names"):
            model.predict(invalid)


def test_fit_and_prediction_do_not_mutate_data(fitted_pair, data):
    _, model = fitted_pair
    _, frame, y = data
    before = frame.clone()
    target_before = y.copy()
    model.predict(frame)
    model.predict_survival(frame, [0.0, 1.0])
    fresh: Any = clone(model)
    with _fit_warning(fresh):
        fresh.fit(frame, y)
    assert frame.equals(before)
    np.testing.assert_array_equal(y, target_before)


def test_prediction_converts_polars_once_without_pandas(fitted_pair, data, monkeypatch):
    _, model = fitted_pair
    _, frame, _ = data
    native_to_numpy = pl.DataFrame.to_numpy
    conversions = []

    def to_numpy(self, *args, **kwargs):
        conversions.append(self)
        return native_to_numpy(self, *args, **kwargs)

    def unexpected_to_pandas(*args, **kwargs):
        raise AssertionError("Polars input must not be converted through pandas")

    monkeypatch.setattr(pl.DataFrame, "to_numpy", to_numpy)
    monkeypatch.setattr(pl.DataFrame, "to_pandas", unexpected_to_pandas)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method, args in [
            ("predict", ()),
            ("predict_cure_probability", ()),
            ("predict_survival", ([0.0, 1.0, 3.0],)),
        ]:
            conversions.clear()
            getattr(model, method)(frame, *args)
            assert len(conversions) == 1
            assert conversions[0] is frame


def test_mixed_dtypes_and_sliced_chunked_frames(fitted_pair, data):
    template, _ = fitted_pair
    X, _, y = data
    frame = pl.DataFrame({
        "age": (X[:, 0] * 10).astype(np.int32),
        "marker": X[:, 1].astype(np.float32),
    })
    frame = pl.concat([frame[:12], frame[12:]], rechunk=False).slice(1, 22)
    assert frame.n_chunks() > 1
    array = frame.to_numpy().astype(np.float64)
    frame_model: Any = clone(template)
    array_model: Any = clone(template)
    with _fit_warning(frame_model):
        frame_model.fit(frame, y[1:-1])
    with _fit_warning(array_model):
        array_model.fit(array, y[1:-1])
    np.testing.assert_allclose(frame_model.predict(frame), array_model.predict(array),
                               rtol=1e-9, atol=1e-11)
    assert frame.n_chunks() > 1


def test_pipeline_polars_output_and_grid_search(data):
    _, frame, y = data
    pipeline = make_pipeline(
        StandardScaler().set_output(transform="polars"),
        MixtureCure(max_em_iter=1),
    )
    search = GridSearchCV(
        pipeline, {"mixturecure__tol": [1e-6, 1e-5]},
        cv=KFold(n_splits=2, shuffle=True, random_state=73),
    )
    search.set_params(error_score="raise")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        warnings.filterwarnings("always", category=ConvergenceWarning,
                                message=r"EM did not converge within 1 iterations\.")
        search.fit(frame, y)
    assert captured
    assert all(issubclass(item.category, ConvergenceWarning) for item in captured)
    model = search.best_estimator_.steps[-1][1]
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    assert search.predict(frame).shape == (len(frame),)
    assert np.isfinite(search.score(frame, y))


@pytest.mark.parametrize("invalid", [
    pl.DataFrame({"age": [1.0, None], "marker": [1.0, 2.0]}),
    pl.DataFrame({"age": [1.0, float("nan")], "marker": [1.0, 2.0]}),
    pl.DataFrame({"age": [1.0, float("inf")], "marker": [1.0, 2.0]}),
    pl.DataFrame({"age": ["young", "old"], "marker": [1.0, 2.0]}),
])
def test_invalid_frame_features_are_rejected(fitted_pair, data, invalid):
    _, model = fitted_pair
    _, _, y = data
    fresh: Any = clone(model)
    with pytest.raises((ValueError, TypeError)):
        fresh.fit(invalid, y[:2])
    with pytest.raises((ValueError, TypeError)):
        model.predict(invalid)


@pytest.mark.parametrize("wrap", [lambda x: x, nw.from_native, nw2.from_native],
                         ids=["native", "narwhals", "stable-v2"])
def test_lazy_inputs_require_explicit_collection(fitted_pair, data, wrap, monkeypatch):
    _, model = fitted_pair
    _, frame, y = data
    lazy = wrap(frame.lazy())

    def unexpected_collect(*args, **kwargs):
        raise AssertionError("Estimators must not implicitly collect lazy inputs")

    monkeypatch.setattr(pl.LazyFrame, "collect", unexpected_collect)
    fresh: Any = clone(model)
    with pytest.raises(TypeError, match="collect"):
        fresh.fit(lazy, y)
    with pytest.raises(TypeError, match="collect"):
        model.predict(lazy)


def test_neural_boundary_accepts_polars_without_optional_backend(data, monkeypatch):
    X, frame, y = data
    received = []

    def build_model(self, breakpoints, seed):
        return SimpleNamespace(endpoint=SimpleNamespace(
            get_weights=lambda: [np.ones(2)],
        ))

    def fit_model(self, model, X, t, event, **kwargs):
        received.append(X.copy())
        return {"loss": [1.0]}, 1

    monkeypatch.setattr(DeepPromotionTimeCure, "_imports",
                        staticmethod(lambda: (None, None)))
    monkeypatch.setattr(DeepPromotionTimeCure, "_build_model", build_model)
    monkeypatch.setattr(DeepPromotionTimeCure, "_fit_model", fit_model)
    monkeypatch.setattr(DeepPromotionTimeCure, "_predict_eta_from_validated_X",
                        lambda self, X: X[:, 0].copy())
    model = DeepPromotionTimeCure(t_func="weibull", epochs=1,
                                  validation_fraction=0).fit(nw.from_native(frame), y)
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], X)
    np.testing.assert_array_equal(getattr(model, "feature_names_in_"), frame.columns)
    np.testing.assert_array_equal(model.predict(frame), X[:, 0])
    np.testing.assert_allclose(model.predict_survival(frame, [0.0, 1.0]),
                               model.predict_survival(pd.DataFrame(X, columns=frame.columns),
                                                      [0.0, 1.0]))
