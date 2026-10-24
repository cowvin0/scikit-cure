"""Estimator contract and scientific regression tests for GGKM."""

import inspect
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammainc
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from skcure.base import BaseCureEstimator
from skcure.ggkm import GGKM


def _fit_cure_model(model, X, t, delta, **fit_params):
    """Adapt separate survival targets for the research compatibility tests."""
    if isinstance(model, BaseCureEstimator):
        delta = np.asarray(delta)
        if delta.ndim != 1 or not np.all(np.isin(delta, [0, 1])):
            raise ValueError("delta must be a one-dimensional array containing only 0 and 1.")

        y = Surv.from_arrays(delta.astype(bool), t)
        if fit_params:
            model.set_params(**fit_params)
        return model.fit(X, y)

    return model.fit(X, t, delta, **fit_params)


@pytest.fixture(scope="module")
def survival_data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    time = np.round(rng.uniform(0.4, 3.2, size=24), 2)
    event = np.tile([True, False, True, True], 6)
    return X, Surv.from_arrays(event, time)


@pytest.fixture(scope="module")
def fitted_model(survival_data):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
        return GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=2).fit(X, y)


def test_ggkm_is_defined_in_its_public_module():
    from skcure import GGKM as root_export

    assert GGKM.__module__ == "skcure.ggkm"
    assert root_export is GGKM


def test_fitted_estimator_pickle_roundtrip_uses_public_module(
    fitted_model, survival_data
):
    restored = pickle.loads(pickle.dumps(fitted_model))

    assert type(restored) is GGKM
    assert restored.get_params() == fitted_model.get_params()
    X, _ = survival_data
    np.testing.assert_array_equal(restored.predict(X), fitted_model.predict(X))
    np.testing.assert_array_equal(
        restored.predict_cure_probability(X),
        fitted_model.predict_cure_probability(X),
    )


def test_estimator_inherits_sklearn_and_sksurv_contracts():
    model = GGKM()
    assert isinstance(model, BaseCureEstimator)
    assert isinstance(model, SurvivalAnalysisMixin)
    assert isinstance(model, BaseEstimator)


def test_constructor_preserves_parameter_objects_and_has_no_fit_state():
    kernel_params = {"scale": 2.5}
    a = np.float64(1.2)
    max_em_iter = np.int64(2)
    model = GGKM(
        latent_distribution="NB",
        a=a,
        kernel_params=kernel_params,
        max_em_iter=max_em_iter,
    )
    assert model.get_params()["a"] is a
    assert model.get_params()["kernel_params"] is kernel_params
    assert model.get_params()["max_em_iter"] is max_em_iter
    assert model.latent_distribution == "NB"
    assert not hasattr(model, "_fit_state")
    assert not any(name.endswith("_") for name in vars(model))
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_clone_and_set_params_are_available():
    def scaled_linear(X1, X2, scale=1.0):
        return scale * (X1 @ X2.T)

    model = GGKM(kernel=scaled_linear, kernel_params={"scale": 2.5})
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert copied.kernel_params is not model.kernel_params
    assert copied.set_params(gamma=0.3, tol=1e-5) is copied
    assert copied.gamma == 0.3
    assert copied.tol == 1e-5


def test_cloning_a_fitted_estimator_starts_unfitted(fitted_model):
    copied = clone(fitted_model)
    assert copied.get_params() == fitted_model.get_params()
    with pytest.raises(NotFittedError):
        check_is_fitted(copied)


def test_fit_has_the_scikit_survival_signature():
    assert list(inspect.signature(GGKM.fit).parameters) == ["self", "X", "y"]


def test_fit_returns_self_and_keeps_constructor_parameters(survival_data):
    X, y = survival_data
    model = GGKM(
        latent_distribution="NB",
        phi=None,
        a=1.2,
        d=1.3,
        p=1.1,
        gamma=0.3,
        lambda_reg=0.1,
        max_em_iter=2,
    )
    original_params = model.get_params().copy()
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
        assert model.fit(X, y) is model
    assert model.get_params() == original_params
    assert model.a_ > 0 and model.d_ > 0 and model.p_ > 0
    assert model.phi is None
    assert model.phi_ > 0
    assert model.n_features_in_ == X.shape[1]
    assert 1 <= model.n_iter_ <= model.max_em_iter
    assert len(model.loglik_history_) == model.n_iter_
    assert model.final_loglik_ == model.loglik_history_[-1]
    check_is_fitted(model)


@pytest.mark.parametrize(
    "method,args",
    [
        ("predict", ()),
        ("predict_eta", ()),
        ("predict_theta", ()),
        ("predict_cure_probability", ()),
        ("predict_survival", ([0.0, 1.0],)),
        ("predict_survival_function", ()),
        ("predict_cumulative_hazard_function", ()),
    ],
)
def test_prediction_requires_a_successful_fit(method, args):
    with pytest.raises(NotFittedError):
        getattr(GGKM(), method)(np.ones((2, 2)), *args)


@pytest.mark.parametrize(
    "params,exception,match",
    [
        ({"warm_start": "yes"}, TypeError, "warm_start"),
        ({"latent_distribution": "unknown"}, ValueError, "latent_distribution"),
        ({"a": 0.0}, ValueError, "a must"),
        ({"d": np.inf}, ValueError, "d must"),
        ({"p": np.nan}, ValueError, "p must"),
        ({"lambda_reg": -1.0}, ValueError, "lambda_reg"),
        ({"gamma": np.inf}, ValueError, "gamma"),
        ({"degree": 0}, ValueError, "degree"),
        ({"coef0": np.nan}, ValueError, "coef0"),
        ({"A": 0.0}, ValueError, "A must"),
        ({"tol": -1.0}, ValueError, "tol"),
        ({"max_em_iter": 0}, ValueError, "max_em_iter"),
        ({"max_em_iter": 1.5}, (TypeError, ValueError), "max_em_iter"),
        ({"latent_distribution": "negative_binomial", "phi": -1.0}, ValueError, "phi"),
        ({"latent_distribution": "binomial", "K_bin": 0}, ValueError, "K_bin"),
    ],
)
def test_hyperparameters_are_validated_during_fit(
    params, exception, match, survival_data
):
    model = GGKM(**params)
    with pytest.raises(exception, match=match):
        model.fit(*survival_data)


@pytest.mark.parametrize(
    "make_y",
    [
        lambda y: np.column_stack([y["event"], y["time"]]),
        lambda y: np.array(
            list(zip(y["event"].astype(int), y["time"])),
            dtype=[("event", int), ("time", float)],
        ),
        lambda y: y[:-1],
        lambda y: Surv.from_arrays(y["event"], -y["time"]),
        lambda y: Surv.from_arrays(y["event"], np.zeros(len(y))),
        lambda y: Surv.from_arrays(y["event"], np.full(len(y), np.nan)),
        lambda y: Surv.from_arrays(np.zeros(len(y), dtype=bool), y["time"]),
    ],
    ids=[
        "unstructured",
        "integer-event",
        "length",
        "negative-time",
        "zero-time",
        "nonfinite-time",
        "all-censored",
    ],
)
def test_invalid_survival_targets_are_rejected(make_y, survival_data):
    X, y = survival_data
    with pytest.raises(ValueError):
        GGKM(max_em_iter=1).fit(X, make_y(y))


def test_target_field_names_are_not_hardcoded(survival_data):
    X, y = survival_data
    custom_y = Surv.from_arrays(
        y["event"], y["time"], name_event="observed", name_time="duration"
    )
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1).fit(X, custom_y)
    expected = concordance_index_censored(
        custom_y["observed"], custom_y["duration"], model.predict(X)
    )[0]
    assert model.score(X, custom_y) == expected


def test_feature_names_are_recorded_and_validated(survival_data):
    X, y = survival_data
    frame = pd.DataFrame(X, columns=["age", "biomarker"])
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1).fit(frame, y)
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    assert model.predict(frame).shape == (len(frame),)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(frame[["biomarker", "age"]])


@pytest.mark.parametrize(
    "method,args",
    [
        ("predict", ()),
        ("predict_cure_probability", ()),
        ("predict_survival", ([1.0],)),
        ("predict_survival_function", ()),
        ("predict_cumulative_hazard_function", ()),
    ],
)
def test_prediction_validates_feature_count(fitted_model, method, args):
    with pytest.raises(ValueError, match="features"):
        getattr(fitted_model, method)(np.ones((3, 3)), *args)


def test_predict_is_a_risk_score_and_score_is_concordance(fitted_model, survival_data):
    X, y = survival_data
    risk = fitted_model.predict(X)
    np.testing.assert_allclose(risk, fitted_model.predict_eta(X))
    expected = concordance_index_censored(y["event"], y["time"], risk)[0]
    assert fitted_model.score(X, y) == expected
    cure = fitted_model.predict_cure_probability(X)[np.argsort(risk)]
    assert np.all(np.diff(cure) <= 0)


def test_survival_function_matches_continuous_model_on_observed_times(
    fitted_model,
    survival_data,
):
    X, y = survival_data
    times = np.unique(y["time"])
    np.testing.assert_array_equal(fitted_model.unique_times_, times)
    expected = fitted_model.predict_survival(X[:3], times)
    curves = fitted_model.predict_survival_function(X[:3])
    observed = fitted_model.predict_survival_function(X[:3], return_array=True)
    assert isinstance(curves, np.ndarray)
    assert curves.shape == (3,)
    assert observed.shape == (3, len(times))
    np.testing.assert_allclose(observed, expected)
    for curve, row in zip(curves, observed):
        assert isinstance(curve, StepFunction)
        np.testing.assert_array_equal(curve.x, times)
        np.testing.assert_allclose(curve(times), row)


def test_cumulative_hazard_function_is_negative_log_survival(
    fitted_model, survival_data
):
    X, _ = survival_data
    survival = fitted_model.predict_survival_function(X[:3], return_array=True)
    observed = fitted_model.predict_cumulative_hazard_function(X[:3], return_array=True)
    curves = fitted_model.predict_cumulative_hazard_function(X[:3])
    np.testing.assert_allclose(observed, -np.log(survival))
    assert np.all(np.diff(observed, axis=1) >= 0)
    for curve, row in zip(curves, observed):
        assert isinstance(curve, StepFunction)
        np.testing.assert_allclose(curve(fitted_model.unique_times_), row)


def test_continuous_prediction_preserves_time_order_and_single_time_shape(
    fitted_model,
    survival_data,
):
    X, _ = survival_data
    values = fitted_model.predict_survival(X[:3], [3.0, 0.0, 1.0, 1.0])
    assert values.shape == (3, 4)
    np.testing.assert_allclose(values[:, 1], 1.0)
    np.testing.assert_allclose(values[:, 2], values[:, 3])
    assert np.all(values[:, 0] <= values[:, 2])
    assert fitted_model.predict_survival(X[:3], [1.0]).shape == (3, 1)


def test_pipeline_and_grid_search_use_survival_score(survival_data):
    X, y = survival_data
    pipeline = make_pipeline(
        StandardScaler(), GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1)
    )
    search = GridSearchCV(
        pipeline,
        {"ggkm__lambda_reg": [0.1, 0.2]},
        cv=KFold(n_splits=2),
        error_score="raise",
    )
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        search.fit(X, y)
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert search.predict(X[:3]).shape == (3,)
    assert 0.0 <= search.score(X, y) <= 1.0


def test_callable_kernel_receives_kernel_params():
    def scaled_linear(X1, X2, scale=1.0):
        return scale * (X1 @ X2.T)

    model = GGKM(kernel=scaled_linear, kernel_params={"scale": 2.5})
    X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    X2 = np.array([[2.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(model._compute_kernel(X1, X2), 2.5 * (X1 @ X2.T))


def test_callable_kernel_wrong_shape_is_rejected():
    model = GGKM(kernel=lambda X1, X2: np.ones(X1.shape[0]))
    X = np.array([[0.0], [1.0]])
    with pytest.raises(ValueError, match="Kernel must return a matrix with shape"):
        model._compute_kernel(X, X)


def test_callable_kernel_nonfinite_values_are_rejected():
    def bad_kernel(X1, X2):
        K = np.ones((X1.shape[0], X2.shape[0]))
        K[0, 0] = np.nan
        return K

    model = GGKM(kernel=bad_kernel)
    X = np.array([[0.0], [1.0]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        model._compute_kernel(X, X)


def test_asymmetric_training_kernel_is_rejected(survival_data):
    def asymmetric_kernel(X1, X2):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        K[0, 1] = 1.0
        return K

    with pytest.raises(ValueError, match="symmetric training kernel matrix"):
        GGKM(kernel=asymmetric_kernel).fit(*survival_data)


@pytest.mark.parametrize(
    "distribution,kwargs",
    [
        ("poisson", {}),
        ("binomial", {"K_bin": 3}),
        ("negative_binomial", {"phi": 0.7}),
        ("negative_binomial", {"phi": None}),
    ],
)
def test_population_survival_matches_latent_count_distribution(
    distribution,
    kwargs,
    survival_data,
):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
        model = GGKM(
            latent_distribution=distribution,
            gamma=0.3,
            lambda_reg=0.1,
            max_em_iter=2,
            **kwargs,
        ).fit(X, y)
    times = np.array([0.0, 0.5, 1.5, 3.0, 5.0])
    F = gammainc(model.d_ / model.p_, (times / model.a_) ** model.p_)
    eta = model.predict(X[:3])[:, None]
    if distribution == "poisson":
        expected = np.exp(-np.exp(eta) * F)
        expected_cure = np.exp(-np.exp(eta[:, 0]))
    elif distribution == "binomial":
        expected = (1 - expit(eta) * F) ** model.K_bin
        expected_cure = (1 - expit(eta[:, 0])) ** model.K_bin
    else:
        expected = np.exp(-np.log1p(model.phi_ * np.exp(eta) * F) / model.phi_)
        expected_cure = np.exp(-np.log1p(model.phi_ * np.exp(eta[:, 0])) / model.phi_)
    observed = model.predict_survival(X[:3], times)
    cure = model.predict_cure_probability(X[:3])
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cure, expected_cure, rtol=1e-10, atol=1e-12)
    assert np.all(np.diff(observed, axis=1) <= 1e-15)
    assert np.all((observed >= 0) & (observed <= 1))
    assert np.all(observed >= cure[:, None] - 1e-15)
    sampled = model.predict_survival_function(X[:3], return_array=True)
    np.testing.assert_allclose(
        sampled, model.predict_survival(X[:3], model.unique_times_)
    )


@pytest.mark.parametrize("warm_start", [False, True])
@pytest.mark.parametrize("phi", [None, 0.7], ids=["estimated-phi", "fixed-phi"])
def test_refit_initializes_gamma_and_dispersion_as_requested(
    warm_start,
    phi,
    survival_data,
    monkeypatch,
):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(
            latent_distribution="negative_binomial",
            phi=phi,
            warm_start=warm_start,
            gamma=0.3,
            lambda_reg=0.1,
            max_em_iter=1,
        ).fit(X, y)
    previous_gamma = [model.a_, model.d_, model.p_]
    previous_phi = model.phi_
    model.set_params(a=1.2, d=1.4, p=1.1)
    expected_gamma = previous_gamma if warm_start else [1.2, 1.4, 1.1]
    expected_phi = previous_phi if warm_start else (0.5 if phi is None else phi)

    class InitializationChecked(Exception):
        pass

    def check_initialization(*args):
        np.testing.assert_allclose([model.a_, model.d_, model.p_], expected_gamma)
        assert model._latent_model.phi == pytest.approx(expected_phi)
        raise InitializationChecked

    monkeypatch.setattr(model, "_e_step", check_initialization)
    with pytest.raises(InitializationChecked):
        model.fit(X, y)
    assert model.phi is phi
    assert [model.a, model.d, model.p] == [1.2, 1.4, 1.1]


@pytest.mark.parametrize("times", [[], [[1.0]], [-1.0], [np.nan], [np.inf]])
def test_invalid_prediction_times_are_rejected(fitted_model, survival_data, times):
    X, _ = survival_data
    with pytest.raises(ValueError, match="times"):
        fitted_model.predict_survival(X[:2], times)


@pytest.mark.parametrize("wrong_feature_count", [False, True])
def test_failed_refit_invalidates_predictions(survival_data, wrong_feature_count):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1).fit(X, y)
    invalid_X = np.ones((len(y), 3)) if wrong_feature_count else X
    with pytest.raises(ValueError):
        model.fit(invalid_X, y[:-1])
    with pytest.raises(NotFittedError):
        model.predict(X)
    with pytest.raises(NotFittedError):
        model.predict_survival_function(X)


def test_failed_optimization_invalidates_predictions(survival_data, monkeypatch):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1).fit(X, y)

    def failed_e_step(*args):
        raise FloatingPointError("Synthetic optimizer failure")

    monkeypatch.setattr(model, "_e_step", failed_e_step)
    with pytest.raises(FloatingPointError, match="Synthetic optimizer failure"):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("params", [{"tol": -1}, {"kernel": "unknown"}])
def test_failed_refit_with_changed_parameters_invalidates_predictions(
    survival_data, params
):
    X, y = survival_data
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1).fit(X, y)
    model.set_params(**params)
    with pytest.raises(ValueError):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_research_adapter_fits_ggkm_and_moves_solver_settings(survival_data):
    X, y = survival_data
    model = GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=5)
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 1 iterations"):
        fitted = _fit_cure_model(
            model, X, y["time"], y["event"].astype(float), max_em_iter=1, tol=1e-5
        )
    assert fitted is model
    assert model.max_em_iter == 1
    assert model.tol == 1e-5
    assert model.n_iter_ == 1
    np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"]))
    assert np.all(np.isfinite(model.predict(X)))


def test_research_adapter_preserves_legacy_fit_arguments(survival_data):
    class LegacyModel:
        def fit(self, X, t, delta, **fit_params):
            self.received = (X, t, delta, fit_params)
            return self

    X, y = survival_data
    times, events = y["time"], y["event"].astype(float)
    model = LegacyModel()
    assert _fit_cure_model(model, X, times, events, max_em_iter=2, tol=1e-4) is model
    received_X, received_times, received_events, received_params = model.received
    assert received_X is X
    assert received_times is times
    assert received_events is events
    assert received_params == {"max_em_iter": 2, "tol": 1e-4}


@pytest.mark.parametrize(
    "delta",
    [
        np.zeros((24, 1)),
        np.full(24, 2),
        np.full(24, -1),
        np.full(24, 0.5),
        np.full(24, np.nan),
    ],
    ids=["matrix", "two", "negative", "fractional", "nan"],
)
def test_research_adapter_rejects_invalid_events(survival_data, delta):
    X, y = survival_data
    with pytest.raises(ValueError, match="delta must"):
        _fit_cure_model(GGKM(max_em_iter=1), X, y["time"], delta)


# Captured before the estimator API refactor with SciPy 1.18.1.
_LEGACY_EXPECTED = [
    {
        "distribution": "poisson",
        "kwargs": {},
        "eta": [0.5411931656450295, 0.12321309439366376, 0.807357617114774],
        "survival": [
            [
                1.0,
                0.9276776419713962,
                0.5959383658933772,
                0.18670958827259063,
                0.17941466919771454,
            ],
            [
                1.0,
                0.9517766534575735,
                0.7112104145766058,
                0.3312486141503235,
                0.322669909037058,
            ],
            [
                1.0,
                0.9066816868796875,
                0.5089194821991067,
                0.1119203715847358,
                0.10624835073049348,
            ],
        ],
        "cure": [0.17941466919771454, 0.322669909037058, 0.10624835073049348],
        "loglik": -27.184181633399735,
    },
    {
        "distribution": "binomial",
        "kwargs": {"K_bin": 3},
        "eta": [-0.17365917453224455, -0.6780165292661656, 0.28192712616483584],
        "survival": [
            [
                1.0,
                0.9371750625624153,
                0.6329135109809073,
                0.1709066345380942,
                0.16037384408127447,
            ],
            [
                1.0,
                0.9534186842042048,
                0.7186901315261809,
                0.303301536656371,
                0.2918245785510766,
            ],
            [
                1.0,
                0.9220041409654391,
                0.5584371298167782,
                0.0878323115745266,
                0.07949667130415283,
            ],
        ],
        "cure": [0.16037384408127447, 0.2918245785510766, 0.07949667130415283],
        "loglik": -26.939764593408213,
    },
    {
        "distribution": "negative_binomial",
        "kwargs": {"phi": 0.7},
        "eta": [0.705575195943397, 0.28386261520633205, 0.9591918887948586],
        "survival": [
            [
                1.0,
                0.9593486551243129,
                0.665383297213703,
                0.28352453227632307,
                0.2833554674493532,
            ],
            [
                1.0,
                0.9730139275882772,
                0.7558515707552668,
                0.3911524655145042,
                0.3909608064726429,
            ],
            [
                1.0,
                0.9481333109418251,
                0.602772010523468,
                0.22677638855615717,
                0.22662735387939587,
            ],
        ],
        "cure": [0.2833554674493532, 0.3909608064726429, 0.22662735387939587],
        "loglik": -28.428822366335304,
    },
    {
        "distribution": "negative_binomial",
        "kwargs": {"phi": None},
        "eta": [0.5348873511342491, 0.12153970043479934, 0.790998657503557],
        "survival": [
            [
                1.0,
                0.9526426007907364,
                0.6436685883900872,
                0.18329168057499717,
                0.18136314121521496,
            ],
            [
                1.0,
                0.9684197111776601,
                0.7472106677249639,
                0.3255506986701266,
                0.3232810082680534,
            ],
            [
                1.0,
                0.939246831975186,
                0.5659936171393345,
                0.11170098599196321,
                0.11018497933659153,
            ],
        ],
        "cure": [0.18136314121521496, 0.3232810082680534, 0.11018497933659153],
        "loglik": -26.574473284858396,
    },
]


@pytest.mark.parametrize(
    "expected",
    _LEGACY_EXPECTED,
    ids=["poisson", "binomial", "nb-fixed", "nb-estimated"],
)
def test_fixed_kernel_fit_improves_legacy_likelihood_and_matches_count_formulas(
    expected, survival_data
):
    X, y = survival_data
    # Hold the legacy Gram matrix fixed to test the optimizer independently
    # of rounding differences between SciPy's and sklearn's RBF distances.
    K = np.exp(-0.3 * cdist(X, X, metric="sqeuclidean"))
    K_new = np.exp(-0.3 * cdist(X[:3], X, metric="sqeuclidean"))
    with pytest.warns(ConvergenceWarning, match="EM did not converge within 2 iterations"):
        model = GGKM(
            latent_distribution=expected["distribution"],
            gamma=0.3,
            kernel="precomputed",
            lambda_reg=0.1,
            max_em_iter=2,
            **expected["kwargs"],
        ).fit(K, y)
    times = np.array([0.0, 0.5, 1.5, 3.0, 5.0])
    eta = model.beta0_ + K_new @ model.alpha_
    with np.errstate(over="ignore"):
        F = gammainc(model.d_ / model.p_, (times / model.a_) ** model.p_)
    if expected["distribution"] == "binomial":
        probability = expit(eta)
        survival = (1. - probability[:, None] * F) ** model.K_bin
        cure = (1. - probability) ** model.K_bin
    else:
        intensity = np.exp(eta)
        if expected["distribution"] == "poisson":
            survival = np.exp(-intensity[:, None] * F)
            cure = np.exp(-intensity)
        else:
            survival = np.exp(-np.log1p(model.phi_ * intensity[:, None] * F) / model.phi_)
            cure = np.exp(-np.log1p(model.phi_ * intensity) / model.phi_)
    np.testing.assert_allclose(model.predict(K_new), eta, rtol=1e-13)
    np.testing.assert_allclose(model.predict_survival(K_new, times), survival, rtol=1e-12)
    np.testing.assert_allclose(model.predict_cure_probability(K_new), cure, rtol=1e-12)
    # Stable tail derivatives change the historical L-BFGS path. Compare the
    # attained likelihood, rather than requiring the old suboptimal solution.
    assert model.final_loglik_ >= expected["loglik"] - 1e-6
