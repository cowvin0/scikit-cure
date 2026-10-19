# """Estimator contracts and regression against the original research models.

# ``data/legacy_models.json`` was captured before modifying either implementation.
# It stores full fitted outputs, rather than recomputing an expectation with the
# same implementation under test.
# """

# import inspect
# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import pytest
# from scipy.special import expit, gammainc
# from sklearn.base import BaseEstimator, clone
# from sklearn.exceptions import NotFittedError
# from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.utils.validation import check_is_fitted
# from sksurv.base import SurvivalAnalysisMixin
# from sksurv.functions import StepFunction
# from sksurv.metrics import concordance_index_censored
# from sksurv.util import Surv

# import skcure
# import skcure.models
# from skcure.base import BaseCureEstimator
# from skcure.models.baseline_model import (
#     MixtureCureModel,
#     PromotionTimeModel,
# )


# ESTIMATORS = [
#     MixtureCureModel, PromotionTimeModel,
# ]
# ESTIMATOR_BY_NAME = {cls.__name__: cls for cls in ESTIMATORS}
# LEGACY = json.loads((Path(__file__).parent / "data" / "legacy_models.json").read_text())
# CASE_NAMES = [name for name, case in LEGACY["cases"].items()
#               if case["class"] in ESTIMATOR_BY_NAME]


# @pytest.fixture(scope="module")
# def survival_data():
#     X = np.asarray(LEGACY["X"])
#     y = Surv.from_arrays(LEGACY["event"], LEGACY["time"])
#     return X, y


# @pytest.fixture(scope="module")
# def regression_models(survival_data):
#     X, y = survival_data
#     return {
#         name: ESTIMATOR_BY_NAME[case["class"]](
#             **case["params"], **LEGACY["fit_params"]
#         ).fit(X, y)
#         for name, case in LEGACY["cases"].items()
#         if case["class"] in ESTIMATOR_BY_NAME
#     }


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_fitted_results_match_unmodified_research_models(
#     case_name, regression_models, survival_data,
# ):
#     X, _ = survival_data
#     model = regression_models[case_name]
#     expected = LEGACY["cases"][case_name]
#     np.testing.assert_allclose(model.predict_eta(X), expected["eta"], rtol=1e-10, atol=1e-12)
#     np.testing.assert_allclose(
#         model.predict_survival(X, LEGACY["prediction_times"]), expected["survival"],
#         rtol=1e-10, atol=1e-12,
#     )
#     np.testing.assert_allclose(
#         model.predict_cure_probability(X), expected["cure"], rtol=1e-10, atol=1e-12,
#     )
#     for name, value in expected["learned"].items():
#         if name in {"a", "d", "p", "phi"}:
#             name += "_"
#         observed = getattr(model, name)
#         np.testing.assert_allclose(observed, value, rtol=1e-10, atol=1e-12, err_msg=name)
#     if "spline_design" in expected:
#         np.testing.assert_allclose(model.spline_.transform(X), expected["spline_design"])


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_estimators_share_the_survival_contract(estimator_class):
#     initial_a = np.float64(1.2)
#     initial_iterations = np.int64(2)
#     model = estimator_class(a=initial_a, max_em_iter=initial_iterations)
#     assert all(isinstance(model, parent) for parent in (
#         BaseCureEstimator, BaseEstimator, SurvivalAnalysisMixin,
#     ))
#     assert model.get_params()["a"] is initial_a
#     assert model.get_params()["max_em_iter"] is initial_iterations
#     assert not any(name.endswith("_") for name in vars(model))
#     assert list(inspect.signature(model.fit).parameters) == ["X", "y"]
#     copied = clone(model)
#     assert copied.get_params() == model.get_params()
#     assert copied.set_params(tol=1e-5) is copied
#     with pytest.raises(NotFittedError):
#         check_is_fitted(copied)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_estimators_are_exported_from_public_namespaces(estimator_class):
#     name = estimator_class.__name__
#     assert getattr(skcure, name) is estimator_class
#     assert getattr(skcure.models, name) is estimator_class
#     assert name in skcure.__all__
#     assert name in skcure.models.__all__


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_constructor_parameters_remain_initial_values(case_name, regression_models, survival_data):
#     X, _ = survival_data
#     model = regression_models[case_name]
#     case = LEGACY["cases"][case_name]
#     for name, value in dict(case["params"], **LEGACY["fit_params"]).items():
#         assert model.get_params()[name] == value
#     assert model.a_ > 0 and model.d_ > 0 and model.p_ > 0
#     assert model.n_features_in_ == X.shape[1]
#     assert model.n_iter_ == len(model.loglik_history_)
#     assert model.final_loglik_ == model.loglik_history_[-1]
#     check_is_fitted(model)
#     copied = clone(model)
#     assert copied.get_params() == model.get_params()
#     with pytest.raises(NotFittedError):
#         copied.predict(X)


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_coefficients_reproduce_the_fitted_predictor(case_name, regression_models, survival_data):
#     X, _ = survival_data
#     model = regression_models[case_name]
#     design = model.spline_.transform(X) if hasattr(model, "spline_") else X
#     assert model.coef_.shape == (design.shape[1],)
#     np.testing.assert_allclose(
#         design @ model.coef_ + model.intercept_, model.predict_eta(X),
#         rtol=1e-12, atol=1e-12,
#     )


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_risk_prediction_and_concordance_score(case_name, regression_models, survival_data):
#     X, y = survival_data
#     model = regression_models[case_name]
#     risk = model.predict(X)
#     np.testing.assert_allclose(risk, model.predict_eta(X))
#     expected = concordance_index_censored(y["event"], y["time"], risk)[0]
#     assert model.score(X, y) == expected
#     cure = model.predict_cure_probability(X)[np.argsort(risk)]
#     assert np.all(np.diff(cure) <= 1e-14)


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_survival_curves_and_cumulative_hazards(case_name, regression_models, survival_data):
#     X, y = survival_data
#     model = regression_models[case_name]
#     np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"]))
#     expected = model.predict_survival(X[:3], model.unique_times_)
#     curves = model.predict_survival_function(X[:3])
#     values = model.predict_survival_function(X[:3], return_array=True)
#     hazards = model.predict_cumulative_hazard_function(X[:3], return_array=True)
#     hazard_curves = model.predict_cumulative_hazard_function(X[:3])
#     assert isinstance(curves, np.ndarray) and curves.shape == (3,)
#     assert values.shape == (3, len(model.unique_times_))
#     np.testing.assert_allclose(values, expected)
#     np.testing.assert_allclose(hazards, -np.log(values))
#     assert np.all(np.diff(values, axis=1) <= 1e-14)
#     for curve, hazard_curve, row, hazard_row in zip(curves, hazard_curves, values, hazards):
#         assert isinstance(curve, StepFunction) and isinstance(hazard_curve, StepFunction)
#         np.testing.assert_allclose(curve(model.unique_times_), row)
#         np.testing.assert_allclose(hazard_curve(model.unique_times_), hazard_row)


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_population_survival_matches_distribution(case_name, regression_models, survival_data):
#     X, _ = survival_data
#     model = regression_models[case_name]
#     times = np.array([0.0, 0.5, 1.5, 3.0, 5.0])
#     with np.errstate(over="ignore"):
#         F = gammainc(model.d_ / model.p_, (times / model.a_) ** model.p_)
#     eta = model.predict_eta(X[:3])[:, None]
#     if isinstance(model, MixtureCureModel):
#         expected = 1 - expit(eta) * F
#         expected_cure = 1 - expit(eta[:, 0])
#     else:
#         expected = np.exp(-np.exp(eta) * F)
#         expected_cure = np.exp(-np.exp(eta[:, 0]))
#     np.testing.assert_allclose(model.predict_survival(X[:3], times), expected)
#     np.testing.assert_allclose(model.predict_cure_probability(X[:3]), expected_cure)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# @pytest.mark.parametrize("method,args", [
#     ("predict", ()), ("predict_eta", ()), ("predict_cure_probability", ()),
#     ("predict_survival", ([0.0, 1.0],)), ("predict_survival_function", ()),
#     ("predict_cumulative_hazard_function", ()),
# ])
# def test_prediction_requires_fitting(estimator_class, method, args):
#     with pytest.raises(NotFittedError):
#         getattr(estimator_class(), method)(np.ones((2, 2)), *args)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# @pytest.mark.parametrize("params", [
#     {"a": 0.0}, {"d": np.inf}, {"p": np.nan}, {"tol": -1.0},
#     {"max_em_iter": 0}, {"max_em_iter": 1.5}, {"warm_start": "yes"},
# ])
# def test_fit_validates_hyperparameters(estimator_class, params, survival_data):
#     model = estimator_class(**params)
#     with pytest.raises((ValueError, TypeError)):
#         model.fit(*survival_data)


# @pytest.mark.parametrize("estimator_class", [MixtureCureModel, PromotionTimeModel])
# def test_intercept_option_is_validated(estimator_class, survival_data):
#     with pytest.raises((TypeError, ValueError), match="fit_intercept"):
#         estimator_class(fit_intercept="yes").fit(*survival_data)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# @pytest.mark.parametrize("make_y", [
#     lambda y: np.column_stack([y["event"], y["time"]]),
#     lambda y: np.array(list(zip(y["event"].astype(int), y["time"])),
#                        dtype=[("event", int), ("time", float)]),
#     lambda y: y[:-1],
#     lambda y: Surv.from_arrays(y["event"], -y["time"]),
#     lambda y: Surv.from_arrays(y["event"], np.zeros(len(y))),
#     lambda y: Surv.from_arrays(y["event"], np.full(len(y), np.nan)),
#     lambda y: Surv.from_arrays(np.zeros(len(y), dtype=bool), y["time"]),
# ], ids=["unstructured", "integer-event", "length", "negative-time",
#         "zero-time", "nonfinite-time", "all-censored"])
# def test_fit_validates_survival_targets(estimator_class, make_y, survival_data):
#     X, y = survival_data
#     with pytest.raises(ValueError):
#         estimator_class(max_em_iter=1).fit(X, make_y(y))


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# @pytest.mark.parametrize("times", [[], [[1.0]], [-1.0], [np.nan], [np.inf]])
# def test_prediction_validates_times(case_name, times, regression_models, survival_data):
#     X, _ = survival_data
#     with pytest.raises(ValueError, match="times"):
#         regression_models[case_name].predict_survival(X[:2], times)


# @pytest.mark.parametrize("case_name", CASE_NAMES)
# def test_prediction_validates_feature_count_and_finiteness(case_name, regression_models):
#     model = regression_models[case_name]
#     with pytest.raises(ValueError, match="features"):
#         model.predict(np.ones((3, 3)))
#     with pytest.raises(ValueError):
#         model.predict(np.full((3, 2), np.nan))


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_fit_returns_self_and_cold_refit_is_reproducible(estimator_class, survival_data):
#     X, y = survival_data
#     model = estimator_class(max_em_iter=1)
#     assert model.fit(X, y) is model
#     first_risk = model.predict(X)
#     first_gamma = [model.a_, model.d_, model.p_]
#     assert model.fit(X, y) is model
#     np.testing.assert_allclose(model.predict(X), first_risk, rtol=1e-12, atol=1e-12)
#     np.testing.assert_allclose([model.a_, model.d_, model.p_], first_gamma)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_failed_refit_invalidates_predictions(estimator_class, survival_data):
#     X, y = survival_data
#     model = estimator_class(max_em_iter=1).fit(X, y)
#     with pytest.raises(ValueError):
#         model.fit(X, y[:-1])
#     with pytest.raises(NotFittedError):
#         model.predict(X)
#     with pytest.raises(NotFittedError):
#         model.predict_survival_function(X)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_changed_invalid_parameter_invalidates_predictions(estimator_class, survival_data):
#     X, y = survival_data
#     model = estimator_class(max_em_iter=1).fit(X, y)
#     model.set_params(tol=-1)
#     with pytest.raises(ValueError):
#         model.fit(X, y)
#     with pytest.raises(NotFittedError):
#         model.predict(X)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_failed_optimization_invalidates_predictions(estimator_class, survival_data, monkeypatch):
#     X, y = survival_data
#     model = estimator_class(max_em_iter=1).fit(X, y)
#     inner_class = model.estimator_class if hasattr(model, "estimator_class") else estimator_class

#     def failed_e_step(*args):
#         raise FloatingPointError("Synthetic optimizer failure")

#     monkeypatch.setattr(inner_class, "_e_step", failed_e_step)
#     with pytest.raises(FloatingPointError, match="Synthetic optimizer failure"):
#         model.fit(X, y)
#     with pytest.raises(NotFittedError):
#         model.predict(X)
#     with pytest.raises(NotFittedError):
#         model.predict_survival_function(X)


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# @pytest.mark.parametrize("warm_start", [False, True])
# def test_refit_starts_from_requested_gamma_parameters(
#     estimator_class, warm_start, survival_data, monkeypatch,
# ):
#     X, y = survival_data
#     model = estimator_class(max_em_iter=1, warm_start=warm_start).fit(X, y)
#     fitted_gamma = [model.a_, model.d_, model.p_]
#     fitted_phi = getattr(model, "phi_", None)
#     model.set_params(a=1.2, d=1.4, p=1.1)
#     expected_gamma = fitted_gamma if warm_start else [1.2, 1.4, 1.1]
#     expected_phi = fitted_phi if warm_start else 0.5
#     inner_class = model.estimator_class if hasattr(model, "estimator_class") else estimator_class

#     class InitializationChecked(Exception):
#         pass

#     def check_initialization(inner, *args):
#         np.testing.assert_allclose([inner.a_, inner.d_, inner.p_], expected_gamma)
#         if hasattr(inner, "phi_"):
#             assert inner.phi_ == expected_phi
#         raise InitializationChecked

#     monkeypatch.setattr(inner_class, "_e_step", check_initialization)
#     with pytest.raises(InitializationChecked):
#         model.fit(X, y)
#     assert [model.a, model.d, model.p] == [1.2, 1.4, 1.1]


# @pytest.mark.parametrize("estimator_class", ESTIMATORS)
# def test_feature_names_and_custom_target_fields(estimator_class, survival_data):
#     X, y = survival_data
#     frame = pd.DataFrame(X, columns=["age", "biomarker"])
#     custom_y = Surv.from_arrays(y["event"], y["time"], name_event="observed", name_time="duration")
#     model = estimator_class(max_em_iter=1).fit(frame, custom_y)
#     np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
#     assert model.predict(frame).shape == (len(frame),)
#     assert 0 <= model.score(frame, custom_y) <= 1
#     with pytest.raises(ValueError, match="feature names"):
#         model.predict(frame[["biomarker", "age"]])


# def test_pipeline_and_grid_search_use_survival_score(survival_data):
#     X, y = survival_data
#     pipeline = make_pipeline(StandardScaler(), PromotionTimeModel(max_em_iter=1))
#     search = GridSearchCV(
#         pipeline, {"promotiontimemodel__fit_intercept": [True, False]},
#         cv=KFold(n_splits=2), error_score="raise",
#     ).fit(X, y)
#     assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
#     assert search.predict(X[:3]).shape == (3,)
#     assert 0 <= search.score(X, y) <= 1
