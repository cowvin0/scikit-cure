"""Boosting estimator contracts, legacy tree updates, and numerical formulas.

The saved legacy results come from an independent, unmodified source snapshot.
Free optimization is compared separately from the frozen-latency tree updates:
using the stable generalized-gamma tail can change an optimizer's trajectory.
"""

import inspect
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, gammainc, gammaincc, gammaln
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

import skcure
import skcure.ensemble
from skcure.base import BaseCureEstimator
from skcure.ensemble import GradientBoostingCure
import skcure.ensemble._gb as boosting_module
import skcure._latent as latent_module

LATENT_DISTRIBUTIONS = ["binomial", "negative_binomial", "poisson"]
# Historical class names identify the independent reference data's provenance.
FIXTURE_FAMILIES = {
    "GGBinomialGB": "binomial",
    "GGNBGB": "negative_binomial",
    "GGPoissonGB": "poisson",
}
LEGACY = json.loads((Path(__file__).parent / "data" / "boost_legacy.json").read_text())
CASES = list(LEGACY["cases"])


@pytest.fixture(scope="module")
def survival_data():
    return np.asarray(LEGACY["X"]), Surv.from_arrays(LEGACY["event"], LEGACY["time"])


@pytest.fixture(scope="module")
def fitted_models(survival_data):
    X, y = survival_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fitted = {}
        for name, case in LEGACY["cases"].items():
            with pytest.warns(
                ConvergenceWarning, match="EM did not converge within 3 iterations"
            ):
                fitted[name] = GradientBoostingCure(
                    latent_distribution=FIXTURE_FAMILIES[case["class_name"]],
                    **case["params"],
                    **LEGACY["fit_params"],
                ).fit(X, y)
        return fitted


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_survival_estimator_contract(distribution):
    a = np.float64(1.2)
    n = np.int64(2)
    model = GradientBoostingCure(
        latent_distribution=distribution, a=a, n_estimators=n, max_em_iter=1
    )
    assert all(
        isinstance(model, parent)
        for parent in (
            BaseCureEstimator,
            BaseEstimator,
            SurvivalAnalysisMixin,
        )
    )
    assert model.get_params()["a"] is a
    assert model.get_params()["n_estimators"] is n
    assert list(inspect.signature(model.fit).parameters) == ["X", "y"]
    assert not any(name.endswith("_") for name in vars(model))
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert copied.set_params(tol=1e-5) is copied
    with pytest.raises(NotFittedError):
        check_is_fitted(copied)


def test_public_exports():
    for package in (skcure, skcure.ensemble):
        assert package.GradientBoostingCure is GradientBoostingCure
        assert "GradientBoostingCure" in package.__all__


@pytest.mark.parametrize("case_name", CASES)
def test_fit_preserves_constructor_parameters(case_name, fitted_models, survival_data):
    X, y = survival_data
    model = fitted_models[case_name]
    case = LEGACY["cases"][case_name]
    for name, value in dict(case["params"], **LEGACY["fit_params"]).items():
        assert model.get_params()[name] == value
    assert model.n_features_in_ == X.shape[1]
    assert model.a_ > 0 and model.d_ > 0 and model.p_ > 0
    assert model.n_iter_ == len(model.loglik_history_)
    assert model.final_loglik_ == model.loglik_history_[-1]
    assert np.isfinite(model.final_loglik_)
    assert len(model.models_) == len(model.model_weights_)
    assert 1 <= len(model.models_) <= model.n_estimators * model.n_iter_
    if model.latent_distribution == "negative_binomial":
        assert model.phi_ > 0
        if model.phi is not None:
            assert model.phi_ == model.phi
    np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"]))
    check_is_fitted(model)


@pytest.mark.parametrize("case_name", CASES)
def test_tree_ensemble_reproduces_training_predictor(
    case_name, fitted_models, survival_data
):
    X, _ = survival_data
    model = fitted_models[case_name]
    eta = np.full(len(X), np.mean(model.initial_eta_), dtype=float)
    for tree, step in zip(model.models_, model.model_weights_):
        assert 0 < step <= model.learning_rate * 10
        eta += step * tree.predict(X)
    np.testing.assert_allclose(model.eta_train_, eta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model.predict_eta(X), eta, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(model.predict(X), eta, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("case_name", CASES)
def test_frozen_latency_retains_original_boosting_updates(
    case_name, monkeypatch, survival_data
):
    """Isolate the original E-step, gradient, trees, and line search from GG optimization."""
    X, y = survival_data
    case = LEGACY["cases"][case_name]
    model = GradientBoostingCure(
        latent_distribution=FIXTURE_FAMILIES[case["class_name"]],
        **case["params"],
        **LEGACY["fit_params"],
    )
    monkeypatch.setattr(model, "_gg_m_step", lambda *args, **kwargs: None)
    if model.latent_distribution == "negative_binomial":
        monkeypatch.setattr(model, "_phi_m_step", lambda *args, **kwargs: None)
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 3 iterations"
    ):
        model.fit(X, y)
    expected = case["fixed_gg_steps"]
    # The safeguarded analytic line search replaces the historical numerical
    # derivative. Retain the independent fixture as a close reference, with
    # tolerances accounting for its finite-difference optimization accuracy.
    np.testing.assert_allclose(model.predict_eta(X), expected["eta"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        model.model_weights_, expected["weights"], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        [tree.predict(X) for tree in model.models_],
        expected["tree_predictions"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.loglik_history_, expected["loglik_history"], rtol=0, atol=5e-6
    )


@pytest.mark.parametrize("case_name", CASES)
def test_first_full_em_update_matches_original_likelihood(case_name, fitted_models):
    # Later optimizer trajectories diverge in these near-degenerate fits. The
    # original first EM update remains a useful independent numerical reference.
    observed = fitted_models[case_name].loglik_history_[0]
    expected = LEGACY["cases"][case_name]["learned"]["loglik_history_"][0]
    assert observed == pytest.approx(expected, rel=0, abs=1e-6)


@pytest.mark.parametrize("case_name", CASES)
def test_score_and_pickle(case_name, fitted_models, survival_data):
    X, y = survival_data
    model = fitted_models[case_name]
    risk = model.predict(X)
    assert (
        model.score(X, y) == concordance_index_censored(y["event"], y["time"], risk)[0]
    )
    assert np.all(np.diff(model.predict_cure_probability(X)[np.argsort(risk)]) <= 1e-14)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), risk)
    np.testing.assert_array_equal(
        restored.predict_survival_function(X, return_array=True),
        model.predict_survival_function(X, return_array=True),
    )
    assert restored.get_params() == model.get_params()


@pytest.mark.parametrize("case_name", CASES)
def test_survival_and_cure_match_independent_count_formulas(
    case_name, fitted_models, survival_data
):
    X, _ = survival_data
    model = fitted_models[case_name]
    times = np.array([0.0, 0.5, 1.5, 3.0, 5.0])
    eta = model.predict_eta(X[:4])[:, None]
    with np.errstate(over="ignore"):
        F = gammainc(model.d_ / model.p_, (times / model.a_) ** model.p_)
    if model.latent_distribution == "binomial":
        expected = (1 - expit(eta) * F) ** model.K_bin
        cure = (1 - expit(eta[:, 0])) ** model.K_bin
    elif model.latent_distribution == "negative_binomial":
        expected = np.exp(-np.log1p(model.phi_ * np.exp(eta) * F) / model.phi_)
        cure = np.exp(-np.log1p(model.phi_ * np.exp(eta[:, 0])) / model.phi_)
    else:
        expected = np.exp(-np.exp(eta) * F)
        cure = np.exp(-np.exp(eta[:, 0]))
    np.testing.assert_allclose(
        model.predict_survival(X[:4], times), expected, rtol=1e-12, atol=1e-14
    )
    np.testing.assert_allclose(
        model.predict_cure_probability(X[:4]), cure, rtol=1e-12, atol=1e-14
    )
    values = model.predict_survival_function(X[:4], return_array=True)
    curves = model.predict_survival_function(X[:4])
    hazards = model.predict_cumulative_hazard_function(X[:4], return_array=True)
    np.testing.assert_allclose(hazards, -np.log(values), rtol=1e-12, atol=1e-14)
    for curve, row in zip(curves, values):
        assert isinstance(curve, StepFunction)
        np.testing.assert_array_equal(curve(model.unique_times_), row)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_e_step_and_observed_likelihood_match_formulas(
    distribution, monkeypatch, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution=distribution,
        a=1.2,
        d=1.3,
        p=1.1,
        n_estimators=1,
        max_em_iter=1,
    )
    monkeypatch.setattr(model, "_gg_m_step", lambda *args, **kwargs: None)
    if model.latent_distribution == "negative_binomial":
        monkeypatch.setattr(model, "_phi_m_step", lambda *args, **kwargs: None)
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model.fit(X, y)
    eta = np.linspace(-2, 2, len(X))
    model.eta_train_ = eta
    t, delta = y["time"], y["event"].astype(float)
    u = (t / model.a_) ** model.p_
    F, S = gammainc(model.d_ / model.p_, u), gammaincc(model.d_ / model.p_, u)
    log_f = (
        np.log(model.p_)
        + (model.d_ - 1) * np.log(t)
        - u
        - model.d_ * np.log(model.a_)
        - gammaln(model.d_ / model.p_)
    )
    if model.latent_distribution == "binomial":
        theta = expit(eta)
        counts = delta + (model.K_bin - delta) * theta * S / (1 - theta + theta * S)
        ll = np.sum(
            delta * (np.log(model.K_bin) + np.log(theta) + log_f)
            + (model.K_bin - delta) * np.log1p(-theta * F)
        )
    elif model.latent_distribution == "negative_binomial":
        theta = np.exp(eta)
        counts = delta + (1 + model.phi_ * delta) * theta * S / (
            1 + model.phi_ * theta * F
        )
        ll = np.sum(
            delta * (eta + log_f)
            - (1 / model.phi_ + delta) * np.log1p(model.phi_ * theta * F)
        )
    else:
        theta = np.exp(eta)
        counts = delta + theta * S
        ll = np.sum(delta * (eta + log_f) - theta * F)
    np.testing.assert_allclose(model._e_step(t, delta), counts, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        model._loglikelihood(t, delta), ll, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("case_name", CASES)
def test_complete_data_functional_gradient(case_name, fitted_models):
    model = fitted_models[case_name]
    eta = np.array([-3.0, -0.4, 0.2, 2.0])
    Nhat = np.array([0.2, 0.7, 1.3, 2.1])

    def objective(z):
        if model.latent_distribution == "binomial":
            return np.sum(Nhat * z - model.K_bin * np.logaddexp(0, z))
        if model.latent_distribution == "negative_binomial":
            return np.sum(
                Nhat * z
                - (Nhat + 1 / model.phi_) * np.logaddexp(0, np.log(model.phi_) + z)
            )
        return np.sum(Nhat * z - np.exp(z))

    h = 1e-5
    finite_difference = np.array(
        [
            (objective(eta + h * row) - objective(eta - h * row)) / (2 * h)
            for row in np.eye(len(eta))
        ]
    )
    np.testing.assert_allclose(
        model._functional_gradient(eta, Nhat), finite_difference, rtol=2e-6, atol=1e-7
    )


@pytest.mark.parametrize("case_name", CASES)
def test_gg_objective_gradient(case_name, fitted_models, survival_data):
    model = fitted_models[case_name]
    _, y = survival_data
    t, delta = y["time"], y["event"].astype(float)
    pars = np.array([1.2, 1.3, 1.1])
    counts = delta + np.linspace(0.1, 0.8, len(t))
    value, grad = model._gg_value_and_grad(pars, t, delta, counts)
    u = (t / pars[0]) ** pars[2]
    log_f = (
        np.log(pars[2])
        + (pars[1] - 1) * np.log(t)
        - u
        - pars[1] * np.log(pars[0])
        - gammaln(pars[1] / pars[2])
    )
    expected = -np.sum(
        delta * log_f + (counts - delta) * np.log(gammaincc(pars[1] / pars[2], u))
    )
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
    h = 1e-5
    numerical = np.array(
        [
            (
                model._gg_value_and_grad(pars + h * row, t, delta, counts)[0]
                - model._gg_value_and_grad(pars - h * row, t, delta, counts)[0]
            )
            / (2 * h)
            for row in np.eye(3)
        ]
    )
    np.testing.assert_allclose(grad, numerical, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("case_name", CASES)
def test_extreme_predictors_preserve_probabilities(
    case_name, monkeypatch, fitted_models, survival_data
):
    model = fitted_models[case_name]
    X, _ = survival_data
    eta = np.array([-1e4, 0.0, 1e4])
    monkeypatch.setattr(model, "predict_eta", lambda X: eta[: len(X)])
    times = np.array([0.0, 1e-200, 1.0, 1e200])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        survival = model.predict_survival(X[:3], times)
        cure = model.predict_cure_probability(X[:3])
    assert np.isfinite(survival).all() and np.isfinite(cure).all()
    assert np.all((survival >= 0) & (survival <= 1))
    np.testing.assert_array_equal(survival[:, 0], 1)
    np.testing.assert_allclose(survival[:, -1], cure, rtol=1e-12, atol=1e-14)
    assert np.all(np.diff(survival, axis=1) <= 1e-14)
    assert cure[0] > 1 - 1e-12 and cure[-1] < 1e-12


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    "pars", [[0, 1, 1], [1, -1, 1], [1, 1, np.inf], [np.nan, 1, 1]]
)
def test_invalid_optimizer_parameters_return_penalty(
    case_name, pars, fitted_models, survival_data
):
    _, y = survival_data
    delta = y["event"].astype(float)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        value, grad = fitted_models[case_name]._gg_value_and_grad(
            pars, y["time"], delta, delta + 0.2
        )
    assert not np.isnan(value) and value > 1e90
    # Poisson uses (inf, NaNs) to reject a trial; the other latent models use
    # a finite penalty with a zero gradient. Both signal an invalid objective.
    assert np.shape(grad) == (3,)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
@pytest.mark.parametrize(
    "method,args",
    [
        ("predict", ()),
        ("predict_eta", ()),
        ("predict_theta", ()),
        ("predict_survival", ([0.0, 1.0],)),
        ("predict_cure_probability", ()),
        ("predict_survival_function", ()),
        ("predict_cumulative_hazard_function", ()),
    ],
)
def test_predictions_require_fitting(distribution, method, args):
    model = GradientBoostingCure(latent_distribution=distribution)
    with pytest.raises(NotFittedError):
        getattr(model, method)(np.ones((2, 2)), *args)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
@pytest.mark.parametrize(
    "params",
    [
        {"a": 0},
        {"d": np.inf},
        {"p": np.nan},
        {"tol": 0},
        {"max_em_iter": 0},
        {"max_em_iter": True},
        {"learning_rate": 0},
        {"learning_rate": 1.1},
        {"learning_rate": np.inf},
        {"n_estimators": 0},
        {"n_estimators": 1.5},
        {"n_estimators": True},
        {"max_depth": 0},
        {"min_samples_leaf": 0},
    ],
)
def test_hyperparameter_validation(distribution, params, survival_data):
    with pytest.raises((ValueError, TypeError)):
        GradientBoostingCure(latent_distribution=distribution, **params).fit(
            *survival_data
        )


@pytest.mark.parametrize("phi", [0.0, -1.0, np.nan, np.inf])
def test_dispersion_validation(phi, survival_data):
    with pytest.raises(ValueError, match="phi"):
        GradientBoostingCure(
            latent_distribution="negative_binomial", phi=phi, max_em_iter=1
        ).fit(*survival_data)


@pytest.mark.parametrize("K_bin", [0, -1, 1.5, True])
def test_binomial_size_validation(K_bin, survival_data):
    with pytest.raises((ValueError, TypeError), match="K_bin"):
        GradientBoostingCure(
            latent_distribution="binomial", K_bin=K_bin, max_em_iter=1
        ).fit(*survival_data)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_refit_is_reproducible_and_resets_ensemble(distribution, survival_data):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution=distribution,
        n_estimators=2,
        max_em_iter=1,
        min_samples_leaf=3,
        random_state=73,
    )
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model.fit(X, y)
    old_predictions = model.predict(X).copy()
    old_a, old_d, old_p = model.a_, model.d_, model.p_
    old_models = list(model.models_)
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), old_predictions)
    assert (model.a_, model.d_, model.p_) == (old_a, old_d, old_p)
    assert len(model.models_) == len(old_models)
    assert all(tree is not old for tree, old in zip(model.models_, old_models))


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_pipeline_and_dataframe_features(distribution, survival_data):
    X, y = survival_data
    frame = pd.DataFrame(X, columns=["first", "second"])
    model = GradientBoostingCure(
        latent_distribution=distribution,
        n_estimators=1,
        max_em_iter=1,
        min_samples_leaf=3,
        random_state=73,
    )
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        pipeline = make_pipeline(StandardScaler(), model).fit(frame, y)
    assert np.isfinite(pipeline.predict(frame)).all()
    assert np.isfinite(pipeline.score(frame, y))
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        copied = clone(model).fit(frame, y)
    np.testing.assert_array_equal(copied.feature_names_in_, frame.columns)
    with pytest.raises(ValueError, match="feature"):
        copied.predict(frame[["second", "first"]])


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
@pytest.mark.parametrize(
    "target",
    [
        "plain",
        "non_boolean_event",
        "zero_time",
        "negative_time",
        "nan_time",
        "wrong_length",
    ],
)
def test_structured_target_validation(distribution, target, survival_data):
    X, y = survival_data
    bad = y.copy()
    if target == "plain":
        bad = np.column_stack([y["event"], y["time"]])
    elif target == "non_boolean_event":
        bad = np.array(
            list(zip(y["event"].astype(int), y["time"])),
            dtype=[("event", int), ("time", float)],
        )
    elif target == "wrong_length":
        bad = bad[:-1]
    else:
        bad["time"][0] = {"zero_time": 0, "negative_time": -1, "nan_time": np.nan}[
            target
        ]
    model = GradientBoostingCure(
        latent_distribution=distribution, n_estimators=1, max_em_iter=1
    )
    with pytest.raises((ValueError, TypeError)):
        model.fit(X, bad)
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("case_name", CASES)
def test_latency_optimizer_retries_in_log_parameters(
    case_name, monkeypatch, fitted_models, survival_data
):
    model = pickle.loads(pickle.dumps(fitted_models[case_name]))
    _, y = survival_data
    # A retry must still pass fresh objective and non-worsening checks. Use
    # the current valid point to isolate the raw-to-log retry contract.
    parameters = np.array([model.a_, model.d_, model.p_])
    calls = []

    def controlled_minimize(fun, x0, **kwargs):
        calls.append((fun.__name__, np.asarray(x0)))
        if len(calls) == 1:
            return SimpleNamespace(
                success=False,
                fun=np.inf,
                x=np.asarray(x0),
                message="controlled raw failure",
            )
        return SimpleNamespace(
            success=True,
            fun=1.0,
            x=np.log(parameters),
            message="controlled log success",
        )

    monkeypatch.setattr(boosting_module, "minimize", controlled_minimize)
    model._gg_m_step(
        y["time"], y["event"].astype(float), y["event"].astype(float) + 0.2
    )
    assert [name for name, _ in calls] == [
        "_gg_value_and_grad",
        "_gg_log_value_and_grad",
    ]
    np.testing.assert_allclose(calls[1][1], np.log(calls[0][1]), rtol=0, atol=0)
    np.testing.assert_allclose([model.a_, model.d_, model.p_], parameters, rtol=1e-14)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_failed_optimizer_invalidates_previous_fit(
    distribution, monkeypatch, survival_data
):
    X, y = survival_data
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model = GradientBoostingCure(
            latent_distribution=distribution,
            n_estimators=1,
            max_em_iter=1,
            min_samples_leaf=3,
            random_state=73,
        ).fit(X, y)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled optimizer failure")

    monkeypatch.setattr(model, "_gg_m_step", fail)
    with pytest.raises(RuntimeError, match="controlled optimizer failure"):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
@pytest.mark.parametrize("random_state", [-1, "invalid", 1.5])
def test_random_state_validation(distribution, random_state, survival_data):
    with pytest.raises(ValueError):
        GradientBoostingCure(
            latent_distribution=distribution,
            n_estimators=1,
            max_em_iter=1,
            random_state=random_state,
        ).fit(*survival_data)


@pytest.mark.parametrize("case_name", ["binomial", "nb_fixed"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_successful_raw_optimizer_with_invalid_objective_retries(
    case_name,
    invalid_value,
    monkeypatch,
    fitted_models,
    survival_data,
):
    model = pickle.loads(pickle.dumps(fitted_models[case_name]))
    _, y = survival_data
    parameters = np.array([model.a_, model.d_, model.p_])
    calls = []

    def controlled_minimize(fun, x0, **kwargs):
        calls.append(fun.__name__)
        value = invalid_value if len(calls) == 1 else 1.0
        coordinates = parameters if len(calls) == 1 else np.log(parameters)
        return SimpleNamespace(
            success=True,
            fun=value,
            x=coordinates,
            message="controlled optimizer success",
        )

    monkeypatch.setattr(boosting_module, "minimize", controlled_minimize)
    delta = y["event"].astype(float)
    model._gg_m_step(y["time"], delta, delta + 0.2)
    assert calls == ["_gg_value_and_grad", "_gg_log_value_and_grad"]
    np.testing.assert_allclose([model.a_, model.d_, model.p_], parameters, rtol=1e-14)


@pytest.mark.parametrize("case_name", ["binomial", "nb_fixed"])
@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_successful_log_optimizer_with_invalid_objective_is_rejected(
    case_name,
    invalid_value,
    monkeypatch,
    fitted_models,
    survival_data,
):
    model = pickle.loads(pickle.dumps(fitted_models[case_name]))
    _, y = survival_data
    original_parameters = (model.a_, model.d_, model.p_)
    calls = []

    def controlled_minimize(fun, x0, **kwargs):
        calls.append(fun.__name__)
        return SimpleNamespace(
            success=len(calls) == 2,
            fun=invalid_value if len(calls) == 2 else np.inf,
            x=np.asarray(x0),
            message="controlled optimizer result",
        )

    monkeypatch.setattr(boosting_module, "minimize", controlled_minimize)
    delta = y["event"].astype(float)
    with pytest.raises(RuntimeError, match="Generalized-gamma"):
        model._gg_m_step(y["time"], delta, delta + 0.2)
    assert calls == ["_gg_value_and_grad", "_gg_log_value_and_grad"]
    assert (model.a_, model.d_, model.p_) == original_parameters


@pytest.mark.parametrize("invalid_part", ["value", "gradient"])
def test_dispersion_fallback_rejects_invalid_objective(
    invalid_part,
    monkeypatch,
    fitted_models,
    survival_data,
):
    model = pickle.loads(pickle.dumps(fitted_models["nb_estimated"]))
    _, y = survival_data
    calls = []
    phi_before = model._latent_model.phi

    def controlled_minimize(fun, x0, **kwargs):
        calls.append(fun.__name__)

        if fun.__name__ == "phi_value_and_grad":
            return SimpleNamespace(
                success=False,
                fun=np.inf,
                x=np.asarray(x0),
                message="controlled natural-phi failure",
            )

        return SimpleNamespace(
            success=True,
            fun=1.0,
            x=np.array([np.log(phi_before)]),
            message="controlled log-phi success",
        )

    monkeypatch.setattr(
        latent_module,
        "minimize",
        controlled_minimize,
    )

    def phi_value_and_grad(*args):
        if invalid_part == "value":
            return np.inf, np.array([0.0])
        return 1.0, np.array([np.nan])

    monkeypatch.setattr(
        model._latent_model,
        "phi_value_and_grad",
        phi_value_and_grad,
    )

    t = y["time"]
    delta = y["event"].astype(float)

    with pytest.raises(FloatingPointError, match="Dispersion"):
        model._phi_m_step(t, delta)

    assert calls == [
        "phi_value_and_grad",
        "log_phi_value_and_grad",
    ]
    assert model._latent_model.phi == pytest.approx(phi_before)


@pytest.mark.parametrize("distribution", LATENT_DISTRIBUTIONS)
def test_maximum_integer_seed_supports_multiple_trees(distribution, survival_data):
    X, y = survival_data
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 1 iterations"
    ):
        model = GradientBoostingCure(
            latent_distribution=distribution,
            n_estimators=2,
            max_em_iter=1,
            min_samples_leaf=3,
            random_state=2**32 - 1,
        ).fit(X, y)
    assert len(model.models_) == 2
    assert [tree.random_state for tree in model.models_] == [2**32 - 1, 0]
    assert np.isfinite(model.predict(X)).all()
