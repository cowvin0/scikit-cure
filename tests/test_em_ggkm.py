from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sksurv.util import Surv

from skcure.ggkm import GGKM
from skcure._latency import _GeneralizedGammaLatencyOptimizer

REFERENCE_PATH = Path(__file__).parent / "data" / "ggkm_em_reference.json"

_STRICT_RTOL = 1e-12
_STRICT_ATOL = 1e-14


def _scaled_rbf(X, Y, scale=1.25):
    """Exercise the existing matrix-valued callable interface."""
    return scale * pairwise_kernels(X, Y, metric="rbf", gamma=0.3)


def _data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    time = np.round(rng.uniform(0.4, 3.2, size=24), 2)
    event = np.tile([True, False, True, True], 6)
    return X, Surv.from_arrays(event, time)


def _cases():
    cases = {
        "poisson_rbf": {},
        "negative_binomial_fixed": {"latent_distribution": "nb", "phi": 0.7},
        "negative_binomial_estimated": {"latent_distribution": "nb"},
        "bernoulli": {"latent_distribution": "binomial", "K_bin": 1},
        "binomial_three": {"latent_distribution": "binomial", "K_bin": 3},
        "linear": {"kernel": "linear", "max_em_iter": 2},
        "polynomial": {
            "kernel": "poly",
            "degree": 2,
            "coef0": 1.0,
            "latent_distribution": "nb",
            "phi": 0.7,
            "max_em_iter": 2,
        },
        "cauchy": {
            "kernel": "cauchy",
            "latent_distribution": "binomial",
            "K_bin": 3,
            "max_em_iter": 2,
        },
        "callable": {
            "kernel": _scaled_rbf,
            "kernel_params": {"scale": 1.25},
            "max_em_iter": 2,
        },
        "precomputed": {
            "kernel": "precomputed",
            "latent_distribution": "nb",
            "phi": 0.7,
            "max_em_iter": 2,
        },
        "early_convergence": {"tol": 1e6},
        "iteration_cap": {"max_em_iter": 1},
        "nondefault_initial_values": {"a": 1.2, "d": 0.9, "p": 1.1},
        "poisson_default_iteration_limit": {"tol": 1e6},
        "nb_default_iteration_limit": {
            "latent_distribution": "nb",
            "tol": 1e6,
        },
        "binomial_default_iteration_limit": {
            "latent_distribution": "binomial",
            "tol": 1e6,
        },
        "cold_refit": {},
        "warm_poisson_refit": {"warm_start": True},
        "warm_nb_refit": {"warm_start": True, "latent_distribution": "nb"},
        "warm_fixed_nb_refit": {
            "warm_start": True,
            "latent_distribution": "nb",
            "phi": 0.7,
        },
        "warm_binomial_refit": {
            "warm_start": True,
            "latent_distribution": "binomial",
            "K_bin": 3,
        },
        "warm_changed_distribution": {"warm_start": True, "latent_distribution": "nb"},
        "warm_changed_phi": {"warm_start": True, "latent_distribution": "nb"},
        "warm_changed_trials": {"warm_start": True, "latent_distribution": "binomial"},
    }
    return cases


def _serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(item) for item in value]
    if callable(value):
        return value.__name__
    return value


@contextmanager
def _record_optimizers(monkeypatch):
    """Trace real optimization across any extraction of the model module."""
    trace = []

    def traced_minimize(*args, **kwargs):
        call = {key: _serial(value) for key, value in kwargs.items()}
        if args:
            call["positional_arguments"] = _serial(args)
        trace.append(call)
        try:
            result = scipy_minimize(*args, **kwargs)
        except Exception as exc:
            call["error"] = [type(exc).__name__, str(exc)]
            raise
        call["result"] = {
            key: _serial(getattr(result, key))
            for key in ("x", "fun", "jac", "success", "status", "nit", "nfev", "njev")
        }
        return result

    with monkeypatch.context() as patch:
        for name, module in list(sys.modules.items()):
            if (
                name.startswith("skcure.")
                and getattr(module, "minimize", None) is scipy_minimize
            ):
                patch.setattr(module, "minimize", traced_minimize)
        yield trace


def _state(model):
    state = {
        "constructor_parameters": _serial(model.get_params()),
        "generalized_gamma": _serial(vars(model._latency)),
        "latent_parameters": _serial(vars(model._latent_model)),
        "fit_state": _serial(
            {
                "beta0": getattr(model, "beta0_", None),
                "alpha": getattr(model, "alpha_", None),
                "K": getattr(model, "K_", None),
                "X_train": getattr(model, "X_train_", None),
                "loglik_history": getattr(
                    model,
                    "loglik_history_",
                    [],
                ),
                "converged": getattr(
                    model,
                    "converged_",
                    False,
                ),
                "n_iter": getattr(
                    model,
                    "n_iter_",
                    0,
                ),
                "final_loglik": getattr(
                    model,
                    "final_loglik_",
                    None,
                ),
            }
        ),
        # "fit_state": _serial(vars(model._fit_state)),
        "latent_configuration": _serial(model._latent_configuration),
        "is_fitted": model.__sklearn_is_fitted__(),
        "n_features_in": model.n_features_in_,
    }
    if hasattr(model, "unique_times_"):
        state["unique_times"] = _serial(model.unique_times_)
    if hasattr(model, "phi_"):
        state["phi"] = model.phi_
        state["estimate_phi"] = model.estimate_phi
    return state


def _predictions(model, X, y, X_test):
    predictions = {
        "risk": model.predict(X_test),
        "eta": model.predict_eta(X_test),
        "cure": model.predict_cure_probability(X_test),
        "continuous_survival": model.predict_survival(
            X_test, [3.1, 0.0, 1.0, 5.5, 1.0]
        ),
        "score": model.score(X, y),
        "survival_array": model.predict_survival_function(X_test, return_array=True),
        "hazard_array": model.predict_cumulative_hazard_function(
            X_test, return_array=True
        ),
    }
    for name, method in (
        ("survival_steps", model.predict_survival_function),
        ("hazard_steps", model.predict_cumulative_hazard_function),
    ):
        predictions[name] = [
            {
                "x": fn.x,
                "y": fn.y,
                "a": fn.a,
                "b": fn.b,
                "evaluated": fn(model.unique_times_),
            }
            for fn in method(X_test)
        ]
    try:
        predictions["theta"] = model.predict_theta(X_test)
    except (AttributeError, RuntimeError) as exc:
        predictions["theta_error"] = [type(exc).__name__, str(exc)]
    return _serial(predictions)


def _assert_fitted_equations(model, X, y, X_test):
    """Check the fitted model without assuming a particular optimizer path."""
    assert model.__sklearn_is_fitted__()
    assert 1 <= model.n_iter_ <= model.max_em_iter
    assert len(model.loglik_history_) == model.n_iter_
    assert np.all(np.isfinite(model.loglik_history_))
    assert model.final_loglik_ == model.loglik_history_[-1]
    assert np.all(np.diff(model.loglik_history_) >= -1e-6)
    assert np.all(np.isfinite([model.a_, model.d_, model.p_]))
    assert np.all(np.array([model.a_, model.d_, model.p_]) > 0.0)
    np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"]))

    eta = model.beta0_ + model.K_ @ model.alpha_
    np.testing.assert_allclose(model.predict(X), eta, rtol=1e-13, atol=1e-14)
    time = y["time"]
    delta = y["event"].astype(float)
    F = model._latency.cdf(time)
    S = model._latency.sf(time)
    log_density = model._latency.logpdf(time)
    event_mask = delta > 0.5
    event_sum = np.sum(eta[event_mask] + log_density[event_mask])
    theta = np.exp(eta)
    family = model._latent_model.name
    if family == "poisson":
        log_likelihood = event_sum - np.sum(theta * F)
        conditional_count = delta + theta * S
    elif family == "negative_binomial":
        phi = model.phi_
        log_likelihood = event_sum - np.sum(
            (1.0 / phi + delta) * np.log1p(phi * theta * F)
        )
        conditional_count = delta + (1.0 + phi * delta) * theta * S / (
            1.0 + phi * theta * F
        )
    else:
        K = model._latent_model.K_bin
        probability = expit(eta)
        event_sum = np.sum(
            np.log(K) + np.log(probability[event_mask]) + log_density[event_mask]
        )
        log_likelihood = event_sum + np.sum((K - delta) * np.log1p(-probability * F))
        conditional_count = delta + (K - delta) * probability * S / (
            1.0 - probability * F
        )
    penalty = 0.5 * model.lambda_reg * (model.alpha_ @ model.K_ @ model.alpha_)
    np.testing.assert_allclose(
        model.final_loglik_, log_likelihood - penalty, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        model._e_step(model.beta0_, model.alpha_, time, delta),
        conditional_count,
        rtol=1e-11,
        atol=1e-12,
    )

    # Check both the intercept and a nontrivial kernel-coefficient direction.
    parameters = np.r_[model.beta0_, model.alpha_]
    value, gradient = model._predictor_value_and_grad(parameters, conditional_count)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    directions = [np.r_[1.0, np.zeros(len(model.alpha_))]]
    direction = np.random.default_rng(19).normal(size=len(parameters))
    directions.append(direction / np.linalg.norm(direction))
    for direction in directions:
        step = 1e-5 * direction
        finite_difference = (
            model._predictor_value_and_grad(parameters + step, conditional_count)[0]
            - model._predictor_value_and_grad(parameters - step, conditional_count)[0]
        ) / 2e-5
        np.testing.assert_allclose(
            gradient @ direction, finite_difference, rtol=1e-5, atol=1e-7
        )

    times = np.array([0.0, 0.5, 1.0, 3.0, 5.5])
    eta_test = model.predict_eta(X_test)
    test_F = model._latency.cdf(times)[None, :]
    if family == "poisson":
        expected_log_survival = -np.exp(eta_test[:, None]) * test_F
        expected_cure = np.exp(-np.exp(eta_test))
    elif family == "negative_binomial":
        expected_log_survival = (
            -np.log1p(model.phi_ * np.exp(eta_test[:, None]) * test_F) / model.phi_
        )
        expected_cure = np.exp(-np.log1p(model.phi_ * np.exp(eta_test)) / model.phi_)
    else:
        probability = expit(eta_test)
        K = model._latent_model.K_bin
        expected_log_survival = K * np.log1p(-probability[:, None] * test_F)
        expected_cure = (1.0 - probability) ** K
    np.testing.assert_allclose(
        model.predict_log_survival(X_test, times),
        expected_log_survival,
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.predict_survival(X_test, times),
        np.exp(expected_log_survival),
        rtol=1e-11,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.predict_cure_probability(X_test), expected_cure, rtol=1e-11, atol=1e-12
    )


def _run_case(case, model_class, monkeypatch):
    X, y = _data()
    X_test = np.array([[0.2, -0.4], [1.1, 0.7], [-0.3, 0.2]])
    parameters = {"gamma": 0.3, "lambda_reg": 0.1}
    if not case.endswith("default_iteration_limit"):
        parameters["max_em_iter"] = 3
    parameters.update(_cases()[case])
    if parameters.get("kernel") == "precomputed":
        X_test = pairwise_kernels(X_test, X, metric="rbf", gamma=0.3)
        X = pairwise_kernels(X, metric="rbf", gamma=0.3)
    model = model_class(**parameters)
    runs = []
    with _record_optimizers(monkeypatch) as trace:
        warning_context = (
            nullcontext()
            if parameters.get("tol") == 1e6
            else pytest.warns(
                ConvergenceWarning,
                match=f"EM did not converge within {model.max_em_iter} iterations",
            )
        )
        with warning_context:
            model.fit(X, y)
        _assert_fitted_equations(model, X, y, X_test)
        runs.append(
            {"state": _state(model), "predictions": _predictions(model, X, y, X_test)}
        )
        if "refit" in case or case.startswith("warm_changed"):
            changes = {
                "warm_changed_distribution": {
                    "latent_distribution": "binomial",
                    "K_bin": 3,
                },
                "warm_changed_phi": {"phi": 0.7},
                "warm_changed_trials": {"K_bin": 3},
            }
            model.set_params(**changes.get(case, {}))
            with pytest.warns(
                ConvergenceWarning,
                match=f"EM did not converge within {model.max_em_iter} iterations",
            ):
                model.fit(X, y)
            _assert_fitted_equations(model, X, y, X_test)
            runs.append(
                {
                    "state": _state(model),
                    "predictions": _predictions(model, X, y, X_test),
                }
            )
    return {"runs": runs, "optimizers": trace}


def _run_failure(distribution, stage, model_class, monkeypatch):
    X, y = _data()
    model = model_class(
        latent_distribution=distribution,
        gamma=0.3,
        lambda_reg=0.1,
        max_em_iter=2,
        warm_start=True,
    )
    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 2 iterations"
    ):
        model.fit(X, y)
    initial_state = _state(model)

    def failed(*args, **kwargs):
        raise ArithmeticError("injected " + stage + " failure")

    if stage in ("initial_likelihood", "post_m_likelihood"):
        original_likelihood = model._penalized_loglikelihood
        calls = 0

        def likelihood(*args, **kwargs):
            nonlocal calls
            calls += 1
            if stage == "initial_likelihood":
                failed()
            if calls == 2:
                return np.nan
            return original_likelihood(*args, **kwargs)

        monkeypatch.setattr(model, "_penalized_loglikelihood", likelihood)
    elif stage == "finalize":
        # The warm fit reuses this same latent model instance.
        monkeypatch.setattr(model._latent_model, "finalize_fit", failed)
    else:
        method = {
            "e_step": "_e_step",
            "predictor": "_predictor_value_and_grad",
            "generalized_gamma": "_gg_log_value_and_grad",
            "phi": "_update_phi",
        }[stage]
        monkeypatch.setattr(model, method, failed)
    warning_context = (
        pytest.warns(
            ConvergenceWarning, match="EM did not converge within 2 iterations"
        )
        if stage == "finalize"
        else nullcontext()
    )
    with _record_optimizers(monkeypatch) as trace, warning_context:
        try:
            model.fit(X, y)
        except Exception as exc:
            error = [type(exc).__name__, str(exc)]
        else:
            raise AssertionError("Expected the injected failure to propagate")
    state = _state(model)
    with pytest.raises(NotFittedError):
        model.predict(X)
    # An instance method installed above is test instrumentation, not fit state.
    state["latent_parameters"].pop("finalize_fit", None)
    return {
        "initial_state": initial_state,
        "error": error,
        "state": state,
        "optimizers": trace,
    }


def _assert_reference(
    actual,
    expected,
    path="result",
    *,
    rtol=_STRICT_RTOL,
    atol=_STRICT_ATOL,
):
    """Compare nested regression output with numerical float tolerances."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _assert_reference(
                actual[key],
                expected[key],
                path + "." + key,
                rtol=rtol,
                atol=atol,
            )
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (item, reference) in enumerate(zip(actual, expected)):
            _assert_reference(
                item,
                reference,
                path + f"[{i}]",
                rtol=rtol,
                atol=atol,
            )
    elif isinstance(expected, float):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=path,
        )
    else:
        assert actual == expected, path


@pytest.mark.parametrize("name", ["poisson", "negative_binomial", "binomial"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_initial_loglik_validation_requires_finite_value(name, value):
    model = GGKM()
    model._latent_model = SimpleNamespace(name=name)

    with pytest.raises(
        FloatingPointError,
        match="Initial penalized log-likelihood is non-finite",
    ):
        model._validate_initial_loglik(value)


@pytest.mark.parametrize("name", ["poisson", "negative_binomial", "binomial"])
def test_gg_latency_result_validation_requires_finite_fun(name):
    optimizer = _GeneralizedGammaLatencyOptimizer()
    latent_model = SimpleNamespace(name=name)
    result = SimpleNamespace(
        success=True,
        fun=np.inf,
        x=np.ones(3),
        message="controlled non-finite objective",
    )

    with pytest.raises(
        RuntimeError,
        match="Generalized-gamma log-parameter optimization failed",
    ):
        optimizer._validate_result(latent_model, result)


@pytest.mark.parametrize("name", ["poisson", "negative_binomial", "binomial"])
def test_gg_latency_log_parameter_validation_rejects_overflow(name):
    optimizer = _GeneralizedGammaLatencyOptimizer()
    latent_model = SimpleNamespace(name=name)
    log_pars = np.full(
        3,
        generalized_gamma_log_overflow_value(),
    )

    with pytest.raises(
        FloatingPointError,
        match="outside the representable floating-point range",
    ):
        optimizer._validate_log_parameters(latent_model, log_pars)


def generalized_gamma_log_overflow_value():
    return np.log(np.finfo(float).max) + 1.0


@pytest.fixture(scope="module")
def em_reference():
    return json.loads(REFERENCE_PATH.read_text())


@pytest.mark.parametrize("case", list(_cases()))
def test_ggkm_em_reference(case, em_reference, monkeypatch):
    observed = _run_case(case, GGKM, monkeypatch)
    expected = em_reference["successful_fits"][case]
    assert len(observed["runs"]) == len(expected["runs"])
    for run, historical in zip(observed["runs"], expected["runs"]):
        state = run["state"]
        historical_state = historical["state"]
        parameters = dict(historical_state["constructor_parameters"])
        parameters.update(em_rtol=0.0, solver_options=None)
        if case.endswith("default_iteration_limit"):
            assert parameters["max_em_iter"] is None
            parameters["max_em_iter"] = 1000
        _assert_reference(state["constructor_parameters"], parameters)
        for key in ("latent_configuration", "n_features_in", "unique_times"):
            _assert_reference(state[key], historical_state[key])
        for key in ("K", "X_train"):
            _assert_reference(
                state["fit_state"][key], historical_state["fit_state"][key]
            )
        assert state["is_fitted"] is True
        predictions = run["predictions"]
        np.testing.assert_array_equal(predictions["risk"], predictions["eta"])
        assert 0.0 <= predictions["score"] <= 1.0
        for array_name, step_name in (
            ("survival_array", "survival_steps"),
            ("hazard_array", "hazard_steps"),
        ):
            for row, step in zip(predictions[array_name], predictions[step_name]):
                np.testing.assert_array_equal(step["x"], state["unique_times"])
                np.testing.assert_array_equal(step["y"], row)
                np.testing.assert_array_equal(step["evaluated"], row)

    # Initial predictor coordinates and posterior counts are unchanged by the
    # new tail/dispersion formulas; optimizer iterates and call counts are not.
    first = observed["optimizers"][0]
    original_first = expected["optimizers"][0]
    for key in ("fun", "x0", "jac", "args", "method"):
        _assert_reference(first[key], original_first[key])


@pytest.mark.parametrize("distribution", ["poisson", "nb", "binomial"])
@pytest.mark.parametrize(
    "stage",
    [
        "initial_likelihood",
        "e_step",
        "predictor",
        "generalized_gamma",
        "phi",
        "post_m_likelihood",
        "finalize",
    ],
)
def test_ggkm_em_failure_state_reference(
    distribution, stage, em_reference, monkeypatch
):
    observed = _run_failure(distribution, stage, GGKM, monkeypatch)
    key = distribution + ":" + stage
    historical = em_reference["failed_refits"][key]
    _assert_reference(observed["error"], historical["error"])
    assert observed["initial_state"]["is_fitted"] is True
    state = observed["state"]
    assert state["is_fitted"] is False
    expected_parameters = dict(historical["state"]["constructor_parameters"])
    expected_parameters.update(em_rtol=0.0, solver_options=None)
    _assert_reference(state["constructor_parameters"], expected_parameters)
    for field in ("latent_configuration", "n_features_in", "unique_times"):
        _assert_reference(state[field], historical["state"][field])
    progress = state["fit_state"]
    for field in ("K", "X_train"):
        _assert_reference(progress[field], historical["state"]["fit_state"][field])
    assert np.all(np.isfinite(list(state["generalized_gamma"].values())))
    assert np.all(np.array(list(state["generalized_gamma"].values())) > 0.0)
    if stage == "finalize":
        assert 1 <= progress["n_iter"] <= 2
        assert len(progress["loglik_history"]) == progress["n_iter"]
        assert np.all(np.isfinite(progress["loglik_history"]))
        assert progress["final_loglik"] == progress["loglik_history"][-1]
        assert np.isfinite(progress["beta0"])
        assert np.all(np.isfinite(progress["alpha"]))
    else:
        assert progress["n_iter"] == (0 if stage == "initial_likelihood" else 1)
        assert progress["loglik_history"] == []
        assert progress["converged"] is False
        assert progress["beta0"] is None
        assert progress["alpha"] is None
        assert progress["final_loglik"] is None
    if stage in ("initial_likelihood", "e_step"):
        assert observed["optimizers"] == []
    elif stage in ("predictor", "generalized_gamma"):
        assert observed["optimizers"][-1]["error"][0] == "ArithmeticError"
