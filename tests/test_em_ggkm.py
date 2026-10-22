"""GGKM regression tests frozen before extracting the EM implementation.

The reference records both observable estimator results and actual SciPy
optimizer calls. Nested structure and non-floating values are compared exactly.
Floating-point values are compared numerically so the regression suite remains
portable across supported numerical environments.

Poisson uses a slightly wider tolerance because its latency evaluation now
delegates to the shared latency distribution, including the numerically stable
survival-function implementation, rather than the legacy ``1 - CDF`` path.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.optimize import minimize as scipy_minimize
from sklearn.metrics.pairwise import pairwise_kernels
from sksurv.util import Surv

from skcure.ggkm import GGKM
from skcure._latency import _GeneralizedGammaLatencyOptimizer

REFERENCE_PATH = Path(__file__).parent / "data" / "ggkm_em_reference.json"

_STRICT_RTOL = 1e-12
_STRICT_ATOL = 1e-14
_POISSON_RTOL = 5e-3
_POISSON_ATOL = 1e-6


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
        "poisson_default_iteration_limit": {"max_em_iter": None, "tol": 1e6},
        "nb_default_iteration_limit": {
            "latent_distribution": "nb",
            "max_em_iter": None,
            "tol": 1e6,
        },
        "binomial_default_iteration_limit": {
            "latent_distribution": "binomial",
            "max_em_iter": None,
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


def _run_case(case, model_class, monkeypatch):
    X, y = _data()
    X_test = np.array([[0.2, -0.4], [1.1, 0.7], [-0.3, 0.2]])
    parameters = {"gamma": 0.3, "lambda_reg": 0.1, "max_em_iter": 3}
    parameters.update(_cases()[case])
    if parameters.get("kernel") == "precomputed":
        X_test = pairwise_kernels(X_test, X, metric="rbf", gamma=0.3)
        X = pairwise_kernels(X, metric="rbf", gamma=0.3)
    model = model_class(**parameters)
    runs = []
    with _record_optimizers(monkeypatch) as trace:
        model.fit(X, y)
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
            model.fit(X, y)
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
    with _record_optimizers(monkeypatch) as trace:
        try:
            model.fit(X, y)
        except Exception as exc:
            error = [type(exc).__name__, str(exc)]
        else:
            raise AssertionError("Expected the injected failure to propagate")
    state = _state(model)
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


def test_initial_loglik_validation_preserves_nb_legacy_semantics():
    model = GGKM()
    model._latent_model = SimpleNamespace(name="negative_binomial")

    assert model._validate_initial_loglik(np.nan) is None


@pytest.mark.parametrize("name", ["poisson", "binomial"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_initial_loglik_validation_requires_finite_value(name, value):
    model = GGKM()
    model._latent_model = SimpleNamespace(name=name)

    with pytest.raises(
        FloatingPointError,
        match="Initial penalized log-likelihood is non-finite",
    ):
        model._validate_initial_loglik(value)


def test_gg_latency_result_validation_preserves_nb_legacy_semantics():
    """
    NB historically accepts an optimizer result with finite parameters even
    when result.fun is non-finite. Preserve that behavior during extraction.
    """
    optimizer = _GeneralizedGammaLatencyOptimizer()
    latent_model = SimpleNamespace(name="negative_binomial")
    result = SimpleNamespace(
        success=True,
        fun=np.inf,
        x=np.ones(3),
        message="controlled result",
    )

    optimizer._validate_result(latent_model, result)


@pytest.mark.parametrize("name", ["poisson", "binomial"])
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


def test_gg_latency_log_parameter_validation_preserves_poisson_legacy_semantics():
    """
    Poisson historically skipped the post-optimizer log-parameter overflow
    check. Preserve that behavior while relocating the validation.
    """
    optimizer = _GeneralizedGammaLatencyOptimizer()
    latent_model = SimpleNamespace(name="poisson")
    log_pars = np.full(
        3,
        generalized_gamma_log_overflow_value(),
    )

    optimizer._validate_log_parameters(latent_model, log_pars)


@pytest.mark.parametrize("name", ["negative_binomial", "binomial"])
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

    distribution = _cases()[case].get("latent_distribution", "poisson")
    if distribution == "poisson":
        rtol = _POISSON_RTOL
        atol = _POISSON_ATOL
    else:
        rtol = _STRICT_RTOL
        atol = _STRICT_ATOL

    _assert_reference(
        observed,
        em_reference["successful_fits"][case],
        case,
        rtol=rtol,
        atol=atol,
    )


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

    if distribution == "poisson":
        rtol = _POISSON_RTOL
        atol = _POISSON_ATOL
    else:
        rtol = _STRICT_RTOL
        atol = _STRICT_ATOL

    _assert_reference(
        observed,
        em_reference["failed_refits"][key],
        key,
        rtol=rtol,
        atol=atol,
    )
