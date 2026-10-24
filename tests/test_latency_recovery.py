"""Recovery from finite L-BFGS-B latency line-search failures."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sklearn.exceptions import ConvergenceWarning
from sksurv.util import Surv

import skcure._latency as latency_module
from skcure._latency import _GeneralizedGammaLatencyOptimizer
from skcure.distributions import GeneralizedGamma
from skcure.ggkm import GGKM


def _quadratic(parameters, *args):
    displacement = np.asarray(parameters) + 0.5
    return float(displacement @ displacement), 2.0 * displacement


def _failed_result(parameters, *, status=2, message="ABNORMAL: "):
    return OptimizeResult(
        success=False,
        status=status,
        message=message,
        x=np.asarray(parameters, dtype=float),
        # A failed line search can cache a rejected trial's objective.
        fun=np.inf,
        jac=np.ones(3),
    )


def _maximize(latency, objective, **kwargs):
    _GeneralizedGammaLatencyOptimizer().maximize(
        latency,
        SimpleNamespace(name="negative_binomial"),
        objective,
        np.ones(2),
        np.zeros(2),
        np.ones(2),
        **kwargs,
    )


@pytest.mark.parametrize("failed_point", [np.full(3, -0.25), np.ones(3)])
def test_line_search_retry_uses_fresh_values_and_best_start(failed_point, monkeypatch):
    original_minimize = latency_module.minimize
    calls = []
    diagnostics = []

    def controlled_minimize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _failed_result(failed_point)
        return original_minimize(**kwargs)

    monkeypatch.setattr(latency_module, "minimize", controlled_minimize)
    latency = GeneralizedGamma()
    _maximize(latency, _quadratic, result_callback=diagnostics.append)

    assert len(calls) == 2
    expected_start = failed_point if np.all(failed_point < 0.0) else np.zeros(3)
    np.testing.assert_array_equal(calls[1]["x0"], expected_start)
    assert calls[1]["options"] == {"maxls": 100}
    assert [result.success for result in diagnostics] == [False, True]
    np.testing.assert_allclose([latency.a, latency.d, latency.p], np.exp(-0.5))


def test_line_search_retry_preserves_explicit_options(monkeypatch):
    original_minimize = latency_module.minimize
    options = {"maxls": 4, "maxiter": 6, "ftol": 1e-10}
    calls = []

    def controlled_minimize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _failed_result(np.full(3, -0.25))
        return original_minimize(**kwargs)

    monkeypatch.setattr(latency_module, "minimize", controlled_minimize)
    _maximize(GeneralizedGamma(), _quadratic, options=options)

    assert len(calls) == 2
    assert calls[0]["options"] == calls[1]["options"] == options
    assert calls[1]["options"] is not options
    assert options == {"maxls": 4, "maxiter": 6, "ftol": 1e-10}


@pytest.mark.parametrize(
    "status,message",
    [(1, "TOTAL NO. OF ITERATIONS REACHED LIMIT"), (2, "OTHER ERROR")],
)
def test_other_optimizer_failures_are_not_retried(status, message, monkeypatch):
    calls = []

    def failed_minimize(**kwargs):
        calls.append(kwargs)
        return _failed_result(np.zeros(3), status=status, message=message)

    monkeypatch.setattr(latency_module, "minimize", failed_minimize)
    latency = GeneralizedGamma()
    with pytest.raises(RuntimeError, match="optimization failed"):
        _maximize(latency, _quadratic)
    assert len(calls) == 1
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


def test_failed_line_search_retry_still_raises_and_records_both_attempts(monkeypatch):
    diagnostics = []
    monkeypatch.setattr(
        latency_module, "minimize", lambda **kwargs: _failed_result(np.zeros(3))
    )
    latency = GeneralizedGamma()
    with pytest.raises(RuntimeError, match="ABNORMAL"):
        _maximize(latency, _quadratic, result_callback=diagnostics.append)
    assert len(diagnostics) == 2
    assert not any(result.success for result in diagnostics)
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("invalid_part", ["value", "gradient"])
def test_line_search_retry_rejects_invalid_fresh_candidate(invalid_part, monkeypatch):
    calls = []

    def failed_minimize(**kwargs):
        calls.append(kwargs)
        return _failed_result(np.ones(3))

    def objective(parameters, *args):
        if np.any(parameters):
            if invalid_part == "value":
                return np.nan, np.ones(3)
            return 0.0, np.full(3, np.nan)
        return _quadratic(parameters)

    monkeypatch.setattr(latency_module, "minimize", failed_minimize)
    latency = GeneralizedGamma()
    with pytest.raises(RuntimeError, match="optimization failed"):
        _maximize(latency, objective)
    assert len(calls) == 1
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


def test_line_search_retry_keeps_non_worsening_safeguard(monkeypatch):
    calls = []

    def controlled_minimize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _failed_result(np.zeros(3))
        return OptimizeResult(
            success=True,
            x=np.ones(3),
            fun=0.0,
            jac=np.zeros(3),
            message="controlled successful status",
        )

    monkeypatch.setattr(latency_module, "minimize", controlled_minimize)
    latency = GeneralizedGamma()
    _maximize(latency, _quadratic)
    assert len(calls) == 2
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("invalid_part", ["value", "gradient"])
def test_successful_retry_still_requires_valid_fresh_objective(invalid_part, monkeypatch):
    calls = []

    def controlled_minimize(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _failed_result(np.zeros(3))
        return OptimizeResult(
            success=True,
            x=np.ones(3),
            fun=0.0,
            jac=np.zeros(3),
            message="controlled successful status",
        )

    def objective(parameters, *args):
        if np.any(parameters):
            if invalid_part == "value":
                return np.nan, np.ones(3)
            return 0.0, np.full(3, np.nan)
        return _quadratic(parameters)

    monkeypatch.setattr(latency_module, "minimize", controlled_minimize)
    latency = GeneralizedGamma()
    with pytest.raises(RuntimeError, match="final objective or gradient is invalid"):
        _maximize(latency, objective)
    assert len(calls) == 2
    assert (latency.a, latency.d, latency.p) == (1.0, 1.0, 1.0)


def test_ggkm_warm_nb_refit_recovers_genuine_line_search_failure(monkeypatch):
    """Exercise the Windows failure path using a genuinely exhausted search."""
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, size=24), 2),
    )
    model = GGKM(
        gamma=0.3,
        lambda_reg=0.1,
        max_em_iter=3,
        warm_start=True,
        latent_distribution="nb",
    )
    with pytest.warns(ConvergenceWarning, match="EM did not converge"):
        model.fit(X, y)

    original_minimize = latency_module.minimize
    results = []

    def limited_first_search(**kwargs):
        if not results:
            kwargs = {**kwargs, "options": {**(kwargs["options"] or {}), "maxls": 1}}
        result = original_minimize(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(latency_module, "minimize", limited_first_search)
    with pytest.warns(ConvergenceWarning, match="EM did not converge"):
        model.fit(X, y)

    assert results[0].status == 2
    assert "ABNORMAL" in str(results[0].message).upper()
    assert results[1].success
    latency_diagnostics = [
        result for result in model.optimizer_results_ if result["stage"] == "latency"
    ]
    assert [result["success"] for result in latency_diagnostics[:2]] == [False, True]
    assert sum(result["success"] for result in latency_diagnostics) == model.n_iter_
    assert len(results) <= 2 * model.n_iter_
    assert results[-1].success
    for index, result in enumerate(results):
        if not result.success:
            assert result.status == 2
            assert "ABNORMAL" in str(result.message).upper()
            assert results[index + 1].success
    assert model.__sklearn_is_fitted__()
    assert np.isfinite(model.final_loglik_)
    assert np.isfinite(model.predict(X)).all()
    assert np.all(np.diff(model.loglik_history_) >= -1e-9)
