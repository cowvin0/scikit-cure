"""Public fitting diagnostics, solver controls, and latent-count semantics."""

from copy import deepcopy
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sksurv.util import Surv

from skcure._em import run_em
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


def _data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    y = Surv.from_arrays(
        event=np.tile([True, False, True, True], 6),
        time=np.round(rng.uniform(0.4, 3.2, size=24), 3),
    )
    return X, y


def _state():
    return SimpleNamespace(n_iter=0, converged=False, loglik_history=[])


@pytest.mark.parametrize("estimator", [GGKM, MixtureCure, PromotionTimeCure])
def test_latent_mean_requires_a_fitted_estimator(estimator):
    with pytest.raises(NotFittedError):
        estimator().predict_latent_mean(np.ones((2, 2)))


def test_likelihood_decrease_does_not_claim_convergence():
    state = _state()
    with pytest.warns(ConvergenceWarning):
        _, likelihood = run_em(
            0,
            e_step=lambda parameters: None,
            m_steps=(lambda parameters, expected: parameters + 1,),
            loglikelihood=lambda parameters: -10.0 - 1e-8 * parameters,
            state=state,
            tol=1e-6,
            rtol=1e-3,
            max_iter=2,
        )
    assert not state.converged
    assert state.termination_reason == "max_em_iter"
    assert state.n_iter == 2
    assert likelihood == state.loglik_history[-1]
    assert np.all(np.diff(state.loglik_history) < 0.0)


@pytest.mark.parametrize("scale", [0.01, 1.0, 10000.0])
def test_relative_convergence_is_invariant_to_likelihood_scale(scale):
    state = _state()
    run_em(
        0,
        e_step=lambda parameters: None,
        m_steps=(lambda parameters, expected: parameters + 1,),
        loglikelihood=lambda parameters: scale * (-100.0 + 0.001 * parameters),
        state=state,
        tol=0.0,
        rtol=1e-4,
        max_iter=2,
    )
    assert state.n_iter == 1
    assert state.converged
    assert state.termination_reason == "converged"


@pytest.mark.parametrize("estimator", [GGKM, MixtureCure, PromotionTimeCure])
def test_cap_warns_and_publishes_optimizer_diagnostics(estimator):
    X, y = _data()
    model = estimator(max_em_iter=1)
    with pytest.warns(ConvergenceWarning, match="EM did not converge"):
        model.fit(X, y)
    check_is_fitted(model)
    assert model.termination_reason_ == "max_em_iter"
    assert not model.converged_
    assert [record["stage"] for record in model.optimizer_results_] == [
        "incidence",
        "latency",
    ]
    for record in model.optimizer_results_:
        assert record["iteration"] == 1
        assert record["success"]
        assert np.isfinite(record["objective"])
        assert record["message"]
        assert record["n_function_evaluations"] >= 1


def test_failed_optimizer_keeps_diagnostics_and_invalidates_estimator(monkeypatch):
    X, y = _data()
    model = PromotionTimeCure(max_em_iter=1)
    module = importlib.import_module("skcure._em")
    original = module.minimize

    def failure(**kwargs):
        result = original(**kwargs)
        result.success = False
        result.message = "intentional optimizer failure"
        return result

    monkeypatch.setattr(module, "minimize", failure)
    with pytest.raises(RuntimeError, match="intentional optimizer failure"):
        model.fit(X, y)
    assert model.termination_reason_ == "failed"
    assert not model.__sklearn_is_fitted__()
    assert model.loglik_history_ == []
    assert model.n_iter_ == 1
    assert len(model.optimizer_results_) == 1
    assert not model.optimizer_results_[0]["success"]
    assert model.optimizer_results_[0]["message"] == "intentional optimizer failure"


def test_per_stage_solver_options_reach_each_optimizer_without_mutation(monkeypatch):
    X, y = _data()
    settings = {
        "incidence": {"maxiter": 2000, "gtol": 1e-6},
        "latency": {"maxiter": 201, "gtol": 2e-6},
        "dispersion": {"maxiter": 202, "gtol": 3e-6},
    }
    before = deepcopy(settings)
    received = []
    for module_name in ("skcure._em", "skcure._latency", "skcure._latent"):
        module = importlib.import_module(module_name)
        original = module.minimize

        def traced(*, _original=original, **kwargs):
            received.append(deepcopy(kwargs["options"]))
            return _original(**kwargs)

        monkeypatch.setattr(module, "minimize", traced)
    model = GGKM(
        latent_distribution="nb",
        max_em_iter=1,
        lambda_reg=0.1,
        solver_options=settings,
    )
    assert model.solver_options is settings
    assert clone(model).get_params() == model.get_params()
    with pytest.warns(ConvergenceWarning):
        model.fit(X, y)
    assert settings == before
    assert received == list(settings.values())
    assert [item["stage"] for item in model.optimizer_results_] == list(settings)


@pytest.mark.parametrize("estimator", [GGKM, MixtureCure, PromotionTimeCure])
@pytest.mark.parametrize("rtol", [-1, np.nan, np.inf, True, "small"])
def test_invalid_relative_tolerances_are_rejected(estimator, rtol):
    X, y = _data()
    with pytest.raises(ValueError, match="em_rtol"):
        estimator(em_rtol=rtol).fit(X, y)


@pytest.mark.parametrize(
    "settings,error",
    [
        ([], TypeError),
        ({"typo": {}}, ValueError),
        ({"incidence": []}, TypeError),
        ({"latency": {"gtoll": 1e-7}}, ValueError),
        ({"dispersion": {"gtol": -1}}, ValueError),
        ({"incidence": {"ftol": np.nan}}, ValueError),
        ({"latency": {"maxiter": True}}, ValueError),
        ({"latency": {"maxls": 0}}, ValueError),
    ],
)
def test_solver_option_validation(settings, error):
    X, y = _data()
    with pytest.raises(error, match="solver|incidence|latency|dispersion"):
        GGKM(solver_options=settings).fit(X, y)


@pytest.mark.parametrize(
    "model,multiplier,binomial",
    [
        (GGKM(max_em_iter=1), 1, False),
        (GGKM(latent_distribution="nb", phi=0.3, max_em_iter=1), 1, False),
        (GGKM(latent_distribution="binomial", K_bin=4, max_em_iter=1), 4, True),
        (MixtureCure(max_em_iter=1), 1, True),
        (PromotionTimeCure(max_em_iter=1), 1, False),
    ],
)
def test_latent_mean_has_consistent_unconditional_count_semantics(
    model, multiplier, binomial
):
    X, y = _data()
    with pytest.warns(ConvergenceWarning):
        model.fit(X, y)
    eta = model.predict_eta(X)
    expected = multiplier * (expit(eta) if binomial else np.exp(eta))
    np.testing.assert_allclose(model.predict_latent_mean(X), expected)
    np.testing.assert_array_equal(model.predict(X), eta)
