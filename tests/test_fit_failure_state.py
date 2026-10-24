from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sksurv.util import Surv

from skcure._latent import _NegativeBinomialLatentModel
from skcure.distributions import GeneralizedGamma
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, len(X)), 3),
    )
    return X, y


@pytest.fixture(
    scope="module", params=[GGKM, MixtureCure, PromotionTimeCure, GradientBoostingCure]
)
def fitted_model(request, data):
    parameters = dict(max_em_iter=1)
    if request.param is GradientBoostingCure:
        parameters.update(n_estimators=1, min_samples_leaf=3, random_state=73)
    with pytest.warns(ConvergenceWarning):
        return request.param(**parameters).fit(*data)


@pytest.mark.parametrize(
    "failure", ["parameter", "solver_options", "nonfinite_features", "no_events"]
)
def test_failed_refit_clears_previous_diagnostics(fitted_model, data, failure):
    model = deepcopy(fitted_model)
    X, y = data
    assert model.optimizer_results_
    assert model.loglik_history_
    if failure == "parameter":
        model.set_params(max_em_iter=0)
    elif failure == "solver_options":
        model.set_params(solver_options={"latency": {"maxiter": 0}})
    elif failure == "nonfinite_features":
        X = X.copy()
        X[0, 0] = np.nan
    else:
        y = y.copy()
        y["event"] = False
    with pytest.raises(ValueError):
        model.fit(X, y)
    assert model.termination_reason_ == "failed"
    assert model.optimizer_results_ == []
    assert model.loglik_history_ == []
    assert model.n_iter_ == 0
    assert not model.converged_
    assert model.final_loglik_ is None
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        model.predict(data[0])


def test_linear_initialization_failure_reports_failed_attempt(data, monkeypatch):
    model = MixtureCure()

    def invalid_design(X):
        raise RuntimeError("invalid design")

    monkeypatch.setattr(model, "_design_matrix", invalid_design)
    with pytest.raises(RuntimeError, match="invalid design"):
        model.fit(*data)
    assert model.termination_reason_ == "failed"
    assert model.optimizer_results_ == []
    assert model.n_iter_ == 0


def test_ggkm_kernel_failure_reports_failed_attempt(data):
    def invalid_kernel(x, y):
        raise RuntimeError("invalid kernel")

    model = GGKM(kernel=invalid_kernel)
    with pytest.raises(RuntimeError, match="invalid kernel"):
        model.fit(*data)
    assert model.termination_reason_ == "failed"
    assert model.optimizer_results_ == []
    assert model.n_iter_ == 0


@pytest.mark.parametrize("log_coordinates", [False, True])
def test_successful_dispersion_optimizer_cannot_worsen_objective(
    log_coordinates, monkeypatch
):
    model = _NegativeBinomialLatentModel()
    const, log_A, delta = np.zeros(3), np.zeros(3), np.ones(3)
    before = model.phi_value_and_grad(np.array([model.phi]), const, log_A, delta)[0]
    candidate_phi = 100.0
    after = model.phi_value_and_grad(np.array([candidate_phi]), const, log_A, delta)[0]
    assert after > before
    monkeypatch.setattr(
        model, "dispersion_objective_terms_from_eta", lambda *args: (const, log_A)
    )
    diagnostics = []
    result = SimpleNamespace(
        success=True,
        fun=after,
        x=np.array([np.log(candidate_phi) if log_coordinates else candidate_phi]),
        message="optimizer reports success",
    )
    with patch("skcure._latent.minimize", return_value=result):
        method = (
            model.update_after_latency_in_log_space_from_eta
            if log_coordinates
            else model.update_after_latency_from_eta
        )
        with pytest.raises(FloatingPointError, match="worsening objective"):
            method(
                GeneralizedGamma(),
                np.zeros(3),
                np.ones(3),
                delta,
                result_callback=diagnostics.append,
            )
    assert model.phi == 0.5
    assert model.phi_ is None
    assert diagnostics == [result]


def test_dispersion_log_fallback_rejects_worsening_candidate_without_mutation(
    monkeypatch,
):
    model = _NegativeBinomialLatentModel()
    monkeypatch.setattr(
        model,
        "dispersion_objective_terms_from_eta",
        lambda *args: (np.zeros(3), np.zeros(3)),
    )
    raw = SimpleNamespace(success=False, fun=np.inf, x=np.array([0.5]), message="retry")
    fallback = SimpleNamespace(
        success=True, fun=13.98, x=np.array([np.log(100.0)]), message="stale success"
    )
    with patch("skcure._latent.minimize", side_effect=[raw, fallback]):
        with pytest.raises(FloatingPointError, match="worsening objective"):
            model.update_after_latency_from_eta(
                GeneralizedGamma(), np.zeros(3), np.ones(3), np.ones(3)
            )
    assert model.phi == 0.5


@pytest.mark.parametrize("estimator", [GGKM, GradientBoostingCure])
def test_worsening_dispersion_fails_fit_with_accessible_diagnostics(estimator, data):
    parameters = dict(latent_distribution="nb", max_em_iter=1)
    if estimator is GradientBoostingCure:
        parameters.update(n_estimators=1, min_samples_leaf=3, random_state=73)
    model = estimator(**parameters)

    def candidate(fun, **kwargs):
        coordinate = (
            np.log(100.0) if fun.__name__ == "log_phi_value_and_grad" else 100.0
        )
        return SimpleNamespace(
            success=True, fun=0.0, x=np.array([coordinate]), message="stale success"
        )

    with patch("skcure._latent.minimize", side_effect=candidate):
        with pytest.raises(FloatingPointError, match="worsening objective"):
            model.fit(*data)
    assert model.termination_reason_ == "failed"
    assert not model.__sklearn_is_fitted__()
    assert model._latent_model.phi == 0.5
    assert model.optimizer_results_[-1]["stage"] == "dispersion"
    assert model.optimizer_results_[-1]["message"] == "stale success"
    assert model.loglik_history_ == []


@pytest.mark.parametrize("log_coordinates", [False, True])
def test_dispersion_guard_allows_roundoff_at_same_candidate(
    log_coordinates, monkeypatch
):
    model = _NegativeBinomialLatentModel()
    calls = iter([3.0 + 4 * np.finfo(float).eps, 3.0])
    monkeypatch.setattr(
        model, "phi_value_and_grad", lambda *args: (next(calls), np.zeros(1))
    )
    monkeypatch.setattr(
        model,
        "dispersion_objective_terms_from_eta",
        lambda *args: (np.zeros(3), np.zeros(3)),
    )
    result = SimpleNamespace(
        success=True,
        fun=3.0,
        x=np.array([np.log(0.5) if log_coordinates else 0.5]),
        message="stationary",
    )
    with patch("skcure._latent.minimize", return_value=result):
        method = (
            model.update_after_latency_in_log_space_from_eta
            if log_coordinates
            else model.update_after_latency_from_eta
        )
        method(GeneralizedGamma(), np.zeros(3), np.ones(3), np.ones(3))
    assert model.phi == 0.5
