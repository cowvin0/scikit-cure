"""Training budgets, validation isolation, and ensemble-prefix predictions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sksurv.util import Surv

from skcure.ensemble import GradientBoostingCure


@pytest.fixture
def data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(64, 3))
    event_time = rng.exponential(np.exp(-0.5 * X[:, 0])) + 0.05
    censoring = rng.uniform(0.5, 3.0, len(X))
    return X, Surv.from_arrays(event_time <= censoring, np.minimum(event_time, censoring))


def short_model(**parameters):
    defaults = dict(
        n_estimators=2,
        max_em_iter=8,
        max_total_estimators=3,
        min_samples_leaf=3,
        random_state=73,
        tol=1e-30,
    )
    defaults.update(parameters)
    return GradientBoostingCure(**defaults)


def test_tree_budget_caps_across_em_updates(data):
    model = short_model().fit(*data)
    assert model.n_estimators_ == 3
    assert len(model.models_) == len(model.model_weights_) == 3
    assert model.n_iter_ == 2
    assert model.termination_reason_ == "max_total_estimators"
    assert not model.converged_
    assert np.isfinite(model.final_loglik_)
    assert model.validation_loglik_history_ == []
    assert model.best_validation_loglik_ is None
    assert model.best_iteration_ == model.n_iter_
    np.testing.assert_array_equal(model.training_indices_, np.arange(len(data[0])))
    assert not model.validation_indices_.size


def test_unlimited_budget_retains_per_em_tree_limit_and_warns(data):
    with pytest.warns(ConvergenceWarning, match="2 iterations"):
        model = short_model(max_total_estimators=None, max_em_iter=2).fit(*data)
    assert model.n_estimators_ == 4
    assert model.termination_reason_ == "max_em_iter"


def test_new_parameters_follow_sklearn_clone_contract():
    options = {"incidence": {"maxiter": 40}, "latency": {"gtol": 1e-7}}
    model = short_model(
        n_iter_no_change=3,
        validation_fraction=0.25,
        validation_tol=0.02,
        em_rtol=1e-8,
        solver_options=options,
    )
    assert model.solver_options is options
    assert clone(model).get_params() == model.get_params()
    assert GradientBoostingCure().max_total_estimators == 1000
    assert model.set_params(max_total_estimators=None).max_total_estimators is None


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("max_total_estimators", 0),
        ("max_total_estimators", True),
        ("max_total_estimators", 2.5),
        ("n_iter_no_change", 0),
        ("n_iter_no_change", False),
        ("n_iter_no_change", 1.5),
        ("validation_fraction", 0),
        ("validation_fraction", 1),
        ("validation_fraction", np.nan),
        ("validation_fraction", True),
        ("validation_tol", -1),
        ("validation_tol", np.inf),
        ("validation_tol", True),
    ],
)
def test_invalid_boosting_controls_fail_at_fit(parameter, value, data):
    model = short_model(**{parameter: value})
    with pytest.raises(ValueError, match=parameter):
        model.fit(*data)
    assert model.termination_reason_ == "failed"
    with pytest.raises(NotFittedError):
        model.predict(data[0])


@pytest.mark.parametrize(
    "distribution,extra",
    [("poisson", {}), ("binomial", {"K_bin": 3}), ("nb", {"phi": 0.5})],
)
def test_staged_predictions_end_at_retained_model(distribution, extra, data):
    X, y = data
    model = short_model(latent_distribution=distribution, **extra).fit(X, y)
    eta = list(model.staged_predict_eta(X))
    risk = list(model.staged_predict(X))
    survival = list(model.staged_predict_survival(X, [0.0, 0.5, 2.0]))
    assert len(eta) == len(risk) == len(survival) == model.n_estimators_
    np.testing.assert_array_equal(eta[-1], model.predict_eta(X))
    np.testing.assert_array_equal(risk[-1], model.predict(X))
    np.testing.assert_array_equal(survival[-1], model.predict_survival(X, [0.0, 0.5, 2.0]))
    for stage in survival:
        assert stage.shape == (len(X), 3)
        assert np.all(stage[:, 0] == 1)
        assert np.all(np.diff(stage, axis=1) <= 0)
    original = eta[-1].copy()
    eta[0][:] = 123.0
    np.testing.assert_array_equal(eta[-1], original)
    np.testing.assert_array_equal(model.predict_eta(X), original)


@pytest.mark.parametrize(
    "method,args",
    [("staged_predict_eta", ()), ("staged_predict", ()),
     ("staged_predict_survival", ([0.0, 1.0],))],
)
def test_staged_predictions_require_fitted_model(method, args):
    with pytest.raises(NotFittedError):
        list(getattr(GradientBoostingCure(), method)(np.ones((3, 2)), *args))


def test_validation_early_stopping_restores_best_complete_nb_model(data, monkeypatch):
    X, y = data
    model = short_model(
        latent_distribution="negative_binomial",
        phi=None,
        max_total_estimators=None,
        n_iter_no_change=1,
        validation_fraction=0.25,
        validation_tol=0.0,
    )
    snapshots = []
    scores = iter([0.0, 2.0, 1.0])

    def validation_score(*args):
        snapshots.append((
            len(model.models_),
            (model.a_, model.d_, model.p_),
            model._latent_model.phi,
            model.eta_train_.copy(),
        ))
        return next(scores)

    monkeypatch.setattr(model, "_validation_loglikelihood", validation_score)
    model.fit(X, y)
    trees, gamma, dispersion, eta = snapshots[1]
    assert model.n_iter_ == 2
    assert model.best_iteration_ == 1
    assert model.n_estimators_ == trees == 2
    np.testing.assert_array_equal((model.a_, model.d_, model.p_), gamma)
    assert model.phi_ == dispersion
    np.testing.assert_array_equal(model.eta_train_, eta)
    assert model.final_loglik_ == model._loglikelihood(
        y["time"][model.training_indices_], y["event"][model.training_indices_].astype(float)
    )
    assert model.validation_loglik_history_ == [2.0, 1.0]
    assert model.initial_validation_loglik_ == 0.0
    assert model.best_validation_loglik_ == 2.0
    assert model.termination_reason_ == "validation_early_stopping"
    assert not model.converged_


def test_validation_can_restore_initial_zero_tree_model(data, monkeypatch):
    model = short_model(n_iter_no_change=1, max_total_estimators=None)
    scores = iter([0.0, -1.0])
    monkeypatch.setattr(model, "_validation_loglikelihood", lambda *args: next(scores))
    model.fit(*data)
    assert model.best_iteration_ == 0
    assert model.n_estimators_ == 0
    np.testing.assert_array_equal(model.predict(data[0]), np.zeros(len(data[0])))
    assert list(model.staged_predict(data[0])) == []
    assert list(model.staged_predict_survival(data[0], [0.0, 1.0])) == []


def test_real_validation_score_matches_returned_best_model(data):
    X, y = data
    model = short_model(
        n_iter_no_change=2, validation_fraction=0.25,
        validation_tol=1e6, max_total_estimators=None,
    ).fit(X, y)
    assert model.n_iter_ == 2
    validation = model.validation_indices_
    score = model._validation_loglikelihood(
        X[validation], y["time"][validation], y["event"][validation].astype(float)
    )
    assert score == model.best_validation_loglik_
    assert score == max([model.initial_validation_loglik_, *model.validation_loglik_history_])


def test_validation_rows_never_enter_training_updates(data, monkeypatch):
    X, y = data
    model = short_model(n_iter_no_change=1, validation_fraction=0.25, validation_tol=1e6)
    original_boosting = model._boosting_m_step
    original_latency = model._gg_m_step
    boosting_rows, latency_times = [], []

    def boosting(X, Nhat):
        boosting_rows.append(X.copy())
        return original_boosting(X, Nhat)

    def latency(t, delta, Nhat):
        latency_times.append(t.copy())
        return original_latency(t, delta, Nhat)

    monkeypatch.setattr(model, "_boosting_m_step", boosting)
    monkeypatch.setattr(model, "_gg_m_step", latency)
    model.fit(X, y)
    training, validation = model.training_indices_, model.validation_indices_
    assert len(training) == 48
    assert len(validation) == 16
    assert not np.intersect1d(training, validation).size
    np.testing.assert_array_equal(np.sort(np.r_[training, validation]), np.arange(len(X)))
    assert y["event"][training].any() and y["event"][validation].any()
    for rows, times in zip(boosting_rows, latency_times):
        np.testing.assert_array_equal(rows, X[training])
        np.testing.assert_array_equal(times, y["time"][training])
    np.testing.assert_array_equal(model.unique_times_, np.unique(y["time"][training]))


def test_validation_partition_and_fit_are_reproducible(data):
    options = dict(n_iter_no_change=1, validation_tol=1e6)
    first = short_model(**options).fit(*data)
    second = short_model(**options).fit(*data)
    np.testing.assert_array_equal(first.training_indices_, second.training_indices_)
    np.testing.assert_array_equal(first.validation_indices_, second.validation_indices_)
    np.testing.assert_array_equal(first.predict(data[0]), second.predict(data[0]))


def test_validation_fit_preserves_named_feature_checks(data):
    X, y = data
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    model = short_model(n_iter_no_change=1, validation_tol=1e6).fit(frame, y)
    np.testing.assert_array_equal(model.feature_names_in_, frame.columns)
    assert model.n_features_in_ == 3
    assert np.isfinite(model.predict(frame)).all()
    with pytest.raises(ValueError, match="feature"):
        model.predict(frame[["b", "a", "c"]])


def test_validation_requires_enough_events(data):
    X, y = data
    y = y.copy()
    y["event"] = False
    y["event"][0] = True
    with pytest.raises(ValueError, match="at least two observed events"):
        short_model(n_iter_no_change=1).fit(X, y)


def test_small_validation_split_keeps_event_in_each_subset(data):
    X, y = data
    y = y.copy()
    y["event"] = False
    y["event"][:2] = True
    model = short_model(
        n_iter_no_change=1, validation_fraction=1 / len(y),
        validation_tol=1e6,
    ).fit(X, y)
    assert y["event"][model.training_indices_].sum() == 1
    assert y["event"][model.validation_indices_].sum() == 1


def test_binomial_latent_mean_uses_fitted_trial_count(data):
    X, y = data
    model = short_model(latent_distribution="binomial", K_bin=4).fit(X, y)
    np.testing.assert_allclose(model.predict_latent_mean(X), 4 * model.predict_theta(X))
    model.set_params(K_bin=9, latent_distribution="poisson")
    np.testing.assert_allclose(model.predict_latent_mean(X), 4 * model.predict_theta(X))


def test_latent_mean_requires_fit():
    with pytest.raises(NotFittedError):
        GradientBoostingCure().predict_latent_mean(np.ones((2, 2)))


def test_solver_diagnostics_cover_incidence_and_latency(data):
    model = short_model(solver_options={"incidence": {"maxiter": 10}}).fit(*data)
    stages = {result["stage"] for result in model.optimizer_results_}
    assert stages >= {"incidence", "latency"}
    assert all(result["iteration"] in {1, 2} for result in model.optimizer_results_)
