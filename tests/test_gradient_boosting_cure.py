"""Unified ensemble API and latent-family fit lifecycle."""

import inspect
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sksurv.util import Surv

import skcure
import skcure.ensemble
from skcure.base import BaseCureEstimator
from skcure.ensemble import GradientBoostingCure


LEGACY = json.loads((Path(__file__).parent / "data" / "boost_legacy.json").read_text())
SHORT_FIT = dict(n_estimators=2, max_em_iter=1, min_samples_leaf=3, random_state=73)
FAMILIES = [
    pytest.param("poisson", {}, id="poisson"),
    pytest.param("binomial", {"K_bin": 1}, id="bernoulli"),
    pytest.param("binomial", {"K_bin": 3}, id="binomial"),
    pytest.param("negative_binomial", {"phi": 0.7}, id="nb-fixed"),
    pytest.param("negative_binomial", {"phi": None}, id="nb-estimated"),
]


@pytest.fixture(scope="module")
def survival_data():
    return np.asarray(LEGACY["X"]), Surv.from_arrays(LEGACY["event"], LEGACY["time"])


def test_public_estimator_contract():
    a = np.float64(1.2)
    n_estimators = np.int64(2)
    model = GradientBoostingCure(
        latent_distribution="NB", a=a, n_estimators=n_estimators
    )
    assert isinstance(model, BaseCureEstimator)
    assert model.latent_distribution == "NB"
    assert model.get_params()["a"] is a
    assert model.get_params()["n_estimators"] is n_estimators
    assert not any(name.endswith("_") for name in vars(model))
    assert list(inspect.signature(model.fit).parameters) == ["X", "y"]
    assert clone(model).get_params() == model.get_params()
    assert model.set_params(latent_distribution="binomial", K_bin=3) is model
    for package in (skcure, skcure.ensemble):
        assert package.GradientBoostingCure is GradientBoostingCure
        assert "GradientBoostingCure" in package.__all__
    assert GradientBoostingCure.__module__.startswith("skcure.ensemble.")
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


@pytest.mark.parametrize("distribution,parameters", FAMILIES)
def test_fitted_family_supports_clone_and_pickle(
    distribution, parameters, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution=distribution, **parameters, **SHORT_FIT
    ).fit(X, y)
    if distribution == "negative_binomial":
        assert model.phi_ > 0
        if parameters["phi"] is not None:
            assert model.phi_ == parameters["phi"]
        assert model.estimate_phi_ is (parameters["phi"] is None)
    else:
        with pytest.raises(AttributeError):
            _ = model.phi_
        with pytest.raises(AttributeError):
            _ = model.estimate_phi_
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    times = np.asarray(LEGACY["prediction_times"])
    np.testing.assert_array_equal(
        restored.predict_survival(X, times), model.predict_survival(X, times)
    )
    np.testing.assert_array_equal(
        restored.predict_cure_probability(X), model.predict_cure_probability(X)
    )
    assert restored.get_params() == model.get_params()
    with pytest.raises(NotFittedError):
        check_is_fitted(clone(model))


@pytest.mark.parametrize("case_name", list(LEGACY["cases"]))
def test_unified_tree_updates_match_independent_legacy_fixture(
    case_name, monkeypatch, survival_data
):
    X, y = survival_data
    case = LEGACY["cases"][case_name]
    distribution = {
        "GGBinomialGB": "binomial",
        "GGNBGB": "negative_binomial",
        "GGPoissonGB": "poisson",
    }[case["class_name"]]
    model = GradientBoostingCure(
        latent_distribution=distribution, **case["params"], **LEGACY["fit_params"]
    )
    monkeypatch.setattr(model, "_gg_m_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_phi_m_step", lambda *args, **kwargs: None)
    model.fit(X, y)
    expected = case["fixed_gg_steps"]
    np.testing.assert_allclose(model.predict_eta(X), expected["eta"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        model.model_weights_, expected["weights"], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        model.loglik_history_, expected["loglik_history"], rtol=0, atol=1e-6
    )


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("Poisson", "poisson"),
        (" poisson ", "poisson"),
        ("BINOMIAL", "binomial"),
        ("NB", "negative_binomial"),
        ("negative-binomial", "negative_binomial"),
        (" Negative Binomial ", "negative_binomial"),
        ("negativebinomial", "negative_binomial"),
    ],
)
def test_distribution_aliases_preserve_constructor_parameter(
    alias, canonical, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(latent_distribution=alias, **SHORT_FIT).fit(X, y)
    expected = GradientBoostingCure(
        latent_distribution=canonical, **SHORT_FIT
    ).fit(X, y)
    assert model.latent_distribution == alias
    assert clone(model).latent_distribution == alias
    np.testing.assert_array_equal(
        model.predict_cure_probability(X), expected.predict_cure_probability(X)
    )


@pytest.mark.parametrize(
    "distribution,parameters",
    [
        ("poisson", {"phi": -1, "K_bin": 0}),
        ("binomial", {"phi": -1, "K_bin": 3}),
        ("negative_binomial", {"phi": 0.7, "K_bin": 0}),
    ],
)
def test_parameters_for_other_latent_families_are_ignored(
    distribution, parameters, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution=distribution, **parameters, **SHORT_FIT
    ).fit(X, y)
    assert np.isfinite(model.predict(X)).all()
    for name, value in parameters.items():
        assert model.get_params()[name] == value


@pytest.mark.parametrize("distribution", ["unknown", "bernoulli", None])
def test_unknown_distribution_fails_at_fit(distribution, survival_data):
    model = GradientBoostingCure(latent_distribution=distribution, **SHORT_FIT)
    with pytest.raises(ValueError, match="latent_distribution"):
        model.fit(*survival_data)
    with pytest.raises(NotFittedError):
        model.predict(survival_data[0])


@pytest.mark.parametrize("phi", [0.0, -1.0, np.nan, np.inf, "invalid"])
def test_active_negative_binomial_dispersion_validation(phi, survival_data):
    model = GradientBoostingCure(
        latent_distribution="nb", phi=phi, **SHORT_FIT
    )
    with pytest.raises(ValueError, match="phi"):
        model.fit(*survival_data)


@pytest.mark.parametrize("K_bin", [0, -1, 1.5, True, np.bool_(True)])
def test_active_binomial_size_validation(K_bin, survival_data):
    model = GradientBoostingCure(
        latent_distribution="binomial", K_bin=K_bin, **SHORT_FIT
    )
    with pytest.raises(ValueError, match="K_bin"):
        model.fit(*survival_data)


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
def test_unified_predictions_require_fitting(method, args):
    with pytest.raises(NotFittedError):
        getattr(GradientBoostingCure(), method)(np.ones((2, 2)), *args)


def test_prediction_uses_fitted_family_after_set_params(survival_data):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution="nb", phi=0.7, **SHORT_FIT
    ).fit(X, y)
    times = np.array([0.0, 0.5, 2.0])
    before = (
        model.predict_theta(X),
        model.predict_survival(X, times),
        model.predict_cure_probability(X),
    )
    model.set_params(latent_distribution="binomial", K_bin=3, phi=0.9)
    after = (
        model.predict_theta(X),
        model.predict_survival(X, times),
        model.predict_cure_probability(X),
    )
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old, new)
    model.fit(X, y)
    expected = GradientBoostingCure(
        latent_distribution="binomial", K_bin=3, **SHORT_FIT
    ).fit(X, y)
    np.testing.assert_array_equal(
        model.predict_survival(X, times), expected.predict_survival(X, times)
    )
    with pytest.raises(AttributeError):
        _ = model.phi_


@pytest.mark.parametrize("warm_start", [False, True])
def test_warm_start_reuses_latency_and_dispersion_but_rebuilds_trees(
    warm_start, monkeypatch, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution="nb", warm_start=warm_start, **SHORT_FIT
    ).fit(X, y)
    gamma = (model.a_, model.d_, model.p_)
    phi = model.phi_
    trees = list(model.models_)
    monkeypatch.setattr(model, "_gg_m_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_phi_m_step", lambda *args, **kwargs: None)
    model.fit(X, y)
    np.testing.assert_array_equal(
        (model.a_, model.d_, model.p_), gamma if warm_start else (1.0, 1.0, 1.0)
    )
    assert model.phi_ == (phi if warm_start else 0.5)
    assert len(model.models_) == len(trees)
    assert all(new is not old for new, old in zip(model.models_, trees))


def test_warm_start_resets_dispersion_when_latent_configuration_changes(
    monkeypatch, survival_data
):
    X, y = survival_data
    model = GradientBoostingCure(
        latent_distribution="nb", warm_start=True, **SHORT_FIT
    ).fit(X, y)
    monkeypatch.setattr(model, "_gg_m_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_phi_m_step", lambda *args, **kwargs: None)
    model.set_params(phi=0.7).fit(X, y)
    assert model.phi_ == 0.7
    assert model.estimate_phi_ is False
    model.set_params(phi=None).fit(X, y)
    assert model.phi_ == 0.5
    assert model.estimate_phi_ is True
    model.set_params(latent_distribution="poisson").fit(X, y)
    with pytest.raises(AttributeError):
        _ = model.phi_
    model.set_params(latent_distribution="nb").fit(X, y)
    assert model.phi_ == 0.5


def test_unified_estimator_supports_pipeline_and_family_grid_search(survival_data):
    X, y = survival_data
    frame = pd.DataFrame(X, columns=["first", "second"])
    pipeline = make_pipeline(StandardScaler(), GradientBoostingCure(**SHORT_FIT))
    search = GridSearchCV(
        pipeline,
        {"gradientboostingcure__latent_distribution": ["poisson", "binomial"]},
        cv=KFold(2),
        error_score="raise",
    ).fit(frame, y)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert np.isfinite(search.predict(frame)).all()
    direct = GradientBoostingCure(**SHORT_FIT).fit(frame, y)
    np.testing.assert_array_equal(direct.feature_names_in_, frame.columns)
    with pytest.raises(ValueError, match="feature"):
        direct.predict(frame[["second", "first"]])


def test_none_em_limit_resolves_to_unified_default(survival_data):
    model = GradientBoostingCure(
        max_em_iter=None, tol=1e100, n_estimators=1, min_samples_leaf=3
    ).fit(*survival_data)
    assert model.get_params()["max_em_iter"] is None
    assert model._max_em_iter == 1000
    assert model.n_iter_ == 1


def test_invalid_family_on_refit_invalidates_previous_fit(survival_data):
    X, y = survival_data
    model = GradientBoostingCure(**SHORT_FIT).fit(X, y)
    model.set_params(latent_distribution="unknown")
    with pytest.raises(ValueError, match="latent_distribution"):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        model.predict(X)

