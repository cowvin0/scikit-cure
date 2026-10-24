"""The public EM iteration parameter has one contract across cure estimators."""

import inspect

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sksurv.util import Surv

from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure

ESTIMATORS = [GGKM, GradientBoostingCure, MixtureCure, PromotionTimeCure]


@pytest.fixture(scope="module")
def survival_data():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(24, 2))
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, size=24), 2),
    )
    return X, y


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_em_iteration_default_is_an_explicit_public_parameter(estimator):
    assert inspect.signature(estimator).parameters["max_em_iter"].default == 1000
    model = estimator()
    assert model.max_em_iter == 1000
    assert clone(model).get_params()["max_em_iter"] == 1000


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_numpy_integer_iteration_limit_is_preserved_and_obeyed(
    estimator, survival_data
):
    max_em_iter = np.int64(2)
    parameters = {"max_em_iter": max_em_iter, "tol": np.finfo(float).tiny}
    if estimator is GradientBoostingCure:
        parameters.update(n_estimators=1, min_samples_leaf=3, random_state=73)
    model = estimator(**parameters)
    original_parameters = model.get_params()

    with pytest.warns(
        ConvergenceWarning, match="EM did not converge within 2 iterations"
    ):
        assert model.fit(*survival_data) is model
    assert model.n_iter_ == max_em_iter
    assert model.max_em_iter is max_em_iter
    assert model.get_params() == original_parameters
    copied = clone(model)
    assert isinstance(copied.max_em_iter, np.integer)
    assert copied.max_em_iter == max_em_iter
    assert copied.get_params() == original_parameters


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("max_em_iter", [None, 0, -1, 1.5, True, np.bool_(True), "2"])
def test_invalid_em_iteration_limits_are_rejected(
    estimator, max_em_iter, survival_data
):
    model = estimator(max_em_iter=max_em_iter)
    with pytest.raises((TypeError, ValueError), match="max_em_iter"):
        model.fit(*survival_data)
