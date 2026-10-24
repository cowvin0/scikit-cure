from contextlib import contextmanager
from copy import deepcopy
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_checks
from sksurv.util import Surv

from skcure.deep_learning import DeepPromotionTimeCure
from skcure.ensemble import GradientBoostingCure
from skcure.ggkm import GGKM
from skcure.linear_model import MixtureCure, PromotionTimeCure


@contextmanager
def _expected_fit_warning(estimator):
    """Require exactly the warning justified by the completed EM fit state."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "always",
            message=r"EM did not converge within \d+ iterations\.",
            category=ConvergenceWarning,
        )
        yield
    if estimator.converged_:
        assert estimator.termination_reason_ == "converged"
        assert not captured
    else:
        assert estimator.termination_reason_ == "max_em_iter"
        assert estimator.n_iter_ == estimator.max_em_iter
        assert len(captured) == 1
        assert issubclass(captured[0].category, ConvergenceWarning)
        assert str(captured[0].message).startswith(
            f"EM did not converge within {estimator.max_em_iter} iterations."
        )


def _survival_target(y):
    """Preserve real survival targets; monotonically map generic labels to time."""
    values = np.asarray(y)
    if values.dtype.names is not None:
        return values
    if values.ndim != 1:
        raise ValueError("Survival targets must be one-dimensional.")
    values = np.asarray(values, dtype=float)
    times = values - np.min(values) + 1.0
    return Surv.from_arrays(np.ones(len(values), dtype=bool), times)


class _SurvivalTargetChecks:
    def fit(self, X, y):
        with _expected_fit_warning(self):
            return getattr(super(), "fit")(X, _survival_target(y))

    def score(self, X, y):
        return getattr(super(), "score")(X, _survival_target(y))


class _CheckedGGKM(_SurvivalTargetChecks, GGKM):
    pass


class _CheckedMixtureCure(_SurvivalTargetChecks, MixtureCure):
    pass


class _CheckedPromotionTimeCure(_SurvivalTargetChecks, PromotionTimeCure):
    pass


class _CheckedGradientBoostingCure(_SurvivalTargetChecks, GradientBoostingCure):
    pass


PUBLIC_ESTIMATORS = [
    pytest.param(GGKM(), id="ggkm"),
    pytest.param(MixtureCure(), id="mixture"),
    pytest.param(PromotionTimeCure(), id="promotion-time"),
    pytest.param(GradientBoostingCure(), id="boosting"),
    pytest.param(DeepPromotionTimeCure(), id="deep-promotion-time"),
]

CHECK_ESTIMATORS = [
    pytest.param(_CheckedGGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1), id="ggkm"),
    pytest.param(_CheckedMixtureCure(max_em_iter=1), id="mixture"),
    pytest.param(_CheckedPromotionTimeCure(max_em_iter=1), id="promotion-time"),
    pytest.param(
        _CheckedGradientBoostingCure(
            n_estimators=1,
            max_em_iter=1,
            min_samples_leaf=3,
            random_state=73,
        ),
        id="boosting",
    ),
]

CONSTRUCTION_CHECKS = [
    estimator_checks.check_estimator_cloneable,
    estimator_checks.check_no_attributes_set_in_init,
    estimator_checks.check_parameters_default_constructible,
    estimator_checks.check_get_params_invariance,
    estimator_checks.check_set_params,
    estimator_checks.check_estimators_unfitted,
]

FITTING_CHECKS = [
    estimator_checks.check_fit_score_takes_y,
    estimator_checks.check_estimators_overwrite_params,
    estimator_checks.check_n_features_in_after_fitting,
    estimator_checks.check_estimators_dtypes,
    estimator_checks.check_fit2d_predict1d,
    estimator_checks.check_methods_sample_order_invariance,
    estimator_checks.check_methods_subset_invariance,
    estimator_checks.check_dict_unchanged,
    estimator_checks.check_estimators_pickle,
    estimator_checks.check_fit_idempotent,
]


@pytest.mark.parametrize("estimator", PUBLIC_ESTIMATORS)
@pytest.mark.parametrize("check", CONSTRUCTION_CHECKS, ids=lambda check: check.__name__)
def test_public_estimators_pass_upstream_construction_checks(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", CHECK_ESTIMATORS)
@pytest.mark.parametrize("check", FITTING_CHECKS, ids=lambda check: check.__name__)
def test_upstream_fitting_checks_with_survival_targets(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(GGKM(gamma=0.3, lambda_reg=0.1, max_em_iter=1), id="ggkm"),
        pytest.param(MixtureCure(max_em_iter=1), id="mixture"),
        pytest.param(PromotionTimeCure(max_em_iter=1), id="promotion-time"),
        pytest.param(
            GradientBoostingCure(
                n_estimators=1,
                max_em_iter=1,
                min_samples_leaf=3,
                random_state=73,
            ),
            id="boosting",
        ),
    ],
)
def test_real_survival_pipeline_selection_preserves_data_and_feature_contract(
    estimator,
):
    rng = np.random.default_rng(73)
    X = pd.DataFrame(rng.normal(size=(24, 2)), columns=["first", "second"])
    y = Surv.from_arrays(
        np.tile([True, False, True, True], 6),
        np.round(rng.uniform(0.4, 3.2, size=24), 3),
    )
    original_X, original_y = X.copy(deep=True), y.copy()
    parameters = deepcopy(estimator.get_params())
    pipeline = Pipeline([("scale", StandardScaler()), ("cure", estimator)])
    search = GridSearchCV(
        pipeline,
        {"cure__tol": [1e-6, 1e-5]},
        cv=KFold(n_splits=2, shuffle=True, random_state=73),
    )
    search.set_params(error_score="raise")
    # Individual CV clones are hidden inside GridSearchCV; capture their cap
    # warnings at this call and require all other warnings to remain errors.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "always",
            message=r"EM did not converge within 1 iterations\.",
            category=ConvergenceWarning,
        )
        search.fit(X, y)
    assert captured
    assert all(issubclass(item.category, ConvergenceWarning) for item in captured)
    assert np.all(np.isfinite(search.cv_results_["mean_test_score"]))
    assert np.all(np.isfinite(search.predict(X)))
    assert 0 <= np.asarray(search.score(X, y), dtype=float).item() <= 1
    pd.testing.assert_frame_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    assert estimator.get_params() == parameters
    with pytest.raises(NotFittedError):
        estimator.predict(X)
    np.testing.assert_array_equal(search.feature_names_in_, X.columns)
    with pytest.raises(ValueError, match="feature names"):
        search.predict(X[["second", "first"]])
    unfitted_pipeline = clone(search.best_estimator_)
    assert isinstance(unfitted_pipeline, Pipeline)
    with pytest.raises(NotFittedError):
        unfitted_pipeline.predict(X)
