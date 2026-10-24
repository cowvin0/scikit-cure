"""Observed-data evaluation uses the correct prediction scale and population."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

from skcure.metrics import (
    auc_cure, cure_tpr_fpr, evaluate_survival,
    model_implied_cure_auc, model_implied_cure_roc,
)


class ExponentialPrediction(BaseEstimator):
    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]

    def predict_survival(self, X, times):
        return np.exp(-np.exp(self.predict(X))[:, None] * np.asarray(times)[None, :])


def _cohorts():
    train = Surv.from_arrays([True, False, True, True, False, True, False], [0.3, 0.8, 1.1, 1.5, 2.8, 3.5, 5.0])
    test = Surv.from_arrays([True, True, False, True, False, True], [0.5, 1, 1.5, 2.0, 3.0, 4.0])
    X = np.array([[1], [.8], [.2], [0], [-.4], [-.6]])
    return train, test, X, np.array([0.7, 1.7, 2.7])


def test_observed_metrics_equal_sksurv_and_use_time_dependent_event_risk():
    train, test, X, times = _cohorts()
    estimator = ExponentialPrediction()
    result = evaluate_survival(estimator, X, test, y_train=train, times=times)
    survival = estimator.predict_survival(X, times)
    assert result.concordance_ipcw == concordance_index_ipcw(train, test, estimator.predict(X))[0]
    expected_auc, expected_mean = cumulative_dynamic_auc(train, test, 1 - survival, times)
    assert_allclose(result.auc, expected_auc)
    assert result.mean_auc == expected_mean
    assert result.integrated_brier_score == integrated_brier_score(train, test, survival, times)
    inverted_auc, _ = cumulative_dynamic_auc(train, test, survival, times)
    assert np.all(result.auc > inverted_auc)


def test_evaluation_supports_preprocessing_pipelines():
    train, test, X, times = _cohorts()
    model = make_pipeline(StandardScaler(), ExponentialPrediction()).fit(X, test)
    result = evaluate_survival(model, X, test, y_train=train, times=times)
    transformed = model[0].transform(X)
    direct = evaluate_survival(model[-1], transformed, test, y_train=train, times=times)
    assert_allclose(result.auc, direct.auc)
    assert result.integrated_brier_score == direct.integrated_brier_score


@pytest.mark.parametrize("times", [[.1, 1], [1, 4], [2, 1], [1], [1, 1], [1, np.nan]])
def test_invalid_grid_is_rejected(times):
    train, test, X, _ = _cohorts()
    with pytest.raises(ValueError, match="times"):
        evaluate_survival(ExponentialPrediction(), X, test, y_train=train, times=times)


def test_missing_training_followup_is_not_silently_truncated():
    train, test, X, times = _cohorts()
    train = train[:-2]
    with pytest.raises(ValueError, match="Training follow-up"):
        evaluate_survival(ExponentialPrediction(), X, test, y_train=train, times=times, tau=2)


def test_nonmonotone_survival_is_rejected():
    train, test, X, times = _cohorts()
    class Invalid(ExponentialPrediction):
        def predict_survival(self, X, times):
            return np.tile([.2, .4, .1], (len(X), 1))
    with pytest.raises(ValueError, match="non-increasing"):
        evaluate_survival(Invalid(), X, test, y_train=train, times=times)


def test_old_diagnostic_names_preserve_results_and_explicit_names_do_not_imply_accuracy():
    probabilities = [.001, .001, .999, .999]
    assert_allclose(auc_cure(probabilities)[0], .999)
    for old, new in ((auc_cure, model_implied_cure_auc), (cure_tpr_fpr, model_implied_cure_roc)):
        for left, right in zip(old(probabilities), new(probabilities)):
            assert_allclose(left, right)

