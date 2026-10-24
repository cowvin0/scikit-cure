"""Observed-data survival evaluation and model-implied cure diagnostics.

Right censoring does not reveal which event-free individuals are cured. The
``model_implied_*`` diagnostics use predicted probabilities as their own class
weights and therefore do not measure accuracy against observed outcomes.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_consistent_length
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import check_y_survival


def model_implied_cure_roc(pi_hat, c_grid=None):
    """Compute model-implied ROC coordinates using predictions as class weights.

    This measures the separation implied by a model's own probabilities. It
    requires no outcomes and cannot establish discrimination or calibration
    against true cure status. In particular, extreme probabilities can produce
    a high value even when the predictions are incorrect.

    Parameters
    ----------
    pi_hat : array-like of shape (n_samples,)
        Estimated cure probabilities.

    c_grid : array-like of shape (n_thresholds,), optional
        Increasing cure-probability thresholds. If None, thresholds are
        constructed from the unique predicted cure probabilities together
        with 0 and 1.

    Returns
    -------
    c_grid : ndarray
        Threshold grid.

    tpr : ndarray
        Weighted true-positive rates.

    fpr : ndarray
        Weighted false-positive rates.
    """
    pi_hat = np.asarray(pi_hat, dtype=float)

    if pi_hat.ndim != 1:
        raise ValueError("pi_hat must be one-dimensional")

    if pi_hat.size == 0:
        raise ValueError("pi_hat cannot be empty")

    if not np.all(np.isfinite(pi_hat)):
        return None, None, None

    numerical_tolerance = 2.0e-12

    if np.any(pi_hat < -numerical_tolerance) or np.any(
        pi_hat > 1.0 + numerical_tolerance
    ):
        raise ValueError("pi_hat must contain probabilities in [0, 1]")

    pi_hat = np.clip(
        pi_hat,
        0.0,
        1.0,
    )

    if c_grid is None:
        c_grid = np.unique(
            np.concatenate(
                (
                    [0.0],
                    pi_hat,
                    [1.0],
                )
            )
        )
    else:
        c_grid = np.asarray(
            c_grid,
            dtype=float,
        )

        if c_grid.ndim != 1:
            raise ValueError("c_grid must be one-dimensional")

        if not np.all(np.isfinite(c_grid)):
            raise ValueError("c_grid must contain only finite values")

        if np.any(c_grid < 0.0) or np.any(c_grid > 1.0):
            raise ValueError("c_grid must lie in [0, 1]")

        if np.any(np.diff(c_grid) < 0.0):
            raise ValueError("c_grid must be sorted increasingly")

    w_pos = 1.0 - pi_hat
    w_neg = pi_hat

    denom_tpr = np.sum(w_pos)
    denom_fpr = np.sum(w_neg)

    if denom_tpr <= 0.0 or denom_fpr <= 0.0:
        return (
            c_grid,
            np.full_like(
                c_grid,
                np.nan,
                dtype=float,
            ),
            np.full_like(
                c_grid,
                np.nan,
                dtype=float,
            ),
        )

    tpr = np.empty_like(
        c_grid,
        dtype=float,
    )
    fpr = np.empty_like(
        c_grid,
        dtype=float,
    )

    for j, c in enumerate(c_grid):
        selected = pi_hat <= c

        tpr[j] = np.sum(selected * w_pos) / denom_tpr
        fpr[j] = np.sum(selected * w_neg) / denom_fpr

    return c_grid, tpr, fpr


def model_implied_cure_auc(pi_hat, c_grid=None):
    """Return the area under the model-implied cure ROC, not observed accuracy.

    Returns ``(area, tpr, fpr, thresholds)``. Predictions supply both scores and
    class weights; use :func:`evaluate_survival` for observed-data evaluation.
    """
    c_grid, tpr, fpr = model_implied_cure_roc(
        pi_hat,
        c_grid=c_grid,
    )

    if tpr is None or fpr is None:
        return np.nan, None, None, None

    if np.any(~np.isfinite(tpr)) or np.any(~np.isfinite(fpr)):
        return np.nan, tpr, fpr, c_grid

    auc = float(
        np.trapezoid(
            tpr,
            fpr,
        )
    )

    return auc, tpr, fpr, c_grid


def cure_tpr_fpr(pi_hat, c_grid=None):
    """Compatibility alias for :func:`model_implied_cure_roc`.

    These coordinates use predicted cure probabilities as class weights;
    they do not compare predictions with observed cure labels.
    """
    return model_implied_cure_roc(pi_hat, c_grid=c_grid)


def auc_cure(pi_hat, c_grid=None):
    """Compatibility alias for :func:`model_implied_cure_auc`.

    This is a model-implied diagnostic, not observed predictive accuracy.
    """
    return model_implied_cure_auc(pi_hat, c_grid=c_grid)


@dataclass(frozen=True)
class SurvivalEvaluation:
    """Observed-data metrics on a fixed evaluation time interval.

    Larger ``concordance_ipcw``/``auc`` are better; smaller
    ``integrated_brier_score`` is better. None of these identify individual
    cure status from right-censored observations.
    """

    concordance_ipcw: float
    auc: np.ndarray
    mean_auc: float
    integrated_brier_score: float
    times: np.ndarray


def _prediction_estimator(estimator: Any, X: Any) -> tuple[Any, Any]:
    """Expose cure prediction methods after applying fitted pipeline transforms."""
    while isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            X = estimator[:-1].transform(X)
        estimator = estimator.steps[-1][1]
    return estimator, X


def evaluate_survival(
    estimator: Any,
    X_test: Any,
    y_test: np.ndarray,
    *,
    y_train: np.ndarray,
    times: Any,
    tau: float | None = None,
) -> SurvivalEvaluation:
    """Evaluate a fitted cure estimator against observed right-censored outcomes.

    ``y_train`` estimates the censoring distribution and must come from the
    training population, not from the held-out evaluation population. IPCW
    assumes independent censoring; this helper does not adjust for censoring
    that depends on covariates. The caller supplies a strictly increasing grid
    of at least two times inside the test follow-up interval. The training
    censoring distribution must cover all test event times and the grid.

    Concordance uses ``estimator.predict(X_test)`` (larger means more risk).
    Time-dependent AUC uses ``1 - survival``; the Brier score uses survival
    probabilities themselves. Test observations are never silently discarded
    or truncated. ``tau`` truncates concordance only; it does not change the
    AUC or Brier evaluation populations.

    References
    ----------
    https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html
    """
    _, train_time = check_y_survival(y_train)
    event, test_time = check_y_survival(y_test)
    check_consistent_length(X_test, y_test)
    grid = np.asarray(times, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(grid < 0.0)
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("times must contain at least two finite, non-negative, strictly increasing values.")
    if grid[0] < test_time.min() or grid[-1] >= test_time.max():
        raise ValueError("times must lie in [min(test time), max(test time)).")
    if np.any(np.sum(event[:, None] & (test_time[:, None] <= grid), axis=0) == 0):
        raise ValueError("Each evaluation time needs at least one observed test event at or before it.")
    if tau is not None and (not np.isfinite(float(tau)) or tau <= 0):
        raise ValueError("tau must be positive and finite.")
    if grid[-1] >= train_time.max() or np.max(test_time[event]) >= train_time.max() or test_time.max() > train_time.max():
        raise ValueError(
            "Training follow-up must cover test follow-up and extend beyond evaluation times and all test event times; "
            "choose a supported held-out cohort rather than silently dropping observations."
        )
    censoring = CensoringDistributionEstimator().fit(y_train)
    support_times = np.concatenate((grid, test_time[event]))
    try:
        censoring_probability = np.asarray(censoring.predict_proba(support_times, return_conf_int=False), dtype=float)
    except ValueError as exc:
        raise ValueError("The training censoring distribution does not cover the evaluation data.") from exc
    if np.any(censoring_probability <= 0.0):
        raise ValueError("The training censoring probability must be positive at evaluation and test event times.")

    prediction_model, prediction_X = _prediction_estimator(estimator, X_test)
    risk = np.asarray(estimator.predict(X_test), dtype=float)
    survival = np.asarray(prediction_model.predict_survival(prediction_X, grid), dtype=float)
    if risk.shape != (len(y_test),) or not np.all(np.isfinite(risk)):
        raise ValueError("predict must return one finite risk score per test observation.")
    if survival.shape != (len(y_test), len(grid)) or not np.all(np.isfinite(survival)):
        raise ValueError("predict_survival must return a finite (n_test, n_times) array.")
    if np.any((survival < 0.0) | (survival > 1.0)) or np.any(np.diff(survival, axis=1) > 1e-12):
        raise ValueError("Survival probabilities must lie in [0, 1] and be non-increasing in time.")
    concordance = concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0]
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, 1.0 - survival, grid)
    brier = integrated_brier_score(y_train, y_test, survival, grid)
    return SurvivalEvaluation(float(concordance), auc, float(mean_auc), float(brier), grid.copy())


__all__ = [
    "auc_cure",
    "cure_tpr_fpr",
    "model_implied_cure_auc",
    "model_implied_cure_roc",
    "evaluate_survival",
    "SurvivalEvaluation",
]
