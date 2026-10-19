"""Evaluation metrics specific to cure models."""

from __future__ import annotations

import numpy as np


def cure_tpr_fpr(pi_hat, c_grid=None):
    """Compute the cure-model ROC coordinates over probability thresholds.

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


def auc_cure(pi_hat, c_grid=None):
    """Compute the area under the cure-model ROC curve."""
    c_grid, tpr, fpr = cure_tpr_fpr(
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


__all__ = [
    "auc_cure",
    "cure_tpr_fpr",
]
