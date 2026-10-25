"""Additional vectorized kernels not provided by scikit-learn."""

from math import isfinite
from numbers import Real

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn.metrics.pairwise import check_pairwise_arrays

from skcure._dataframe import ensure_eager_dataframe

__all__ = ["exponential_kernel", "cauchy_kernel", "wavelet_kernel"]


def exponential_kernel(X, Y=None, *, gamma=1.0):
    """Return ``exp(-gamma * ||x - y||_2)`` for each pair of samples."""
    X, Y = check_pairwise_arrays(
        ensure_eager_dataframe(X), ensure_eager_dataframe(Y), dtype="float64"
    )
    return np.exp(-gamma * cdist(X, Y, metric="euclidean"))


def cauchy_kernel(X, Y=None, *, gamma=1.0):
    """Return ``1 / (1 + gamma * ||x - y||_2**2)`` for each sample pair."""
    X, Y = check_pairwise_arrays(
        ensure_eager_dataframe(X), ensure_eager_dataframe(Y), dtype="float64"
    )
    return 1.0 / (1.0 + gamma * cdist(X, Y, metric="sqeuclidean"))


def wavelet_kernel(X, Y=None, *, A=10.0):
    """Return the product wavelet kernel with positive scale ``A``.

    Each feature contributes ``cos(1.75 * (x - y) / A)`` and the product
    is weighted by ``exp(-||x - y||_2**2 / (2 * A**2))``.
    """
    X, Y = check_pairwise_arrays(
        ensure_eager_dataframe(X), ensure_eager_dataframe(Y), dtype="float64"
    )
    if not isinstance(A, Real) or not isfinite(A) or A <= 0.0:
        raise ValueError("A must be positive and finite.")
    K = np.ones((len(X), len(Y)), dtype=float)
    for j in range(X.shape[1]):
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            diff = np.subtract.outer(X[:, j], Y[:, j])
            overflowing_difference = np.isinf(diff)
            diff /= A
            if np.any(overflowing_difference):
                scaled_difference = np.subtract.outer(X[:, j] / A, Y[:, j] / A)
                diff[overflowing_difference] = scaled_difference[overflowing_difference]
            component = np.exp(-0.5 * diff * diff)
            nonzero = component > 0.0
            component[nonzero] *= np.cos(1.75 * diff[nonzero])
            K *= component
    return K


def _validate_kernel_hyperparameters(gamma, degree, coef0, A, *, kernel=None):
    """Validate GGKM's kernel settings without modifying constructor values."""
    if not isinstance(gamma, Real) or not isfinite(gamma):
        raise ValueError("gamma must be a finite real number.")
    if not isinstance(degree, (int, np.integer)) or isinstance(
        degree, (bool, np.bool_)
    ):
        raise TypeError("degree must be a positive integer.")
    if degree < 1:
        raise ValueError("degree must be a positive integer.")
    if not isinstance(coef0, Real) or not isfinite(coef0):
        raise ValueError("coef0 must be a finite real number.")
    if not isinstance(A, Real) or not isfinite(A) or A <= 0.0:
        raise ValueError("A must be positive and finite.")
    gamma_kernels = {
        "rbf",
        "gaussian",
        "poly",
        "polynomial",
        "sigmoid",
        "laplacian",
        "exponential",
        "cauchy",
        "chi2",
    }
    if isinstance(kernel, str) and kernel in gamma_kernels and gamma < 0.0:
        raise ValueError(f"gamma must be non-negative for kernel={kernel!r}.")
    if isinstance(kernel, str) and kernel in {"poly", "polynomial"} and coef0 < 0.0:
        raise ValueError("coef0 must be non-negative for polynomial kernels.")


def _validate_positive_semidefinite(K):
    """Reject indefinite training kernels, allowing floating-point roundoff."""
    scale = float(np.max(np.abs(K)))
    if scale == 0.0:
        return
    normalized = K / scale
    normalized = 0.5 * normalized + 0.5 * normalized.T
    minimum = float(
        eigh(normalized, eigvals_only=True, subset_by_index=[0, 0], check_finite=False)[
            0
        ]
    )
    tolerance = 100.0 * np.finfo(float).eps * len(K)
    if minimum < -tolerance:
        raise ValueError(
            "The training kernel must be positive semidefinite; "
            "an indefinite kernel does not define a valid kernel penalty."
        )


def _validate_kernel_matrix(K, expected_shape):
    """Reject malformed or non-finite custom kernel outputs."""
    K = np.asarray(K, dtype=float)
    if K.shape != expected_shape:
        raise ValueError(
            f"Kernel must return a matrix with shape {expected_shape}, got {K.shape}."
        )
    if np.any(~np.isfinite(K)):
        raise FloatingPointError("Kernel matrix contains non-finite values.")
    return K
