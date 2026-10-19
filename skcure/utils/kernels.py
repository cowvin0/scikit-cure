"""Additional vectorized kernels not provided by scikit-learn."""

from numbers import Real

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import check_pairwise_arrays

__all__ = ["exponential_kernel", "cauchy_kernel", "wavelet_kernel"]


def exponential_kernel(X, Y=None, *, gamma=1.0):
    """Return ``exp(-gamma * ||x - y||_2)`` for each pair of samples."""
    X, Y = check_pairwise_arrays(X, Y, dtype=float)
    return np.exp(-gamma * cdist(X, Y, metric="euclidean"))


def cauchy_kernel(X, Y=None, *, gamma=1.0):
    """Return ``1 / (1 + gamma * ||x - y||_2**2)`` for each sample pair."""
    X, Y = check_pairwise_arrays(X, Y, dtype=float)
    return 1.0 / (1.0 + gamma * cdist(X, Y, metric="sqeuclidean"))


def wavelet_kernel(X, Y=None, *, A=10.0):
    """Return the product wavelet kernel with positive scale ``A``.

    Each feature contributes ``cos(1.75 * (x - y) / A)`` and the product
    is weighted by ``exp(-||x - y||_2**2 / (2 * A**2))``.
    """
    X, Y = check_pairwise_arrays(X, Y, dtype=float)
    K = np.exp(-0.5 * cdist(X, Y, metric="sqeuclidean") / (A**2))
    for j in range(X.shape[1]):
        diff = np.subtract.outer(X[:, j], Y[:, j])
        diff *= 1.75 / A
        np.cos(diff, out=diff)
        K *= diff
    return K


def _validate_kernel_hyperparameters(gamma, degree, coef0, A):
    """Validate GGKM's kernel settings without modifying constructor values."""
    if not isinstance(gamma, Real) or not np.isfinite(gamma):
        raise ValueError("gamma must be a finite real number.")
    if not isinstance(degree, (int, np.integer)) or isinstance(degree, (bool, np.bool_)):
        raise TypeError("degree must be a positive integer.")
    if degree < 1:
        raise ValueError("degree must be a positive integer.")
    if not isinstance(coef0, Real) or not np.isfinite(coef0):
        raise ValueError("coef0 must be a finite real number.")
    if not isinstance(A, Real) or not np.isfinite(A) or A <= 0.0:
        raise ValueError("A must be positive and finite.")


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
