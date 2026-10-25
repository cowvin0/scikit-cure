import numpy as np


def _safe_exp(log_x) -> np.ndarray:
    """Exponentiate in float64, preserving NaNs, infinities, and subnormals.

    Expected overflow and underflow produce infinity and rounded tiny values
    respectively. Callers validate whether those results are admissible.
    """
    log_x = np.asarray(log_x, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        return np.asarray(np.exp(log_x))


def _exp_minus_exp(log_quantity) -> np.ndarray:
    """Compute exp(-exp(x)), allowing expected rounding to zero or one.

    Inner overflow maps to a zero probability. Underflow in either exponential
    is expected in the tails; NaNs and representable subnormals are preserved.
    """
    log_quantity = np.asarray(log_quantity, dtype=float)
    with np.errstate(over="ignore", under="ignore"):
        return np.asarray(np.exp(-np.exp(log_quantity)))


def _maximum_log_float() -> float:
    """Largest representable floating-point value on the log scale."""
    return float(np.log(np.finfo(float).max))
