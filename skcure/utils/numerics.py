import numpy as np


def _exp_log_quantity(log_quantity):
    """Exponentiate a log quantity without losing NaNs or overflow state."""
    log_quantity = np.asarray(
        log_quantity,
        dtype=float,
    )
    result = np.zeros_like(
        log_quantity,
        dtype=float,
    )
    max_log = _maximum_log_float()
    positive_infinity = np.isposinf(log_quantity)
    too_large = log_quantity > max_log
    regular = np.isfinite(log_quantity) & ~too_large
    result[positive_infinity | too_large] = np.inf
    result[regular] = np.exp(log_quantity[regular])
    result[np.isnan(log_quantity)] = np.nan

    return result


def _exp_minus_exp(log_quantity):
    """Compute exp(-exp(log_quantity)), preserving GGKM's legacy guards."""
    log_quantity = np.asarray(
        log_quantity,
        dtype=float,
    )

    result = np.empty_like(
        log_quantity,
        dtype=float,
    )

    max_log = _maximum_log_float()
    very_large = log_quantity > max_log
    result[very_large] = 0.0
    regular = ~very_large
    inner = np.exp(log_quantity[regular])
    result[regular] = np.exp(-inner)

    return result


def _maximum_log_float() -> float:
    """Largest representable floating-point value on the log scale."""
    return float(np.log(np.finfo(float).max))


def _safe_exp(log_x):
    """Exponentiate values while handling floating-point overflow/underflow."""
    with np.errstate(under="ignore"):
        return _exp_log_quantity(log_x)
