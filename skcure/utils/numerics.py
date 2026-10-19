import numpy as np


def _exp_log_quantity(log_quantity):
    """Compute exp(log_quantity), preserving GGKM's legacy overflow guards."""
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
    log_x = np.asarray(log_x, dtype=float)

    result = np.empty_like(
        log_x,
        dtype=float,
    )

    log_max = np.log(np.finfo(float).max)
    log_min = np.log(np.nextafter(0.0, 1.0))

    finite = np.isfinite(log_x)
    low = finite & (log_x < log_min)
    high = finite & (log_x > log_max)
    middle = finite & ~low & ~high

    result[low] = 0.0
    result[high] = np.inf
    result[middle] = np.exp(log_x[middle])

    result[np.isneginf(log_x)] = 0.0
    result[np.isposinf(log_x)] = np.inf
    result[np.isnan(log_x)] = np.nan

    return result
