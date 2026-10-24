"""Shape derivatives of the regularized incomplete gamma CDF.

The default implementation uses differentiated series and continued fractions.
JAX is an explicitly requested optional backend and is never imported as a
side effect of importing or fitting the ordinary cure estimators.
"""

from __future__ import annotations

from importlib import import_module
from math import copysign, exp, log, log1p, pi, sqrt

import numpy as np
from scipy.integrate import quad

from skcure.distributions.generalized_gamma import (
    _gamma_lower_log_and_shape_derivative,
    _gamma_upper_tail_terms,
)


def _large_shape_derivative(x: float, shape: float) -> float:
    """Integrate the gamma shape score in standardized coordinates.

    Differentiating the normalized gamma density gives
    dP(s, x)/ds = integral_0^x f_s(t) * (log(t) - digamma(s)) dt.
    Near a very large shape the lower series needs O(sqrt(shape)) terms.
    The standardized density is instead concentrated on a fixed interval.
    Centering its log density with Stirling's expansion avoids subtracting
    quantities of order shape*log(shape). Scaling the shape score by sqrt(shape)
    also keeps quadrature tolerances useful when the derivative is tiny.
    """
    root_shape = sqrt(shape)
    boundary = (x - shape) / root_shape
    if abs(boundary) >= 50.0:
        # Beyond this interval the derivative is below the float64 range for
        # shape >= 1e5, including the small skewness correction of the gamma.
        return 0.0
    inverse = 1.0 / shape
    center_log_density = -0.5 * np.log(2.0 * pi) - inverse / 12.0
    center_log_density += inverse**3 / 360.0 - inverse**5 / 1260.0
    scaled_center_score = root_shape * (
        inverse / 2.0 + inverse**2 / 12.0 - inverse**4 / 120.0
    )

    def log_density_and_score(z):
        relative = z / root_shape
        log_relative = log1p(relative)
        if abs(relative) < 0.001:
            log_remainder = relative * relative * (
                -0.5 + relative * (
                    1.0 / 3.0 + relative * (
                        -0.25 + relative * (
                            0.2 + relative * (-1.0 / 6.0 + relative / 7.0)
                        )
                    )
                )
            )
        else:
            log_remainder = log_relative - relative
        log_density = center_log_density + shape * log_remainder - log_relative
        scaled_score = root_shape * log_relative + scaled_center_score
        return log_density, scaled_score

    log_scale = log_density_and_score(boundary)[0]

    def integrand(z):
        log_density, scaled_score = log_density_and_score(z)
        # Normalize by the density at the integration boundary. Without this,
        # an absolute quadrature tolerance can accept a poorly resolved but
        # representable derivative in a tail such as z=20 or z=38.
        return exp(log_density - log_scale) * scaled_score

    if boundary <= 0.0:
        integral = quad(integrand, -50.0, boundary, epsabs=1e-13, epsrel=5e-12)[0]
    else:
        # The complete shape-score integral is zero. Integrating the shorter
        # upper tail avoids cancellation when the CDF is close to one.
        integral = -quad(integrand, boundary, 50.0, epsabs=1e-13, epsrel=5e-12)[0]
    if integral == 0.0:
        return 0.0
    return copysign(exp(log_scale + log(abs(integral)) - log(root_shape)), integral)


def _scipy_shape_derivative(u: np.ndarray, shape: float) -> np.ndarray:
    result = np.full_like(u, np.nan, dtype=float)
    endpoints = (u == 0.0) | np.isposinf(u)
    result[endpoints] = 0.0
    interior = np.isfinite(u) & (u > 0.0)
    if not np.any(interior):
        return result
    if shape >= 1e5:
        result[interior] = [
            _large_shape_derivative(float(x), shape) for x in u[interior]
        ]
        return result
    lower = interior & (u < shape + 1.0)
    upper = interior & ~lower
    with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
        if np.any(lower):
            log_probability, log_derivative = _gamma_lower_log_and_shape_derivative(
                shape, u[lower], np.log(u[lower])
            )
            result[lower] = np.sign(log_derivative) * np.exp(
                log_probability + np.log(np.abs(log_derivative))
            )
        if np.any(upper):
            log_probability, derivative_ratio, _ = _gamma_upper_tail_terms(
                shape, u[upper], np.log(u[upper])
            )
            result[upper] = np.sign(derivative_ratio) * np.exp(
                log_probability + np.log(np.abs(derivative_ratio))
            )
    return np.where((u == 0.0) | np.isposinf(u), 0.0, result)


def pgamma_shape_derivative_vec(u, shape, *, backend="scipy") -> np.ndarray:
    """Return d P(shape, u) / d shape, preserving the shape of ``u``.

    ``shape`` must be one positive finite scalar. Arguments zero and positive
    infinity have derivative zero; negative arguments and NaNs return NaN, as
    with SciPy's gamma CDF. The default ``"scipy"`` backend requires no JAX.
    ``backend="jax"`` lazily loads the optional dependency and uses a temporary
    float64 context without changing the application's global precision setting.
    """
    if np.ndim(shape) != 0:
        raise ValueError("shape must be a positive finite scalar.")
    try:
        shape = float(shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must be a positive finite scalar.") from exc
    if not np.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be a positive finite scalar.")
    u = np.asarray(u, dtype=float)
    if backend == "scipy":
        return _scipy_shape_derivative(u, shape)
    if backend != "jax":
        raise ValueError("backend must be 'scipy' or 'jax'.")
    try:
        # JAX is an optional extra and is absent from the core typing job.
        import jax  # pyright: ignore[reportMissingImports]
        import jax.numpy as jnp  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "The JAX backend requires the optional 'jax' dependency. "
            "Install skcure[jax] to use backend='jax'."
        ) from exc
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = getattr(import_module("jax.experimental"), "enable_x64")
    with enable_x64(True):
        value = jax.lax.igamma_grad_a(
            jnp.asarray(shape, dtype=jnp.float64),
            jnp.asarray(u, dtype=jnp.float64),
        )
        result = np.asarray(value, dtype=float)
    return np.where((u == 0.0) | np.isposinf(u), 0.0, result)


def pgamma_shape_derivative(x, shape, *, backend="scipy") -> float:
    """Scalar counterpart of :func:`pgamma_shape_derivative_vec`."""
    return float(pgamma_shape_derivative_vec(x, shape, backend=backend))
