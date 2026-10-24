"""Gamma shape derivatives without import-time JAX or global configuration."""

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import gammainc, gammaincc

from skcure.utils.pgamma_derivate import (
    pgamma_shape_derivative,
    pgamma_shape_derivative_vec,
)


@pytest.mark.parametrize("shape", [0.01, 0.1, 1.0, 10.0, 100.0, 1e4, 1e5, 1e6, 1e8])
def test_derivative_matches_independent_five_point_cdf_differences(shape):
    # Scale the stencil by the width of the gamma distribution rather than
    # its mean. A relative-to-shape step is too wide for large gamma shapes.
    argument = np.maximum(
        shape + np.array([-3.0, -1.0, 0.0, 1.0, 3.0]) * np.sqrt(shape),
        shape * 0.1,
    )
    step = min(0.01 * shape, 0.001 * np.sqrt(shape))
    expected = (
        -gammainc(shape + 2.0 * step, argument)
        + 8.0 * gammainc(shape + step, argument)
        - 8.0 * gammainc(shape - step, argument)
        + gammainc(shape - 2.0 * step, argument)
    ) / (12.0 * step)
    actual = pgamma_shape_derivative_vec(argument, shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-14)
    assert np.all(actual < 0.0)


@pytest.mark.parametrize("shape", [1e5, 1e6, 1e8])
@pytest.mark.parametrize("standardized_argument", [10.0, 20.0, 30.0, 38.0])
def test_large_shape_quadrature_resolves_representable_upper_tail_derivatives(
    shape, standardized_argument
):
    argument = shape + standardized_argument * np.sqrt(shape)
    step = 0.0001 * np.sqrt(shape)
    # Use the upper CDF directly: the lower CDF is already rounded to one.
    expected = (
        gammaincc(shape + 2.0 * step, argument)
        - 8.0 * gammaincc(shape + step, argument)
        + 8.0 * gammaincc(shape - step, argument)
        - gammaincc(shape - 2.0 * step, argument)
    ) / (12.0 * step)
    actual = pgamma_shape_derivative(argument, shape)
    assert actual < 0.0
    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-322)


@pytest.mark.parametrize("shape", [0.01, 1.0, 100.0, 1e8])
def test_endpoints_invalid_arguments_and_array_shapes(shape):
    argument = np.array([[0.0, np.inf, np.nan], [-1.0, -np.inf, shape]])
    result = pgamma_shape_derivative_vec(argument, shape)
    assert result.shape == argument.shape
    np.testing.assert_array_equal(result[0, :2], [0.0, 0.0])
    assert np.isnan(result[0, 2])
    assert np.isnan(result[1, 0])
    assert np.isnan(result[1, 1])
    assert result[1, 2] < 0.0
    assert pgamma_shape_derivative_vec([], shape).shape == (0,)


def test_scalar_and_vector_interfaces_agree():
    value = pgamma_shape_derivative(2.5, 1.3)
    assert isinstance(value, float)
    assert value == pgamma_shape_derivative_vec(2.5, 1.3)


@pytest.mark.parametrize("shape", [0, -1, np.nan, np.inf, [1, 2], "invalid", None])
def test_invalid_shapes_are_rejected(shape):
    with pytest.raises(ValueError, match="shape must be a positive finite scalar"):
        pgamma_shape_derivative_vec([1.0], shape)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        pgamma_shape_derivative_vec([1.0], 1.0, backend="unknown")


def test_importing_library_and_default_derivatives_does_not_import_jax():
    script = """
import sys
import skcure
from skcure.utils.pgamma_derivate import pgamma_shape_derivative_vec
assert 'jax' not in sys.modules
assert pgamma_shape_derivative_vec([0.1, 1.0, 10.0], 1.0).shape == (3,)
assert 'jax' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_optional_jax_matches_default_and_preserves_global_precision_setting():
    jax = pytest.importorskip("jax")
    before = jax.config.jax_enable_x64
    arguments = np.array([0.0, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, np.inf])
    for shape in (0.01, 1.0, 5.0, 100.0):
        result = pgamma_shape_derivative_vec(arguments, shape, backend="jax")
        expected = pgamma_shape_derivative_vec(arguments, shape)
        np.testing.assert_allclose(result, expected, rtol=2e-11, atol=2e-14)
        assert jax.config.jax_enable_x64 == before


def test_missing_optional_jax_has_an_actionable_error(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def block_jax(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("optional dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_jax)
    with pytest.raises(ImportError, match=r"Install skcure\[jax\]"):
        pgamma_shape_derivative_vec([1.0], 1.0, backend="jax")
    assert np.isfinite(pgamma_shape_derivative(1.0, 1.0))
